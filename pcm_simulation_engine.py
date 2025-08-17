"""
PCM placement optimizer (updated)

- User provides: latitude & longitude, building element properties, PCM properties (including
  PCM mass available per m^2), area, and the target energy savings (kWh).
- The code computes the PCM thickness from either:
    a) the minimum thickness required to store the requested energy (using PCM latent heat), or
    b) the thickness available from the user-specified PCM mass per m^2 (mass / density).
  If available mass < required mass, the simulation uses the available mass (thickness) and reports
  that the target cannot be fully met (but still returns the best section + thickness).
- The optimizer brute-force tests candidate sections (Wall, Roof, Floor) and PCM positions (relative),
  uses the computed thickness per candidate, runs the transient 1D PCM simulation, estimates energy
  saved and picks the optimal recommendation.
- Output: a single one-line recommendation:
    - either "<Section>, <thickness_m> m"  (optimal section and thickness)
    - or "No PCM needed." when the simulation finds negligible benefit or target is essentially zero.

Usage: python pcm_placement_optimizer.py
Requires: requests, numpy, matplotlib (matplotlib is optional; only used for an optional final plot)
"""

import numpy as np
import requests
import sys
import math

# If you want plots, uncomment these imports and the plotting section at the bottom
import matplotlib.pyplot as plt

# --------------------------- Simulation core (adapted) ---------------------------

def transient_pcm_wall_simulation(
    wall_thickness_m,
    wall_k,
    wall_rho,
    wall_cp,
    pcm_thickness_m,
    pcm_k,
    pcm_rho,
    pcm_cp,
    pcm_latent_heat,  # J/kg
    pcm_Tmelt,
    pcm_dT,
    T_indoor,
    T_outdoor_series,
    duration_hours,
    dx,
    dt,
    pcm_position_relative
):
    """Returns x (positions) and T_hist (hourly temperature profiles)"""
    if T_outdoor_series is None:
        T_outdoor_series = [18 + 6*np.sin(2*np.pi*t/24) for t in range(duration_hours)]

    steps_per_hour = int(max(1, 3600 / dt))
    total_steps = duration_hours * steps_per_hour

    # Spatial domain
    total_thickness = wall_thickness_m
    nx = max(3, int(max(1, round(total_thickness / dx))))
    x = np.linspace(0, total_thickness, nx)

    # Material layers (centered node properties)
    k = np.full(nx, wall_k, dtype=float)
    rho = np.full(nx, wall_rho, dtype=float)
    cp = np.full(nx, wall_cp, dtype=float)
    alpha = np.full(nx, wall_k / (wall_rho * wall_cp), dtype=float)

    # PCM placement
    pcm_start_m = wall_thickness_m * pcm_position_relative
    pcm_end_m = pcm_start_m + pcm_thickness_m

    # clamp PCM inside the wall
    if pcm_end_m > wall_thickness_m:
        pcm_thickness_m = max(0.0, wall_thickness_m - pcm_start_m)
        pcm_end_m = pcm_start_m + pcm_thickness_m

    pcm_start_node = int(max(0, np.floor(pcm_start_m / total_thickness * nx)))
    pcm_end_node = int(min(nx, np.ceil(pcm_end_m / total_thickness * nx)))

    if pcm_thickness_m > 0 and pcm_start_node < pcm_end_node:
        k[pcm_start_node:pcm_end_node] = pcm_k
        rho[pcm_start_node:pcm_end_node] = pcm_rho
        cp[pcm_start_node:pcm_end_node] = pcm_cp
        alpha[pcm_start_node:pcm_end_node] = pcm_k / (pcm_rho * pcm_cp)

    # Temperatures: initialize at indoor temp
    T = np.full(nx, T_indoor, dtype=float)
    T_hist = []

    for step in range(total_steps):
        hour = step // steps_per_hour
        T_out = T_outdoor_series[min(hour, len(T_outdoor_series)-1)]

        # Phase change: simple linearized heat capacity within pcm nodes
        if pcm_thickness_m > 0 and pcm_start_node < pcm_end_node:
            for i in range(pcm_start_node, pcm_end_node):
                if abs(T[i] - pcm_Tmelt) <= pcm_dT:
                    cp[i] = pcm_cp + pcm_latent_heat / (2.0 * pcm_dT)
                else:
                    cp[i] = pcm_cp
            alpha[pcm_start_node:pcm_end_node] = k[pcm_start_node:pcm_end_node] / (rho[pcm_start_node:pcm_end_node] * cp[pcm_start_node:pcm_end_node])

        T_new = T.copy()
        for i in range(1, nx-1):
            T_new[i] = T[i] + alpha[i] * dt / ( (x[1]-x[0])**2 ) * (T[i+1] - 2.0*T[i] + T[i-1])

        # Boundary conditions (Dirichlet): exterior at x=0 equals outdoor, interior at x=thickness equals indoor
        T_new[0] = T_out
        T_new[-1] = T_indoor
        T = T_new

        # store hourly
        if (step+1) % steps_per_hour == 0:
            T_hist.append(T.copy())

    return x, np.array(T_hist)

# --------------------------- Weather fetcher ---------------------------

def fetch_hourly_temperature_open_meteo(lat, lon, duration_hours):
    """Fetch hourly temperature_2m from Open-Meteo. Returns list of length >= duration_hours or raises."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m&temperature_unit=celsius&timezone=UTC&forecast_days=7"
        )
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        temps = data.get('hourly', {}).get('temperature_2m', [])
        temps = list(temps)
        if len(temps) == 0:
            raise RuntimeError("Open-Meteo returned no hourly temps")
        # extend cyclically if not long enough
        while len(temps) < duration_hours:
            temps += temps
        return temps[:duration_hours]
    except Exception as e:
        raise RuntimeError(f"Weather fetch failed: {e}")

# --------------------------- Energy estimation ---------------------------

def estimate_hourly_energy_into_indoor(T_hist, k_arr, dx, area, indoor_temp):
    """
    Estimate cumulative energy into the indoor space (J) integrated over the hourly snapshots in T_hist.
    We approximate heat flux at the interior boundary using the adjacent interior node:
      flux = k_local * (T_node - T_indoor) / (dx/2)
    where dx/2 approximates distance from node center to the boundary.
    """
    hours = T_hist.shape[0]
    nx = T_hist.shape[1]
    energy = 0.0
    # choose last internal computational node index as representative
    idx = max(0, nx-2)
    for h in range(hours):
        T_node = T_hist[h, idx]
        k_local = k_arr[idx]
        flux = k_local * (T_node - indoor_temp) / (dx/2.0)  # W/m2
        energy += flux * 3600.0 * area  # J over hour
    return energy

# --------------------------- Search + thickness logic ---------------------------

def compute_required_thickness_for_target(energy_target_J_per_m2, pcm_latent_heat_J_per_kg, pcm_rho):
    """
    energy_target_J_per_m2: J per m2 desired to be saved (target J total divided by area)
    pcm_latent_heat_J_per_kg: J/kg
    pcm_rho: kg/m3

    Returns required_thickness_m to store that energy in PCM (latent heat only).
    required_mass_kg_per_m2 = energy_target_J_per_m2 / pcm_latent_heat_J_per_kg
    thickness = mass_per_m2 / rho
    """
    if pcm_latent_heat_J_per_kg <= 0 or pcm_rho <= 0:
        return 0.0, 0.0
    required_mass_per_m2 = energy_target_J_per_m2 / pcm_latent_heat_J_per_kg
    required_thickness_m = required_mass_per_m2 / pcm_rho
    return required_thickness_m, required_mass_per_m2

def find_best_pcm_placement(
    base_params,
    T_outdoor_series,
    duration_hours,
    dx,
    dt,
    area_m2,
    search_positions=None
):
    """
    This version expects that thickness is computed (per candidate) from either required_thickness (to meet target)
    or from available mass (user-specified mass_per_m2). It returns the best one-line recommendation.
    Selection logic:
      - If any candidate meets or exceeds the target energy saved (within small tolerance), choose the candidate
        that meets the target with the smallest thickness (economical).
      - Else choose the candidate that achieves the largest energy saved (best possible under mass constraint).
    """
    if search_positions is None:
        search_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    sections = ['Wall', 'Roof', 'Floor']

    # baseline (no PCM)
    params_no_pcm = base_params.copy()
    params_no_pcm['pcm_thickness_m'] = 0.0

    x_no_pcm, T_hist_no_pcm = transient_pcm_wall_simulation(
        wall_thickness_m=params_no_pcm['wall_thickness_m'],
        wall_k=params_no_pcm['wall_k'],
        wall_rho=params_no_pcm['wall_rho'],
        wall_cp=params_no_pcm['wall_cp'],
        pcm_thickness_m=0.0,
        pcm_k=0,
        pcm_rho=0,
        pcm_cp=0,
        pcm_latent_heat=0,
        pcm_Tmelt=0,
        pcm_dT=0,
        T_indoor=params_no_pcm['T_indoor'],
        T_outdoor_series=T_outdoor_series,
        duration_hours=duration_hours,
        dx=dx,
        dt=dt,
        pcm_position_relative=0.0
    )
    k_no_pcm = np.full(len(x_no_pcm), params_no_pcm['wall_k'])
    E_no_pcm_J = estimate_hourly_energy_into_indoor(T_hist_no_pcm, k_no_pcm, dx, area_m2, params_no_pcm['T_indoor'])

    # user target and available mass
    target_total_J = base_params['energy_target_J']  # target energy to be saved, total (not per m2)
    mass_available_kg_per_m2 = base_params['pcm_mass_kg_per_m2']

    # energy target per m2
    energy_target_J_per_m2 = target_total_J / max(area_m2, 1e-12)

    required_thickness_m, required_mass_per_m2 = compute_required_thickness_for_target(
        energy_target_J_per_m2,
        base_params['pcm_latent_heat'],
        base_params['pcm_rho']
    )

    # thickness the user has available (from mass)
    available_thickness_m = mass_available_kg_per_m2 / base_params['pcm_rho'] if base_params['pcm_rho'] > 0 else 0.0

    # tolerances
    tiny_energy_tol_J = 1e-3  # negligibly small energy

    results = []

    for sect in sections:
        for pos in search_positions:
            # Decide thickness to use:
            # If available mass >= required_mass -> use required_thickness (minimal to meet target)
            # Else use available thickness (we cannot meet full target)
            if mass_available_kg_per_m2 >= required_mass_per_m2 and required_mass_per_m2 > 0:
                thickness_to_use = required_thickness_m
                mass_used_per_m2 = required_mass_per_m2
            else:
                thickness_to_use = available_thickness_m
                mass_used_per_m2 = mass_available_kg_per_m2

            # clamp to not exceed element thickness
            if thickness_to_use > base_params['wall_thickness_m']:
                thickness_to_use = base_params['wall_thickness_m']

            # run sim with this thickness at position pos
            x_pcm, T_hist_pcm = transient_pcm_wall_simulation(
                wall_thickness_m=base_params['wall_thickness_m'],
                wall_k=base_params['wall_k'],
                wall_rho=base_params['wall_rho'],
                wall_cp=base_params['wall_cp'],
                pcm_thickness_m=thickness_to_use,
                pcm_k=base_params['pcm_k'],
                pcm_rho=base_params['pcm_rho'],
                pcm_cp=base_params['pcm_cp'],
                pcm_latent_heat=base_params['pcm_latent_heat'],
                pcm_Tmelt=base_params['pcm_Tmelt'],
                pcm_dT=base_params['pcm_dT'],
                T_indoor=base_params['T_indoor'],
                T_outdoor_series=T_outdoor_series,
                duration_hours=duration_hours,
                dx=dx,
                dt=dt,
                pcm_position_relative=pos
            )

            # build k array for energy calculation (reflecting PCM presence)
            k_arr = np.full(len(x_pcm), base_params['wall_k'])
            pcm_start_m = base_params['wall_thickness_m'] * pos
            pcm_end_m = pcm_start_m + thickness_to_use
            pcm_start_node = int(pcm_start_m / base_params['wall_thickness_m'] * len(x_pcm))
            pcm_end_node = min(len(x_pcm), int(np.ceil(pcm_end_m / base_params['wall_thickness_m'] * len(x_pcm))))
            if pcm_end_node > pcm_start_node:
                k_arr[pcm_start_node:pcm_end_node] = base_params['pcm_k']

            E_with_pcm_J = estimate_hourly_energy_into_indoor(T_hist_pcm, k_arr, dx, area_m2, base_params['T_indoor'])
            energy_saved_J = E_no_pcm_J - E_with_pcm_J  # positive means pcm reduced energy into indoor

            results.append({
                'section': sect,
                'position': pos,
                'thickness_m': thickness_to_use,
                'mass_used_kg_per_m2': mass_used_per_m2,
                'energy_saved_J': energy_saved_J,
                'energy_saved_kWh': energy_saved_J / 3.6e6
            })

    # Decide recommendation
    # 1) candidates meeting or exceeding the target (within a small tolerance)
    tol = 1e-6
    meeters = [r for r in results if r['energy_saved_J'] + tol >= target_total_J]
    if meeters:
        # choose the one with smallest thickness_m (economical)
        best = min(meeters, key=lambda r: (r['thickness_m'], -r['energy_saved_J']))
    else:
        # choose candidate with maximum energy_saved_J
        best = max(results, key=lambda r: r['energy_saved_J'])

    # Determine whether PCM is effectively needed:
    # If best energy saved is negligibly small (e.g., <0.001 kWh) or target <= 0 => No PCM needed.
    if target_total_J <= 0 or best['energy_saved_kWh'] < 1e-3:
        return None, E_no_pcm_J, results  # None indicates "No PCM needed"

    return best, E_no_pcm_J, results

# --------------------------- User I/O ---------------------------

def get_user_inputs():
    print("Please enter the parameters for the simulation:\n(Press Enter to use suggested defaults in parentheses)")

    # Location
    lat = float(input("Latitude (°) (e.g., 6.5244): ").strip() or "6.5244")
    lon = float(input("Longitude (°) (e.g., 3.3792): ").strip() or "3.3792")

    # Wall parameters (single representative element used for wall/roof/floor in this simple MVP)
    wall_thickness = float(input("Element thickness in meters (e.g., 0.2): ").strip() or "0.2")
    wall_k = float(input("Element thermal conductivity (W/m·K) (e.g., 1.5): ").strip() or "1.5")
    wall_rho = float(input("Element density (kg/m³) (e.g., 2500): ").strip() or "2500")
    wall_cp = float(input("Element specific heat capacity (J/kg·K) (e.g., 800): ").strip() or "800")

    # PCM properties
    pcm_k = float(input("PCM thermal conductivity (W/m·K) (default 0.25): ").strip() or "0.25")
    pcm_rho = float(input("PCM density (kg/m³) (default 900): ").strip() or "900")
    pcm_cp = float(input("PCM specific heat (J/kg·K) (default 2000): ").strip() or "2000")
    pcm_latent_heat = float(input("PCM latent heat (J/kg) (default 200000): ").strip() or "200000")
    pcm_Tmelt = float(input("PCM melting temperature (°C) (default 25): ").strip() or "25")
    pcm_dT = float(input("PCM phase change temperature range ±°C (default 2): ").strip() or "2")

    # PCM mass available per square meter (user-specified) -> thickness will be derived from this
    pcm_mass_kg_per_m2 = float(input("PCM mass available per m² (kg/m²) (e.g., 10.0): ").strip() or "10.0")

    # Area and target
    area = float(input("Area of the element in m² (e.g., 10): ").strip() or "10.0")
    energy_target_kwh = float(input("Target energy savings (kWh) (e.g., 5.0): ").strip() or "5.0")

    indoor_temp = float(input("Indoor temperature (°C) (e.g., 22): ").strip() or "22.0")
    duration = int(input("Simulation duration (hours) (e.g., 48): ").strip() or "48")
    dx = float(input("Spatial step size (m) (e.g., 0.002): ").strip() or "0.002")
    dt = float(input("Time step (seconds) (e.g., 1.0): ").strip() or "1.0")

    pos_input = input("PCM positions to test (comma list between 0 and 1) or Enter for defaults 0,0.25,0.5,0.75,1: ").strip()
    if pos_input:
        search_positions = [float(s) for s in pos_input.split(',')]
    else:
        search_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    return {
        'lat': lat,
        'lon': lon,
        'wall_thickness_m': wall_thickness,
        'wall_k': wall_k,
        'wall_rho': wall_rho,
        'wall_cp': wall_cp,
        'pcm_k': pcm_k,
        'pcm_rho': pcm_rho,
        'pcm_cp': pcm_cp,
        'pcm_latent_heat': pcm_latent_heat,
        'pcm_Tmelt': pcm_Tmelt,
        'pcm_dT': pcm_dT,
        'pcm_mass_kg_per_m2': pcm_mass_kg_per_m2,
        'area_m2': area,
        'energy_target_kwh': energy_target_kwh,
        'T_indoor': indoor_temp,
        'duration_hours': duration,
        'dx': dx,
        'dt': dt,
        'search_positions': search_positions
    }

# --------------------------- Main ---------------------------

def main():
    params = get_user_inputs()
    duration = params['duration_hours']

    # Fetch weather for location
    try:
        T_out_series = fetch_hourly_temperature_open_meteo(params['lat'], params['lon'], duration)
    except Exception:
        # fallback: simple sinusoidal daily temperature
        amplitude = 6.0
        base_outdoor_temp = 18.0
        T_out_series = [base_outdoor_temp + amplitude * math.sin(2 * math.pi * t / 24.0) for t in range(duration)]

    # Base params for simulation/optimizer
    base = {
        'wall_thickness_m': params['wall_thickness_m'],
        'wall_k': params['wall_k'],
        'wall_rho': params['wall_rho'],
        'wall_cp': params['wall_cp'],
        'pcm_k': params['pcm_k'],
        'pcm_rho': params['pcm_rho'],
        'pcm_cp': params['pcm_cp'],
        'pcm_latent_heat': params['pcm_latent_heat'],
        'pcm_Tmelt': params['pcm_Tmelt'],
        'pcm_dT': params['pcm_dT'],
        'T_indoor': params['T_indoor'],
        # total J target
        'energy_target_J': params['energy_target_kwh'] * 3.6e6,
        'pcm_mass_kg_per_m2': params['pcm_mass_kg_per_m2']
    }

    best, E_no_pcm_J, all_results = find_best_pcm_placement(
        base_params=base,
        T_outdoor_series=T_out_series,
        duration_hours=duration,
        dx=params['dx'],
        dt=params['dt'],
        area_m2=params['area_m2'],
        search_positions=params['search_positions']
    )

    # One-line recommendation:
    if best is None:
        # No PCM needed or negligible benefit
        print("No PCM needed.")
        return

    # If best thickness is effectively zero or extremely small, treat as no need
    if best['thickness_m'] <= 1e-6 or best['energy_saved_kWh'] < 1e-3:
        print("No PCM needed.")
        return

    # Format one-liner: "<Section>, <thickness_m> m"
    # Round thickness to sensible digits
    thickness_m = float(best['thickness_m'])
    # Round to mm precision for readability unless very small
    if thickness_m >= 0.001:
        thickness_str = f"{thickness_m:.4f} m"  # 4 decimals for meters (~0.1 mm)
    else:
        thickness_str = f"{thickness_m:.6f} m"

    # Final one-line output exactly as requested
    print(f"{best['section']}, {thickness_str}")

    # Optional: if you want to see additional info (uncomment)
    # print(f"# energy saved (kWh): {best['energy_saved_kWh']:.3f}, position: {best['position']}, mass_used_kg/m2: {best['mass_used_kg_per_m2']:.3f}")

    # Optional plot (comment out if not desired)
    try:
        # Show final-hour temperature profiles for no PCM vs best candidate for visualization
        x_no, T_no = transient_pcm_wall_simulation(
            wall_thickness_m=base['wall_thickness_m'],
            wall_k=base['wall_k'],
            wall_rho=base['wall_rho'],
            wall_cp=base['wall_cp'],
            pcm_thickness_m=0.0,
            pcm_k=0,
            pcm_rho=0,
            pcm_cp=0,
            pcm_latent_heat=0,
            pcm_Tmelt=0,
            pcm_dT=0,
            T_indoor=base['T_indoor'],
            T_outdoor_series=T_out_series,
            duration_hours=duration,
            dx=params['dx'],
            dt=params['dt'],
            pcm_position_relative=0.0
        )
        x_b, T_b = transient_pcm_wall_simulation(
            wall_thickness_m=base['wall_thickness_m'],
            wall_k=base['wall_k'],
            wall_rho=base['wall_rho'],
            wall_cp=base['wall_cp'],
            pcm_thickness_m=best['thickness_m'],
            pcm_k=base['pcm_k'],
            pcm_rho=base['pcm_rho'],
            pcm_cp=base['pcm_cp'],
            pcm_latent_heat=base['pcm_latent_heat'],
            pcm_Tmelt=base['pcm_Tmelt'],
            pcm_dT=base['pcm_dT'],
            T_indoor=base['T_indoor'],
            T_outdoor_series=T_out_series,
            duration_hours=duration,
            dx=params['dx'],
            dt=params['dt'],
            pcm_position_relative=best['position']
        )
        plt.figure(figsize=(7,4))
        plt.plot(x_no, T_no[-1], label='No PCM (final hour)')
        plt.plot(x_b, T_b[-1], label='Best PCM (final hour)')
        plt.xlabel('Depth (m)')
        plt.ylabel('Temperature (°C)')
        plt.title('Final-hour temperature profile')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
