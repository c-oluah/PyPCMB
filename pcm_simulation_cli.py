import numpy as np
import matplotlib.pyplot as plt

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
    if T_outdoor_series is None:
        T_outdoor_series = [18 + 6*np.sin(2*np.pi*t/24) for t in range(duration_hours)]

    steps_per_hour = int(3600 / dt)
    total_steps = duration_hours * steps_per_hour

    # Spatial domain
    total_thickness = wall_thickness_m
    nx = int(total_thickness / dx)
    x = np.linspace(0, total_thickness, nx)

    # Material layers
    k = np.full(nx, wall_k)
    rho = np.full(nx, wall_rho)
    cp = np.full(nx, wall_cp)
    alpha = np.full(nx, wall_k / (wall_rho * wall_cp))

    # Calculate PCM start and end nodes based on its relative position
    pcm_start_m = wall_thickness_m * pcm_position_relative
    pcm_end_m = pcm_start_m + pcm_thickness_m

    # Ensure PCM is within the wall
    if pcm_end_m > wall_thickness_m:
        print("Warning: PCM extends beyond wall thickness. Adjusting PCM thickness.")
        pcm_thickness_m = wall_thickness_m - pcm_start_m
        if pcm_thickness_m < 0:
            pcm_thickness_m = 0

    pcm_start_node = int(pcm_start_m / dx)
    pcm_end_node = int(pcm_end_m / dx)

    # Apply PCM properties to the designated nodes
    if pcm_thickness_m > 0:
        k[pcm_start_node:pcm_end_node] = pcm_k
        rho[pcm_start_node:pcm_end_node] = pcm_rho
        cp[pcm_start_node:pcm_end_node] = pcm_cp
        alpha[pcm_start_node:pcm_end_node] = pcm_k / (pcm_rho * pcm_cp)

    T = np.full(nx, T_indoor)
    T_hist = []

    for step in range(total_steps):
        hour = step // steps_per_hour
        T_out = T_outdoor_series[min(hour, len(T_outdoor_series)-1)]

        # Update cp for phase change only within PCM nodes
        if pcm_thickness_m > 0:
            for i in range(pcm_start_node, pcm_end_node):
                if abs(T[i] - pcm_Tmelt) <= pcm_dT:
                    cp[i] = pcm_cp + pcm_latent_heat / (2 * pcm_dT)  # simple linearized
                else:
                    cp[i] = pcm_cp
            alpha[pcm_start_node:pcm_end_node] = k[pcm_start_node:pcm_end_node] / (rho[pcm_start_node:pcm_end_node] * cp[pcm_start_node:pcm_end_node])


        T_new = T.copy()
        for i in range(1, nx - 1):
            T_new[i] = T[i] + alpha[i] * dt / dx**2 * (T[i+1] - 2*T[i] + T[i-1])

        # Boundary conditions
        T_new[0] = T_out
        T_new[-1] = T_indoor
        T = T_new.copy()

        if step % steps_per_hour == 0:
            T_hist.append(T.copy())

    return x, np.array(T_hist)

def get_user_inputs():
    print("Please enter the parameters for the simulation:")

    # Wall parameters
    wall_thickness = float(input("Wall thickness in meters (e.g., 0.2): "))
    wall_k = float(input("Wall thermal conductivity (W/m·K) (e.g., 1.5): "))
    wall_rho = float(input("Wall density (kg/m³) (e.g., 2500): "))
    wall_cp = float(input("Wall specific heat capacity (J/kg·K) (e.g., 800): "))

    # PCM parameters
    use_pcm = input("Do you want to include a PCM layer? (yes/no): ").lower()
    pcm_thickness = 0.0
    pcm_k = 0.25
    pcm_rho = 900
    pcm_cp = 2000
    pcm_latent_heat = 200000
    pcm_Tmelt = 25
    pcm_dT = 2
    pcm_position_relative = 0.0 # Default to the exterior surface

    if use_pcm == 'yes':
        pcm_thickness = float(input("PCM layer thickness in meters (e.g., 0.01): "))
        pcm_k = float(input("PCM thermal conductivity (W/m·K) (default: 0.25): ") or pcm_k)
        pcm_rho = float(input("PCM density (kg/m³) (default: 900): ") or pcm_rho)
        pcm_cp = float(input("PCM specific heat (J/kg·K) (default: 2000): ") or pcm_cp)
        pcm_latent_heat = float(input("PCM latent heat (J/kg) (default: 200000): ") or pcm_latent_heat)
        pcm_Tmelt = float(input("PCM melting temperature (°C) (default: 25): ") or pcm_Tmelt)
        pcm_dT = float(input("PCM phase change temperature range (±°C) (default: 2): ") or pcm_dT)

        while True:
            pcm_position_input = input("Best position of PCM (relative to wall thickness, 0=exterior, 1=interior, e.g., 0.2 for 20% from exterior): ")
            try:
                pcm_position_relative = float(pcm_position_input)
                if 0 <= pcm_position_relative <= 1:
                    break
                else:
                    print("Please enter a value between 0 and 1.")
            except ValueError:
                print("Invalid input. Please enter a numerical value.")
        
        # Calculate amount of PCM needed (per unit area)
        amount_pcm_kg_per_m2 = pcm_thickness * pcm_rho
        print(f"\nAmount of PCM needed per square meter of wall: {amount_pcm_kg_per_m2:.2f} kg/m²")

    # Environmental + Simulation
    indoor_temp = float(input("Indoor temperature (°C) (e.g., 22): "))
    duration = int(input("Simulation duration (hours) (e.g., 48): "))
    dx = float(input("Spatial step size (m) (e.g., 0.002): "))
    dt = float(input("Time step (seconds) (e.g., 1.0): "))
    amplitude = float(input("Daily outdoor temperature swing amplitude (°C) (e.g., 6.0): "))
    base_outdoor_temp = float(input("Base outdoor temperature (°C) (e.g., 18.0): "))

    return {
        "wall_thickness_m": wall_thickness,
        "wall_k": wall_k,
        "wall_rho": wall_rho,
        "wall_cp": wall_cp,
        "pcm_thickness_m": pcm_thickness,
        "pcm_k": pcm_k,
        "pcm_rho": pcm_rho,
        "pcm_cp": pcm_cp,
        "pcm_latent_heat": pcm_latent_heat,
        "pcm_Tmelt": pcm_Tmelt,
        "pcm_dT": pcm_dT,
        "T_indoor": indoor_temp,
        "duration_hours": duration,
        "dx": dx,
        "dt": dt,
        "amplitude": amplitude,
        "base_outdoor_temp": base_outdoor_temp,
        "pcm_position_relative": pcm_position_relative
    }

def main():
    params = get_user_inputs()

    T_out_series = [
        params["base_outdoor_temp"] + params["amplitude"] * np.sin(2 * np.pi * t / 24)
        for t in range(params["duration_hours"])
    ]

    x, T_hist = transient_pcm_wall_simulation(
        wall_thickness_m=params["wall_thickness_m"],
        wall_k=params["wall_k"],
        wall_rho=params["wall_rho"],
        wall_cp=params["wall_cp"],
        pcm_thickness_m=params["pcm_thickness_m"],
        pcm_k=params["pcm_k"],
        pcm_rho=params["pcm_rho"],
        pcm_cp=params["pcm_cp"],
        pcm_latent_heat=params["pcm_latent_heat"],
        pcm_Tmelt=params["pcm_Tmelt"],
        pcm_dT=params["pcm_dT"],
        T_indoor=params["T_indoor"],
        T_outdoor_series=T_out_series,
        duration_hours=params["duration_hours"],
        dx=params["dx"],
        dt=params["dt"],
        pcm_position_relative=params["pcm_position_relative"]
    )

    # Plot
    plt.figure(figsize=(10, 5))
    # Adjusting the plotting frequency for better visualization if duration is long
    plot_interval = max(1, len(T_hist) // 10) # Plot at least 10 lines, or more if total hours are fewer
    for i in range(0, len(T_hist), plot_interval):
        plt.plot(x, T_hist[i], label=f"{i}h")
    
    plt.xlabel("Wall depth (m)")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature Evolution Through Wall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()