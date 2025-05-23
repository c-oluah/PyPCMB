import argparse
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
    dt
):
    if T_outdoor_series is None:
        T_outdoor_series = [18 + 6*np.sin(2*np.pi*t/24) for t in range(duration_hours)]

    steps_per_hour = int(3600 / dt)
    total_steps = duration_hours * steps_per_hour

    # Spatial domain
    total_thickness = pcm_thickness_m + wall_thickness_m
    nx = int(total_thickness / dx)
    x = np.linspace(0, total_thickness, nx)

    # Material layers
    pcm_nodes = int(pcm_thickness_m / dx)
    k = np.full(nx, wall_k)
    rho = np.full(nx, wall_rho)
    cp = np.full(nx, wall_cp)
    alpha = np.full(nx, wall_k / (wall_rho * wall_cp))

    k[:pcm_nodes] = pcm_k
    rho[:pcm_nodes] = pcm_rho
    cp[:pcm_nodes] = pcm_cp
    alpha[:pcm_nodes] = pcm_k / (pcm_rho * pcm_cp)

    T = np.full(nx, T_indoor)
    T_hist = []

    for step in range(total_steps):
        hour = step // steps_per_hour
        T_out = T_outdoor_series[min(hour, len(T_outdoor_series)-1)]

        # Update cp for phase change
        for i in range(pcm_nodes):
            if abs(T[i] - pcm_Tmelt) <= pcm_dT:
                cp[i] = pcm_cp + pcm_latent_heat / (2 * pcm_dT)  # simple linearized
            else:
                cp[i] = pcm_cp

        alpha[:pcm_nodes] = pcm_k / (pcm_rho * cp[:pcm_nodes])

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

def main():
    parser = argparse.ArgumentParser(description="Simulate temperature profile through a wall with optional PCM.")

    # Wall parameters
    parser.add_argument("--wall_thickness", type=float, required=True, help="Wall thickness in meters")
    parser.add_argument("--wall_k", type=float, required=True, help="Wall thermal conductivity (W/m·K)")
    parser.add_argument("--wall_rho", type=float, required=True, help="Wall density (kg/m³)")
    parser.add_argument("--wall_cp", type=float, required=True, help="Wall specific heat capacity (J/kg·K)")

    # PCM parameters
    parser.add_argument("--pcm_thickness", type=float, default=0.0, help="PCM layer thickness in meters (0 = no PCM)")
    parser.add_argument("--pcm_k", type=float, default=0.25, help="PCM thermal conductivity (W/m·K)")
    parser.add_argument("--pcm_rho", type=float, default=900, help="PCM density (kg/m³)")
    parser.add_argument("--pcm_cp", type=float, default=2000, help="PCM specific heat (J/kg·K)")
    parser.add_argument("--pcm_latent_heat", type=float, default=200000, help="Latent heat (J/kg)")
    parser.add_argument("--pcm_Tmelt", type=float, default=25, help="Melting temperature (°C)")
    parser.add_argument("--pcm_dT", type=float, default=2, help="Phase change temperature range (±°C)")

    # Environmental + Simulation
    parser.add_argument("--indoor_temp", type=float, default=22, help="Indoor temperature (°C)")
    parser.add_argument("--duration", type=int, default=48, help="Simulation duration (hours)")
    parser.add_argument("--dx", type=float, default=0.002, help="Spatial step size (m)")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step (seconds)")
    parser.add_argument("--amplitude", type=float, default=6.0, help="Daily outdoor temp swing amplitude")
    parser.add_argument("--base_outdoor_temp", type=float, default=18.0, help="Base outdoor temp (°C)")

    args = parser.parse_args()

    T_out_series = [
        args.base_outdoor_temp + args.amplitude * np.sin(2 * np.pi * t / 24)
        for t in range(args.duration)
    ]

    x, T_hist = transient_pcm_wall_simulation(
        wall_thickness_m=args.wall_thickness,
        wall_k=args.wall_k,
        wall_rho=args.wall_rho,
        wall_cp=args.wall_cp,
        pcm_thickness_m=args.pcm_thickness,
        pcm_k=args.pcm_k,
        pcm_rho=args.pcm_rho,
        pcm_cp=args.pcm_cp,
        pcm_latent_heat=args.pcm_latent_heat,
        pcm_Tmelt=args.pcm_Tmelt,
        pcm_dT=args.pcm_dT,
        T_indoor=args.indoor_temp,
        T_outdoor_series=T_out_series,
        duration_hours=args.duration,
        dx=args.dx,
        dt=args.dt
    )

    # Plot
    plt.figure(figsize=(10, 5))
    for i in range(0, len(T_hist), max(1, len(T_hist)//10)):
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
