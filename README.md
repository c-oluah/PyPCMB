
PyPCMB is a Python-based simulation tool for evaluating the thermal performance of walls with and without Phase Change Materials (PCMs). The tool simulates temperature profiles over time through a wall structure and helps estimate:

📉 Energy savings

🧱 Optimal PCM placement

⚖️ Required PCM mass

🚀 Features
Simulates transient 1D heat transfer through multilayer walls

Allows user-supplied wall and PCM parameters

Models PCM latent heat effect during phase transition

Generates temperature profiles as plots

CLI-based: Easily automate or integrate with larger systems

📦 Requirements
Install dependencies via pip:

pip install numpy matplotlib

🛠️ Usage
Run the tool from the command line:

python pcm_simulation_cli.py --wall_thickness 0.2 --wall_k 0.6 --wall_rho 1800 --wall_cp 1000 \
  --pcm_thickness 0.02 --pcm_k 0.25 --pcm_rho 900 --pcm_cp 2000 --pcm_latent_heat 200000 \
  --pcm_Tmelt 25 --pcm_dT 2 --indoor_temp 22 --duration 48

⚙️ Command-line Arguments
Parameter         Type Description
--wall_thickness float Total thickness of the wall (m)
--wall_k         float Wall thermal conductivity (W/m·K)
--wall_rho         float Wall density (kg/m³)
--wall_cp         float Wall specific heat capacity (J/kg·K)
--pcm_thickness     float Thickness of the PCM layer (m)
--pcm_k             float PCM thermal conductivity (W/m·K)
--pcm_rho         float PCM density (kg/m³)
--pcm_cp         float PCM specific heat capacity (J/kg·K)
--pcm_latent_heat float PCM latent heat during phase change (J/kg)
--pcm_Tmelt         float PCM melting temperature (°C)
--pcm_dT         float Phase transition temperature range (°C)
--indoor_temp     float Fixed indoor air temperature (°C)
--duration         int     Simulation duration (in hours)
--dx (optional)     float Spatial resolution (default: 0.002 m)
--dt (optional)     float Time step (seconds; default: 1 s)
--amplitude (opt) float Amplitude of sinusoidal outdoor temperature variation
--base_outdoor_temp float Base outdoor temperature (°C; default: 18 °C)

📊 Output
Shows temperature distribution plots over time through the wall.

Helps visualize the thermal buffering effect of PCM layers.

🧠 Interpretation
Compare runs with vs without PCM to estimate energy savings.

Higher thermal stability and lower wall surface variation = better insulation performance.

🔧 Future Features (Roadmap)
Export results to CSV

Calculate total energy flux and savings

GUI and web version (with Plotly)

📚 Citation

🧑‍💻 Author
Oluah Chukwumaobi
