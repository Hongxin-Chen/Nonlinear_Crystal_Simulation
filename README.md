# Nonlinear Crystal SHG Calculator

## Introduction
This is an optical simulation tool developed using **Python** and **Streamlit**. It is designed to calculate critical parameters for Second Harmonic Generation (SHG) processes in common nonlinear crystals, such as BBO, LBO, CLBO, and KDP.

## Features
- **Multi-Crystal Support**: Built-in database containing parameters for common crystals including BBO, LBO, CLBO, KDP, and DKDP.
- **Phase Matching Calculation**: Supports precise calculation of phase matching angles for both **Type I** and **Type II** interactions.
- **Effective Nonlinear Coefficient ($d_{eff}$)**: Calculates $d_{eff}$ for various crystal orientations based on **IEEE standards** and **Kleinman symmetry**.
- **Interactive Interface**: Provides a user-friendly web interface for real-time parameter tuning and result visualization.
- **3D Visualization**: Supports 3D rendering of crystal orientation schematics and refractive index ellipsoids.

## Demo
<img width="1575" height="551" alt="Phase Matching Results" src="https://github.com/user-attachments/assets/9301460a-6002-4135-83ce-c7ed678033fb" />
<img width="1136" height="806" alt="3D Crystal Visualization" src="https://github.com/user-attachments/assets/5a8c314a-f220-4ebb-9b62-8f72d62df564" />
<img width="1508" height="772" alt="Analysis Charts" src="https://github.com/user-attachments/assets/5a721261-3119-4d04-932f-c9eccc8c3d71" />

## Installation & Usage

1. **Clone the repository**
   
   git clone [https://github.com/Hongxin-Chen/Nonlinear_Crystal_Simulation_of_SHG.git](https://github.com/Hongxin-Chen/Nonlinear_Crystal_Simulation_of_SHG.git)
   cd Nonlinear_Crystal_Simulation_of_SHG

2. **Install dependencies**
   
   pip install numpy streamlit plotly scipy pandas

3. **Run the application**

   streamlit run launcher.py
