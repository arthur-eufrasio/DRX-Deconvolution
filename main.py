import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def simulate_xrd_measurements(spline_path, start_pos, end_pos, step_size, beam_diameter):
    """
    Simulates XRD measurements by averaging a continuous stress profile 
    over a specified beam diameter at regular intervals. Assumes the 
    stress profile is symmetric around x = 0.
    """
    # 1. Load the previously generated spline
    if not os.path.exists(spline_path):
        print(f"Error: Spline file '{spline_path}' not found.")
        print("Make sure you run the profile creation script first.")
        return None, None

    with open(spline_path, "rb") as f:
        spline = pickle.load(f)

    # 2. Define measurement points (centers of the X-ray beam)
    measurement_centers = np.arange(start_pos, end_pos + (step_size * 0.01), step_size)
    simulated_stresses = []

    # 3. Perform the moving average simulation with SYMMETRY
    for center in measurement_centers:
        # Define the spatial window of the X-ray beam
        window_start = center - (beam_diameter / 2.0)
        window_end = center + (beam_diameter / 2.0)
        
        # Sample the window densely
        window_points = np.linspace(window_start, window_end, 100)
        
        # APPLY SYMMETRY: Use np.abs() to mirror negative X values to positive X values
        # before passing them to the spline function
        window_stresses = spline(np.abs(window_points))
        
        # Calculate the mean stress in this window
        avg_stress = np.mean(window_stresses)
        simulated_stresses.append(avg_stress)

    measurement_centers = np.array(measurement_centers)
    simulated_stresses = np.array(simulated_stresses)

    # --- 4. Plotting the Results ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the continuous "true" symmetric profile for reference
    plot_start = min(start_pos - beam_diameter, -max(spline.x))
    plot_end = max(end_pos + beam_diameter, max(spline.x))
    x_continuous = np.linspace(plot_start, plot_end, 1000)
    
    # APPLY SYMMETRY to the continuous plot as well
    y_continuous = spline(np.abs(x_continuous))
    
    ax.plot(x_continuous, y_continuous, 'g-', linewidth=2, alpha=0.4, label='Continuous Profile (Symmetric)')

    # Connect the simulated measurements with a dashed line
    ax.plot(measurement_centers, simulated_stresses, 'r--', linewidth=1.5, alpha=0.7, label='Measured Trend')

    # Plot the simulated XRD measurements with error bars representing the beam width
    ax.errorbar(measurement_centers, simulated_stresses, 
                xerr=beam_diameter/2, fmt='ro', capsize=4, elinewidth=2,
                label=f'XRD Measurement Points\n(Beam: {beam_diameter}mm)')

    ax.set_title(f"Simulated XRD Measurements (Symmetric Profile)\n(Step: {step_size}mm | Beam Dia: {beam_diameter}mm)")
    ax.set_xlabel("Distance from Center (mm)")
    ax.set_ylabel("Averaged Residual Stress (MPa)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5) # Add a center line indicator
    ax.legend()
    
    plt.tight_layout()
    plt.show()

    return measurement_centers, simulated_stresses

# --- 5. EXECUTION EXAMPLE ---
if __name__ == "__main__":
    START_POS = -4.0      # Starting position of the measurement map
    END_POS = 4.0         # Ending position
    STEP_SIZE = 0.3       # Distance between measurement centers
    BEAM_DIAMETER = 0.3   # Spot size of the X-ray beam
    
    PROFILE_PATH = "rs_profile/sim_data/surface_spline.pkl"
    
    print("\n" + "="*40)
    print("   RUNNING XRD SIMULATION (SYMMETRIC)")
    print("="*40)
    
    centers, stresses = simulate_xrd_measurements(
        PROFILE_PATH, 
        START_POS, 
        END_POS, 
        STEP_SIZE, 
        BEAM_DIAMETER
    )

    if centers is not None:
        print("\n>>> SIMULATION RESULTS:")
        print(f"{'Position Center':>15} | {'Measured Stress':>15}")
        print("-" * 35)
        for c, s in zip(centers, stresses):
            print(f"{c:15.3f} | {s:15.3f}")