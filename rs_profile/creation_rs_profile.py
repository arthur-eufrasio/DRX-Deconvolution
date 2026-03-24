import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import json
import pickle
import os

# --- 1. LOAD SIMULATION DATA ---
file_path = "rs_profile/sim_data/0006_model_stress_profile.json"
with open(file_path, "r") as f:
    data_sim = json.load(f)

# Assuming data is structured as lists of [x, y] coordinate pairs
data_surface = np.array(data_sim["0006_model"]['surface'])
data_depth = np.array(data_sim["0006_model"]['depth'])

# Ensure the output directory exists
os.makedirs("rs_profile", exist_ok=True)

def create_and_save_profile(data_ref, profile_name, output_filename):
    """
    Plots reference data, allows user to click points, generates a spline, 
    and saves the model as a .pkl file.
    """
    print(f"\n" + "="*40)
    print(f"   PROCESSING {profile_name.upper()} PROFILE")
    print("="*40)
    
    # Extract X and Y from the reference data for plotting
    x_ref = data_ref[:, 0]
    y_ref = data_ref[:, 1]

    # --- 2. INTERACTIVE PLOT ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_ref, y_ref, 'b--', alpha=0.6, label=f'Simulation Reference ({profile_name})')
    
    ax.set_title(f"STEP 1: Click to draw the {profile_name} profile. Press ENTER to finish.")
    ax.set_xlabel("Distance") 
    ax.set_ylabel("Residual Stress") 
    ax.grid(True, alpha=0.3)
    ax.legend()

    print(f">>> INSTRUCTIONS:")
    print(f"1. Click on the graph to trace your desired {profile_name} curve.")
    print(f"2. Press ENTER (or middle mouse button) when finished.")

    # --- 3. POINT COLLECTION ---
    points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig) # Close interactive window after pressing ENTER

    if len(points) < 2:
        print(f"Error: You must select at least 2 points for the {profile_name} profile!")
        return None

    # --- 4. SPLINE GENERATION ---
    points = np.array(points)
    order = np.argsort(points[:, 0]) # Sort points by X axis
    
    x_user = points[order, 0]
    y_user = points[order, 1]

    # Create the spline
    spline = CubicSpline(x_user, y_user, bc_type='natural')

    # --- 5. FINAL PLOT PREPARATION ---
    # Generate points for smooth plotting based on user input range
    x_final = np.linspace(min(x_user), max(x_user), 500)
    y_final = spline(x_final)

    fig_result, ax_result = plt.subplots(figsize=(10, 6))
    
    # Plot reference, user points, and final spline
    ax_result.plot(x_ref, y_ref, 'b--', alpha=0.3, label='Simulation Reference')
    ax_result.plot(x_user, y_user, 'ro', label='Selected Points')
    ax_result.plot(x_final, y_final, 'g-', linewidth=2, label=f'Final {profile_name} Spline')
    
    ax_result.set_title(f"Final {profile_name} Residual Stress Profile")
    ax_result.set_xlabel("Distance")
    ax_result.set_ylabel("Residual Stress")
    ax_result.grid(True, alpha=0.3)
    ax_result.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax_result.legend()

    # --- 6. SAVE FILE ---
    with open(output_filename, "wb") as f:
        pickle.dump(spline, f)

    print(f"\n>>> SUCCESS!")
    print(f"The {profile_name} curve was successfully saved to '{output_filename}'.")
    
    return fig_result

# Execute for Surface
fig_surf = create_and_save_profile(data_surface, "Surface", "rs_profile/surface_spline.pkl")

# Execute for Depth
fig_dep = create_and_save_profile(data_depth, "Depth", "rs_profile/depth_spline.pkl")

# Show both final result plots together at the end
if fig_surf is not None or fig_dep is not None:
    print("\nDisplaying final generated profiles. Close the windows to exit the script.")
    plt.show()