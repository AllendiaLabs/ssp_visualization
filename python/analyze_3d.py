import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D toolkit

from utils import make_good_unitary, power_ssp

# Parameters
d = 10  # Dimension of the SSP
ssp = make_good_unitary(d)

# Define the range and number of exponents
n_frames = 400
exponents = np.linspace(-10, 10, n_frames)

# Precompute all SSP powers
# This creates an (n_frames, d) array where each row is ssp_powered for a given exponent
ssp_powers = np.array([power_ssp(ssp, e) for e in exponents])

# Extract coordinates for background scatter
x_all = ssp_powers[:, 0]
y_all = ssp_powers[:, 1]
z_all = ssp_powers[:, 2]

# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

# Set axis limits (adjust based on expected SSP values)
# Optionally, compute based on ssp_powers
margin = 0.1
x_min, x_max = np.min(x_all), np.max(x_all)
y_min, y_max = np.min(y_all), np.max(y_all)
z_min, z_max = np.min(z_all), np.max(z_all)

ax.set_xlim(x_min - margin, x_max + margin)
ax.set_ylim(y_min - margin, y_max + margin)
ax.set_zlim(z_min - margin, z_max + margin)

# Label the axes
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')

# Plot all SSP points as background with lighter appearance
background_scatter = ax.scatter(x_all, y_all, z_all, color='lightblue', alpha=0.3, s=20, label='All SSP Points')

# Initialize the scatter plot and lines for the current point
current_scatter = ax.scatter([], [], [], color='blue', s=100, label='Current SSP Point')
line, = ax.plot([], [], [], color='blue', linewidth=2, label='Connection Line')

# Optional: Add a legend
ax.legend(loc='upper left')

# Function to initialize the animation
def init():
    current_scatter._offsets3d = ([], [], [])  # Initialize empty scatter
    line.set_data([], [])                       # Initialize empty line data
    line.set_3d_properties([])                  # Initialize empty line z-data
    return current_scatter, line

# Function to update the plot
def update(frame):
    # Get the current exponent and corresponding SSP power
    exponent = exponents[frame]
    ssp_powered = ssp_powers[frame]
    print("norm of ssp_powered: ", np.linalg.norm(ssp_powered))
    print("mean of ssp_powered: ", np.mean(ssp_powered))
    print("std of ssp_powered: ", np.std(ssp_powered))

    # Update the title with the current exponent
    ax.set_title(f'SSP Raised to Power {exponent:.2f}')

    # Update current scatter point
    current_scatter._offsets3d = ([ssp_powered[0]], [ssp_powered[1]], [ssp_powered[2]])

    # Update line from origin to the current SSP point
    x = [0, ssp_powered[0]]
    y = [0, ssp_powered[1]]
    z = [0, ssp_powered[2]]
    line.set_data(x, y)
    line.set_3d_properties(z)

    return current_scatter, line

# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init, blit=False, interval=50, repeat=True
)

plt.tight_layout()
plt.show()
