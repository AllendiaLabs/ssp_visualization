import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import make_good_unitary, power_ssp

# Parameters
d = 3  # Dimension of the SSP
ssp = make_good_unitary(d)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlim(-1, min(d, 20))
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('Dimension Index')
ax.set_ylabel('SSP Value')

# Initialize the lines for the parallel coordinates plot
lines = []
for i in range(d):
    # Each line represents a dimension
    line, = ax.plot([], [], marker='o', color='blue')
    lines.append(line)

# Function to initialize the animation
def init():
    for line in lines:
        line.set_data([], [])
    return lines

n_frames = 400

# Function to update the plot
def update(frame):

    exponent = np.linspace(0, 2, n_frames)[frame]
    ssp_powered = power_ssp(ssp, exponent)

    ax.set_title(f'SSP Raised to Power {exponent:.2f}')

    # Update lines for parallel coordinates plot
    for i, line in enumerate(lines):
        # For parallel coordinates, plot a vertical line at x=i from y=0 to y=ssp_powered[i]
        line.set_data([i, i], [0, ssp_powered[i]])

    return lines

# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init, blit=False, interval=50, repeat=True
)

plt.tight_layout()
plt.show()
