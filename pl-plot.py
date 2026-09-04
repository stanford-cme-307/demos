import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button

# Generate x and y values
x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)

# Define the function to plot
Z = (Y - np.sin(X))**2

# Create the figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# Add sliders for rotation
ax_azim = plt.axes([0.25, 0.05, 0.50, 0.03])
ax_elev = plt.axes([0.25, 0.01, 0.50, 0.03])
azim_slider = Slider(ax_azim, 'Azimuth', 0, 360, valinit=ax.azim)
elev_slider = Slider(ax_elev, 'Elevation', -90, 90, valinit=ax.elev)

def update_plot(val):
    ax.view_init(elev=elev_slider.val, azim=azim_slider.val)
    fig.canvas.draw_idle()

azim_slider.on_changed(update_plot)
elev_slider.on_changed(update_plot)

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Plot of (y - np.sin(x))**2')

plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Generate x and y values
# x = np.linspace(-10, 10, 50)
# y = np.linspace(-10, 10, 50)
# X, Y = np.meshgrid(x, y)

# # Define the function to plot
# Z = (Y - np.sin(X))**2 + .2*X

# # Create the 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')

# # Set labels and title
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.set_title('3D Plot of (x - sin(x))^2 + y^2')

# # Show the plot
# plt.savefig('polyak_lojasiewicz_3d_plot.png')