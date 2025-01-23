import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Polygon

plt.rcParams.update({'font.size': 20})

# Define parameters for the normal distribution
mean1 = 100
std_dev1 = 20

fig, ax = plt.subplots()
fig.set_size_inches(6, 6)

# Generate data for the normal distribution curve
x = np.linspace(mean1 - 4*std_dev1, mean1 + 4*std_dev1, 1000)
y1 = norm.pdf(x, mean1, std_dev1)

# Plot the normal distribution curve
plt.plot(x, y1, label='Historical Normal Distribution')

# Find the point on the curve corresponding to the mean
point_y = norm.pdf(mean1, mean1, std_dev1)

# Create a diamond shape to represent the 1-Wasserstein ambiguity set with the l1-norm.
vertices = [( mean1 + 15, point_y ), ( mean1, point_y + 0.0045 ), ( mean1 - 15, point_y ), ( mean1, point_y - 0.0045 )]
# Create a Polygon object
diamond = Polygon(vertices, fill=True, edgecolor='red', facecolor='red', label = '1-Wasserstein Ambiguity Set')
# Set alpha
diamond.set_alpha(0.2)

ax.add_patch(diamond)

# Plot additional normal distribution curves with different means and standard deviations
means = [80, 105, 120]
std_devs = [10, 15, 30]
for mean, std_dev in zip(means, std_devs):
    y = norm.pdf(x, mean, std_dev)
    plt.plot(x, y, label=f'Mean: {mean}, Std Dev: {std_dev}')

# Add labels and legend
plt.xlabel('Value')
plt.ylabel('Probability Density')
# plt.title('Normal Distribution Curves')
# plt.legend()
plt.tight_layout()
# Show the plot
plt.show()

# Save the plot as a PNG file
# fig.savefig('figure/Ambiguity_Set.jpg', dpi = 330 )