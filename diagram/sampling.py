import matplotlib.pyplot as plt

# Better style for the graph
plt.style.use('seaborn-darkgrid')

# Increase the size of the figure for better readability
plt.figure(figsize=(10, 6))

# x-axis values
x = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95, 1]
# y-axis values
y = [2, 4, 5, 7, 6, 8, 9, 11, 12, 12]

# Plotting points as a scatter plot with improved marker style and size
plt.scatter(x, y, label="Recall Rate", color="darkgreen", marker="*", s=100)

# Improved x-axis and y-axis labels
plt.xlabel('Negative Sampling Ratio')
plt.ylabel('Recall Rate')

# More descriptive title
plt.title('Scatter Plot of Recall Rate vs. Negative Sampling Ratio')

# Add grid lines for better readability
plt.grid(True)

# Show a legend on the plot with a frame
plt.legend(frameon=True)

# Set the limits for the axes if necessary (commented out for now)
# plt.xlim([0.45, 1.05])
# plt.ylim([0, 13])

# Function to save the plot as a high-resolution PNG image
plt.savefig('scatter_plot_recall_vs_sampling.png', dpi=300)

# Function to show the plot
plt.show()