import matplotlib.pyplot as plt

# Better style for the graph
plt.style.use('seaborn-darkgrid')

# Increase the size of the figure for better readability
plt.figure(figsize=(10, 6))

search_space_sizes = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000]  # Number of items
linear_scan_times = [0.33, 1.08, 2.26, 4.15, 5.72, 8.23, 10.09, 12.18]  # Linear increase in time
annoy_times = [0.109, 0.111, 0.119, 0.142, 0.154, 0.178, 0.187, 0.211]  # Sub-linear increase in time

# Plotting the hypothetical data
plt.plot(search_space_sizes, linear_scan_times, marker='o', linestyle='-', color='blue', label='Brute-force', linewidth=2)
plt.plot(search_space_sizes, annoy_times, marker='s', linestyle='--', color='red', label='ANNs (Annoy)', linewidth=2)

# Naming the x-axis, y-axis, and the whole graph
plt.xlabel('Search Space Size (Number of Items)')
plt.ylabel('Average Query Latency (seconds)')
plt.title('Hypothetical Comparison of Retrieval Times on MovieLens-1M')

# Set the y-axis to a logarithmic scale
plt.yscale('log')

# Show a legend on the plot at the best location
plt.legend(loc='best')

# Show grid lines for better readability
plt.grid(True, which="both", ls="--")  # 'which' includes minor ticks on the log scale

# Function to save the plot as a high-resolution PNG image
plt.savefig('hypothetical_comparison_of_retrieval_times_log_scale.png', dpi=300)

# Function to show the plot
plt.show()