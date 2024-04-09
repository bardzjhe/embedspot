import matplotlib.pyplot as plt

# Better style for the graph
plt.style.use('seaborn-darkgrid')

# Increase the size of the figure for better readability
plt.figure(figsize=(10, 6))

search_space_sizes = [1000, 10000, 100000]  # Number of items
linear_scan_times = [0.05, 0.5, 5]  # Linear increase in time
annoy_times = [0.005, 0.02, 0.15]  # Sub-linear increase in time
milvus_ivf_flat_times = [0.006, 0.03, 0.2]  # Good performance, slightly worse at scale
milvus_ivf_pq_times = [0.007, 0.035, 0.18]  # Similar to Annoy, but with a little overhead

# Plotting the hypothetical data
plt.plot(search_space_sizes, linear_scan_times, marker='o', linestyle='-', color='blue', label='Linear Scan', linewidth=2)
plt.plot(search_space_sizes, annoy_times, marker='s', linestyle='--', color='red', label='Annoy', linewidth=2)
plt.plot(search_space_sizes, milvus_ivf_flat_times, marker='^', linestyle='-.', color='green', label='Milvus (IVF_FLAT)', linewidth=2)
plt.plot(search_space_sizes, milvus_ivf_pq_times, marker='x', linestyle=':', color='purple', label='Milvus (IVF_PQ)', linewidth=2)

# Naming the x-axis, y-axis, and the whole graph
plt.xlabel('Search Space Size (Number of Items)')
plt.ylabel('Average Query Latency (seconds)')
plt.title('Hypothetical Comparison of Retrieval Times on MovieLens-1M')

# Show a legend on the plot at the best location
plt.legend(loc='best')

# Show grid lines for better readability
plt.grid(True)

# Function to save the plot as a high-resolution PNG image
plt.savefig('hypothetical_comparison_of_retrieval_times.png', dpi=300)

# Function to show the plot
plt.show()