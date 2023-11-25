import numpy as np
import matplotlib.pyplot as plt

# Function to generate random points in d-dimensional space
def generate_points(num_points, dimensions):
    return np.random.uniform(0, 1, size=(num_points, dimensions))

# Function to calculate L1, L2, and Linfinity distances
def calculate_distances(query_point, data_points):
    l1_distances = np.sum(np.abs(data_points - query_point), axis=1)
    l2_distances = np.sqrt(np.sum((data_points - query_point)**2, axis=1))
    linfinity_distances = np.max(np.abs(data_points - query_point), axis=1)
    return l1_distances, l2_distances, linfinity_distances

# Set dimensions to [1, 2, 4, 8, 16, 32, 64]
dimensions_list = [2**i for i in range(7)]


# Generate a dataset of 1 million random points for each dimension
num_points = 1000000
dataset = {d: generate_points(num_points, d) for d in dimensions_list}

# Choose 100 random query points
query_points_indices = np.random.choice(num_points, size=100, replace=False)

# Initialize lists to store average ratios for each distance measure
avg_ratios_l1 = []
avg_ratios_l2 = []
avg_ratios_linfinity = []

# Iterate through dimensions
for d in dimensions_list:
    # Initialize lists to store ratios for each query point
    ratios_l1 = []
    ratios_l2 = []
    ratios_linfinity = []
    
    # Iterate through query points
    for query_index in query_points_indices:
        query_point = dataset[d][query_index]
        data_points = np.delete(dataset[d], query_index, axis=0)  # Exclude the query point
        
        # Calculate distances
        l1_distances, l2_distances, linfinity_distances = calculate_distances(query_point, data_points)
        
        # Find farthest and nearest distances
        farthest_distance_l1 = np.max(l1_distances)
        nearest_distance_l1 = np.min(l1_distances)
        
        farthest_distance_l2 = np.max(l2_distances)
        nearest_distance_l2 = np.min(l2_distances)

        farthest_distance_linf = np.max(linfinity_distances)
        nearest_distance_linf  = np.min(linfinity_distances)
        
        # Calculate ratios and append to lists
        ratios_l1.append(farthest_distance_l1 / nearest_distance_l1)
        ratios_l2.append(farthest_distance_l2 / nearest_distance_l2)
        ratios_linfinity.append(farthest_distance_linf / nearest_distance_linf)
    
    # Calculate average ratios for the current dimension
    avg_ratios_l1.append(np.mean(ratios_l1))
    avg_ratios_l2.append(np.mean(ratios_l2))
    avg_ratios_linfinity.append(np.mean(ratios_linfinity))

# Plot the average ratios versus dimensions for each distance measure
plt.plot(dimensions_list, avg_ratios_l1, label='L1 Distance')
plt.plot(dimensions_list, avg_ratios_l2, label='L2 Distance')
plt.plot(dimensions_list, avg_ratios_linfinity, label='Linfinity Distance')

# Add labels and legend
plt.xlabel('Dimension (d)')
plt.ylabel('Average Ratio of Farthest to Nearest Distance')
plt.legend()
plt.yscale('log')  # Set y-axis to default logarithmic scale
plt.title('Behavior of Uniformly Distributed Points in High-dimensional Space')

# Show the plot
plt.show()
