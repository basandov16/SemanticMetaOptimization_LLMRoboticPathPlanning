import numpy as np
import matplotlib.pyplot as plt

class OccupancyGridMap:
    def __init__(self, width, height, resolution=0.1, obstacle_probability=0.2):
        """
        Initialize an occupancy grid map.
        
        Args:
            width: Map width in meters
            height: Map height in meters
            resolution: Grid cell size in meters
            obstacle_probability: Probability of a cell being an obstacle (0-1)
        """
        print(f"Initializing an occupancy grid map with width={width}m, height={height}m, resolution={resolution}m, obstacle_probability={obstacle_probability}...")
        self.width = width
        self.height = height
        self.resolution = resolution
        self.obstacle_probability = obstacle_probability
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        self.grid = np.zeros((self.grid_height, self.grid_width))
    
    def generate_random_map(self):
        """Generate a random occupancy grid map."""
        print("Generating random occupancy grid map...")
        self.grid = np.random.choice(
            [0, 1],
            size=(self.grid_height, self.grid_width),
            p=[1 - self.obstacle_probability, self.obstacle_probability]
        )
        return self.grid
    
    def visualize(self):
        """Visualize the occupancy grid map."""
        print("Visualizing occupancy grid map...")
        plt.figure(figsize=(10, 8))
        plt.imshow(self.grid, cmap='binary', origin='lower')
        plt.xlabel('X (grid cells)')
        plt.ylabel('Y (grid cells)')
        plt.title('Occupancy Grid Map')
        plt.colorbar(label='Occupied')
        plt.xticks(np.arange(0, self.grid_width, 1/self.resolution))
        plt.yticks(np.arange(0, self.grid_height, 1/self.resolution))
        plt.grid(True, alpha=0.3)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create a 10x10 meter map with 0.1m resolution and 20% obstacle probability
    ogm = OccupancyGridMap(width=10, height=10, resolution=1, obstacle_probability=0.2)
    ogm.generate_random_map()
    ogm.visualize()