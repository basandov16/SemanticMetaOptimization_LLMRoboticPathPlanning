import os
import numpy as np
import matplotlib.pyplot as plt

from map_parser import MapParser
from path_planning_utils import create_occupancy_grid_v2, read_map_file

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
    
    @classmethod
    def from_omron_map(cls, map_file_path, occ_grid_res_mm=100, padding_mm=600, enable_padding=True):
        """
        Build an OccupancyGridMap from an Omron .map using occMapGenerator (no GUI).
        Assumes mapParser.py and path_planning_utils.py are in the same directory.
        """
        print(f"Building occupancy grid map from Omron .map file: {map_file_path} with resolution={occ_grid_res_mm}mm, padding={padding_mm}mm, enable_padding={enable_padding}...")

        # Initialize map parser and set map path
        map_parser = MapParser()
        map_parser.set_map_path(map_file_path, "output/output")
        map_parser.set_map_data(*read_map_file(map_parser.org_map_path))

        map_parser.set_occ_grid_settings(occ_grid_res_mm, enable_padding, padding_mm)

        occupancy_grid, map_origin = create_occupancy_grid_v2(
            map_parser.map_points,
            map_parser.occ_grid_res,
            map_parser.enable_padding,
            map_parser.padding_res,
            map_parser.forbidden_areas
        )

        # occMapGenerator grids are (x, y); convert to (row=y, col=x)
        grid = occupancy_grid.T.astype(int)

        # Normalize Omron grid to 0=free, 1=occupied
        # Omron: -1 = occupied, 0 = free
        grid = (grid < 0).astype(int)

        # Calculate map dimensions in meters based on grid size and resolution
        resolution_m = occ_grid_res_mm / 1000.0
        grid_height, grid_width = grid.shape
        width_m = grid_width * resolution_m
        height_m = grid_height * resolution_m

        # Create and return the OccupancyGridMap instance
        ogm = cls(width=width_m, height=height_m, resolution=resolution_m, obstacle_probability=0.0)
        ogm.grid = grid
        return ogm
    
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
    # ogm = OccupancyGridMap(width=10, height=10, resolution=1, obstacle_probability=0.2)
    # ogm.generate_random_map()
    ogm = OccupancyGridMap.from_omron_map("./src/Base_Maps/input/input.map", occ_grid_res_mm=100, padding_mm=600, enable_padding=True)
    ogm.visualize()