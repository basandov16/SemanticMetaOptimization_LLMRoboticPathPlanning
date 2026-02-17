import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dotenv import load_dotenv
from openai import AzureOpenAI
from occupancy_grid_map_generator import OccupancyGridMap

load_dotenv()

class SemanticZoneGenerator:
    def __init__(self, occupancy_grid_map):
        """
        Initialize semantic zone generator.
        
        Args:
            occupancy_grid_map: OccupancyGridMap instance
        """
        self.ogm = occupancy_grid_map
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("OPENAI_API_BASE"),
            default_headers={"OpenAI-Organization": os.getenv("OPENAI_ORGANIZATION")}
        )
        self.semantic_zones = []
        self.costmap = np.ones((self.ogm.grid_height, self.ogm.grid_width))
    
    def grid_to_text(self):
        """Convert occupancy grid to text representation."""
        print("Converting occupancy grid to text representation...")
        text = f"Occupancy Grid Map ({self.ogm.grid_width}x{self.ogm.grid_height}):\n"
        text += f"Resolution: {self.ogm.resolution}m per cell\n"
        text += f"Total area: {self.ogm.width}m x {self.ogm.height}m\n\n"
        
        # Simplified representation using symbols
        for i in range(self.ogm.grid_height):
            row = ""
            for j in range(self.ogm.grid_width):
                row += "█" if self.ogm.grid[i, j] == 1 else "·"
            text += row + "\n"
        
        return text
    
    def generate_semantic_zones_with_llm(self, num_zones=5):
        """
        Use LLM to generate semantic zones based on occupancy grid.
        
        Args:
            num_zones: Number of semantic zones to generate
        """
        grid_text = self.grid_to_text()
        
        prompt = f"""Given the following occupancy grid map where '█' represents obstacles and '·' represents free space:

{grid_text}

Generate {num_zones} semantic zones for robot path planning. Each zone should:
1. Have a label (e.g., "forbidden", "high-cost", "preferred", "caution")
2. Have coordinates (x_min, y_min, x_max, y_max) in grid cells
3. Have a cost multiplier (1.0 = normal, >1.0 = higher cost, 0 = forbidden)
4. Have a brief reasoning for placement

Provide your response as a JSON array with this structure:
[
  {{
    "label": "zone_name",
    "x_min": 0,
    "y_min": 0,
    "x_max": 5,
    "y_max": 5,
    "cost_multiplier": 2.0,
    "reasoning": "explanation"
  }}
]

Consider:
- Placing "forbidden" zones near dense obstacle clusters
- "High-cost" zones in narrow passages
- "Preferred" zones in open areas
- "Caution" zones near obstacle edges

Respond ONLY with the JSON array, no additional text."""

        print("Requesting LLM to generate semantic zones...")
        response = self.client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=[
                {"role": "system", "content": "You are a robotics expert specializing in path planning and semantic mapping."},
                {"role": "user", "content": prompt}
            ],
            # temperature=0.7
        )
        
        response_text = response.choices[0].message.content.strip()
        print("\nLLM Response:")
        print(response_text)
        
        # Parse JSON response
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        self.semantic_zones = json.loads(response_text)
        return self.semantic_zones
    
    def apply_semantic_zones_to_costmap(self):
        """Apply semantic zones to create a costmap."""
        # Start with base cost of 1.0 for free space
        self.costmap = np.ones((self.ogm.grid_height, self.ogm.grid_width))
        
        # Set obstacles to infinity
        self.costmap[self.ogm.grid == 1] = np.inf
        
        print("\nApplying semantic zones to costmap:")
        for zone in self.semantic_zones:
            x_min = max(0, zone['x_min'])
            y_min = max(0, zone['y_min'])
            x_max = min(self.ogm.grid_width, zone['x_max'])
            y_max = min(self.ogm.grid_height, zone['y_max'])
            
            multiplier = zone['cost_multiplier']
            
            # Apply cost multiplier to free space only
            for i in range(y_min, y_max):
                for j in range(x_min, x_max):
                    if self.ogm.grid[i, j] == 0:  # Only free space
                        if multiplier == 0:  # Forbidden zone
                            self.costmap[i, j] = np.inf
                        else:
                            self.costmap[i, j] *= multiplier
            
            print(f"  - {zone['label']}: [{x_min}:{x_max}, {y_min}:{y_max}], cost={multiplier}x")
            print(f"    Reasoning: {zone['reasoning']}")
        
        return self.costmap
    
    def visualize_with_zones(self):
        """Visualize occupancy grid with semantic zones overlaid."""
        print("Visualizing occupancy grid with semantic zones overlaid...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Original occupancy grid
        axes[0].imshow(self.ogm.grid, cmap='binary', origin='lower')
        axes[0].set_title('Occupancy Grid')
        axes[0].set_xlabel('X (grid cells)')
        axes[0].set_ylabel('Y (grid cells)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Semantic zones overlay
        axes[1].imshow(self.ogm.grid, cmap='binary', origin='lower', alpha=0.5)
        
        # Color map for different zone types
        zone_colors = {
            'forbidden': 'red',
            'high-cost': 'orange',
            'caution': 'yellow',
            'preferred': 'green',
            'default': 'blue'
        }
        
        for zone in self.semantic_zones:
            x_min, y_min = zone['x_min'], zone['y_min']
            width = zone['x_max'] - zone['x_min']
            height = zone['y_max'] - zone['y_min']
            
            # Determine color based on label
            color = zone_colors.get(zone['label'].lower(), zone_colors['default'])
            
            rect = Rectangle((x_min - 0.5, y_min - 0.5), width, height,
                           linewidth=2, edgecolor=color, facecolor=color,
                           alpha=0.3, label=zone['label'])
            axes[1].add_patch(rect)
        
        axes[1].set_title('Semantic Zones')
        axes[1].set_xlabel('X (grid cells)')
        axes[1].set_ylabel('Y (grid cells)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Costmap
        costmap_display = np.copy(self.costmap)
        costmap_display[costmap_display == np.inf] = np.nanmax(costmap_display[costmap_display != np.inf]) * 2
        
        im = axes[2].imshow(costmap_display, cmap='hot', origin='lower')
        axes[2].set_title('Costmap')
        axes[2].set_xlabel('X (grid cells)')
        axes[2].set_ylabel('Y (grid cells)')
        plt.colorbar(im, ax=axes[2], label='Cost')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create occupancy grid
    # ogm = OccupancyGridMap(width=10, height=10, resolution=1, obstacle_probability=0.2)
    # ogm.generate_random_map()
    # ogm.visualize()

    # Load from Omron .map file
    ogm = OccupancyGridMap.from_omron_map("./src/Base_Maps/input/input.map", occ_grid_res_mm=100, padding_mm=600, enable_padding=True)
    ogm.visualize()
    
    # Generate semantic zones using LLM
    szg = SemanticZoneGenerator(ogm)
    zones = szg.generate_semantic_zones_with_llm(num_zones=5)
    
    # Apply zones to create costmap
    costmap = szg.apply_semantic_zones_to_costmap()
    
    # Visualize all steps
    szg.visualize_with_zones()