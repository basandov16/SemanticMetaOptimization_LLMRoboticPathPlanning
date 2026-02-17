import os, math

class MapParser:
    def __init__(self, distance_to_dock=1500):
        # Original map file path
        self.org_map_path = None

        # Output map file directory
        # NOTE: To be set based on the path specified in a config file
        self.output_path_dir = "RF_Survey_Maps/"

        # Output map file path
        self.output_map_file_path = None

        # Map name
        self.map_name = ""

        # Map data to be extracted from map file (2D points & Lines & forbidden areas & Goal Points & dock position)        
        self.map_lines = None
        self.map_points = None
        self.forbidden_areas = None
        self.goal_points = None
        self.dock_pos = None
        self.lines_start_line = None

        # Occupancy Grid settings
        self.occ_grid_res = 500
        self.enable_padding = True
        self.padding_res = 500

        # Occupancy Grid
        self.occupancy_grid = None
        self.original_grid = None  

        # Map origin
        self.map_origin = None

        # Dock goal point position (to enable autonomous docking)
        self.dock_goal = None
        
        # Set distance to search for the dock in mm
        self.distance_to_dock = distance_to_dock

        # Complete Coverage Path linked list
        self.ccp_list = None
        
        self.Final_map_destination=None


        self.brush_size=3
        self.brush_type="erase"
    def set_output_directory(self, output_path_dir):
        self.output_path_dir = output_path_dir
        
    def update_brush_type_parser(self,brush_type):
        self.brush_type=brush_type

    def update_brush_size_parser(self,brush_size):
        self.brush_size=brush_size

    def set_map_path(self, map_path, output_map_name ):
        # Set original map path
        self.org_map_path = map_path
        # Set map name based on map file name
        self.map_name = output_map_name
        print(f"Map loaded: {self.org_map_path}")
        
        # Create output directory if it does not exist
        if not os.path.exists(self.output_path_dir):
            os.mkdir(self.output_path_dir) 
            print(f"Output map directory created: {self.output_path_dir}") 

        # Set output map file path
        self.output_map_file_path = os.path.join(self.output_path_dir, self.map_name + ".map")



    def create_final_path(self, source_path, destination_dir, new_name):
        # Ensure the source file exists
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"The file {source_path} does not exist.")
        
        # Ensure the destination directory exists
        if not os.path.isdir(destination_dir):
            os.makedirs(destination_dir)
        
            # # Extract the file name from the source path
        original_file_name = os.path.basename(source_path)

        base_name = new_name if new_name else original_file_name
        # 5) If you want to always have ".map" as an extension, enforce it:
        if not base_name.lower().endswith(".map"):
            base_name += ".map"

        # 6) Construct the final destination path = (folder) + (filename)
        destination_path = os.path.join(destination_dir, base_name)

        print(f"File copied to {destination_path}")
        self.map_parser.Final_map_destination = destination_path



    # Function to set map data after reading the map file
    def set_map_data(self, map_points, map_lines, forbidden_areas, goal_points, dock_pos, lines_start_line):
        self.map_points = map_points
        self.map_lines = map_lines
        self.forbidden_areas = forbidden_areas
        self.goal_points = goal_points
        self.dock_pos = dock_pos
        self.lines_start_line = lines_start_line

        # Generate dock search goal point if dock_pos exists in the original map
        if self.dock_pos:
            self.set_dock_pos()

    # Function to set occupancy grid settings (grid resolution & padding resolution)
    def set_occ_grid_settings(self, occ_grid_res, enable_padding, padding_res, path_grid_res = 1000):
        self.occ_grid_res = occ_grid_res
        self.enable_padding = enable_padding
        self.padding_res = padding_res
        self.path_grid_res = path_grid_res

    def set_occupancy_grid(self, occupancy_grid, map_origin):
        self.occupancy_grid = occupancy_grid
        self.map_origin = map_origin

    def set_original_grid(self, original_grid):
        self.original_grid = original_grid

    def reset_occupancy_grid(self):
        self.occupancy_grid = self.original_grid

    def set_points(self, x_points, y_points):
        self.points_x = x_points
        self.points_y = y_points

    def get_grid_dims(self):
        return self.occupancy_grid.shape



    def set_dock_pos(self):
        if self.dock_pos:
            # Get dock position
            x, y, heading = self.dock_pos

            # Calculate change in x and change in y
            delta_x = self.distance_to_dock * math.cos(math.radians(heading))
            delta_y = self.distance_to_dock * math.sin(math.radians(heading))

            # Calculate dock search goal point position
            dock_x = round(x + delta_x)
            dock_y = round(y + delta_y)
            dock_heading = round(math.degrees(math.atan2(-delta_y, -delta_x)))

            # Set dock search goal position
            self.dock_goal = [dock_x, dock_y, dock_heading]



    def display_info(self, brush_size):
        self.brush_size=brush_size
    

    def display_info(self):
        if self.map_name: print(f"Map name: {self.map_name}")
        if self.org_map_path: print(f"Map Path: {self.org_map_path}")
        if self.output_path_dir: print(f"Output map directory: {self.output_path_dir}")
        if self.output_map_file_path: print(f"Output map file path: {self.output_map_file_path}")
        if self.map_points: print(f"Number of map points: {len(self.map_points)}")
        if self.map_lines: print(f"Number of map lines: {len(self.map_lines)}")
        if self.forbidden_areas: print(f"Number of forbidden areas: {len(self.forbidden_areas)}")
        if self.goal_points: print(f"Number of goal points: {len(self.goal_points)}")
        if self.dock_pos: print(f"Dock position: {self.dock_pos}")
        if self.dock_goal: print(f"Dock search goal point position: {self.dock_goal}")
        if self.occ_grid_res: print(f"Occupancy grid resolution = {self.occ_grid_res} mm")
        if self.path_grid_res: print(f"CCP Path Grid Resolution = {self.path_grid_res} mm")
        if self.enable_padding: print(f"Padding resolution = {self.padding_res} mm")
        if self.occupancy_grid is not None: print(f"Occupancy grid dimensions = {self.get_grid_dims()}")
        print("\n")

    def set_occ_grid_res(self, grid_res):
        self.occ_grid_res = grid_res

    def set_path_grid_res(self, path_res):
        self.path_grid_res = path_res

    def set_padding_res(self, padding_res):
        self.padding_res = padding_res