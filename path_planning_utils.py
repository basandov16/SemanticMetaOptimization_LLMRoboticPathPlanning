import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
# from PathPermTree import PathPermTree, PathPermTreeNode
from utils import *

# from PyQt5.QtWidgets import QVBoxLayout

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# Configuration flags
FILTER_BOUNDARIES = True
PADDING = True

# Flags for pathing algorithm optimizations
REMOVE_REDUNDANT_GOALPOINTS = True
DISTANCE_OPTIMIZATION = True
DO_MAX_WINDOW_SIZE = 9
SIMPLIFY_STRAIGHT_LINES = True

DOCK_POS_WORD = "DockPos"

# Visualization configuration flags
PLOT_POINTS = True
PLOT_LINES = True
REMOVE_OFFSET = True
DISPLAY_AXES = False
DISPLAY_LEGEND = False

# Helper function to read Omron's map file and return all the map file's lines
def read_map_file(map_file_path):

    # Open and read map file from path
    with open(map_file_path, 'r') as file:
        input_file_lines = file.readlines()

    first_forbidden_area = True

    forbidden_area_line_list = []

    # Read map file and get the first line index of each data segment type
    data_start_line, lines_start_line, forbidden_start_line, goals_start_line, goals_line_end = get_seg_start(input_file_lines, first_forbidden_area, forbidden_area_line_list)

    # Get map lines segment from the loaded map file
    lines_text = input_file_lines[lines_start_line+1:data_start_line]

    # Get map points text segment from the loaded map file
    points_text = input_file_lines[data_start_line + 1::]

    # Get forbidden region if it exists
    if forbidden_start_line != -1:
        forbidden_area_text = [input_file_lines[i] for i in forbidden_area_line_list]

    # Get goal points (specifically DockPos) if exists
    if goals_start_line != -1:
        goals_text = input_file_lines[goals_start_line : goals_line_end + 1]
    
    # Initialize list of map 2D points
    map_points = []

    # Initialize list of map lines
    lines_x = []
    lines_y = []
    
    # Initialize list of forbidden region data (angle, top-left & bottom-right corners)
    forbidden_areas = []

    # Initialize a dict for goal point(s) represented by their
    goal_points = {}

    # Initialize variable for dock pos goal point
    dock_pos = None

    # Extract 2D map points from text segment
    for text_line in points_text:
        # Note: populated `x` and `y` variables instead of a list
        x, y = text_line.strip().split(' ')
        map_points.append((int(x), int(y)))
    
    # Extract map lines from text segment
    for text_line in lines_text:
        x1, y1, x2, y2 = text_line.strip().split(' ')
        lines_y.append([int(y1), int(y2)])
        lines_x.append([int(x1), int(x2)])
    
    # Extract forbidden region points
    if forbidden_start_line != -1:
        for text_line in forbidden_area_text:
            text_line = text_line.split()
            forbidden_areas.append([float(text_line[4]), (int(text_line[-4]), int(text_line[-3])), (int(text_line[-2]), int(text_line[-1]))])
    # Extract all goal points from text segment
    if goals_start_line != -1:
        for text_line in goals_text:
            # Get line identified
            identifier = text_line.split(' ')[1]
            if identifier in ["Goal", "GoalWithHeading", "DockLynx"]:  
                # Extract goal point name
                goal_name = text_line.split(' ')[-1].split('"')[1]
                x, y, heading = text_line.split(' ')[2:5]
                # Add goal point and its coordinates to dict
                goal_points[goal_name] = [int(x), int(y), float(heading)]

                # Check if goal point is "DockPos"
                if goal_name == DOCK_POS_WORD:
                    dock_pos = [int(x), int(y), float(heading)]

    # Combine line elements into a list
    map_lines = [lines_x, lines_y]

    return map_points, map_lines, forbidden_areas, goal_points, dock_pos, lines_start_line

# Function to plot map
def plot_map(map_name, map_points, map_lines):
    points_x, points_y = map(list, zip(*map_points))

    lines_x = map_lines[0]
    lines_y = map_lines[1]
    # Plot lines
    fig = plt.figure(frameon=False)
    
    # Fix aspect ratio of the plot
    fig.gca().set_aspect('equal')
    
    if PLOT_POINTS:
        # Remove offset and plot points
        plt.plot(points_x, points_y, '.', color='Black', markersize=1, label="LiDAR Points")  

    if PLOT_LINES:
        num_lines = len(lines_x)
        # Plot lines
        for i in range(num_lines):
            # Plot Lines
            if i == 0:
                plt.plot(lines_x[i], lines_y[i], color = 'Black', label="Lines")
            else:
                plt.plot(lines_x[i], lines_y[i], color = 'Black')

    if DISPLAY_AXES is False:
        plt.axis('off')

    if DISPLAY_LEGEND is False:
        fig.legend([]).set_visible(False)

    # Add title and axes labels
    plt.title(f"Map: {map_name}")
    plt.xlabel("x-coordinate (mm)")
    plt.ylabel("y-coordinate (mm)")

    plt.show()

def create_occupancy_grid_v2(map_points, occ_grid_res, add_padding = True, padding_amount = 500, forbidden_areas = []):

    all_map_points = map_points

    # Generate additional map points based on forbidden areas and selected grid resolution
    if forbidden_areas:
        forbidden_map_points = gen_forbidden_area_pts(forbidden_areas, occ_grid_res)
        all_map_points += forbidden_map_points

    # Find minimum and maximum coordinates of the map
    min_x_coordinate_mm = min(all_map_points, key=lambda x: x[0])[0]
    min_y_coordinate_mm = min(all_map_points, key=lambda x: x[1])[1]
    max_x_coordinate_mm = max(all_map_points, key=lambda x: x[0])[0]
    max_y_coordinate_mm = max(all_map_points, key=lambda x: x[1])[1]

    # Define origin point
    map_origin = (min_x_coordinate_mm, min_y_coordinate_mm)

    # Calculate the grid's width and height
    grid_width, grid_height = calculate_grid_resolution(min_x_coordinate_mm, min_y_coordinate_mm, max_x_coordinate_mm, max_y_coordinate_mm, occ_grid_res)  

    # Generate occupancy grid and add padding
    occ_grid = gen_occupancy_grid(all_map_points, grid_width, grid_height, map_origin, occ_grid_res)
    if add_padding:
        occ_grid = dilate_grid_v2(occ_grid, occ_grid_res, padding_amount)

    if FILTER_BOUNDARIES:

        # Filter boundaries of the map
        filter_map_boundaries(occ_grid)

        # Remove untraversable spaces in the occupancy grid
        filter_untraversable_space(occ_grid)

    return occ_grid, map_origin

    


# Function to generate an occupancy grid based on the selected grid and padding resolutions
def create_occupancy_grid(map_points, occ_grid_res, add_padding = True, padding_grid_res = 250, forbidden_areas = []):
    all_map_points = map_points
    # Generate additional map points based on forbidden areas and selected grid resolution
    if forbidden_areas:
        forbidden_map_points = gen_forbidden_area_pts(forbidden_areas, occ_grid_res)
        # Append to list of map points
        all_map_points += forbidden_map_points

    # TODO: Read min and max coordinates from the map file
    # Find minimum and maximum coordinates of the map
    min_x_coordinate_mm = min(all_map_points, key=lambda x: x[0])[0]
    min_y_coordinate_mm = min(all_map_points, key=lambda x: x[1])[1]
    max_x_coordinate_mm = max(all_map_points, key=lambda x: x[0])[0]
    max_y_coordinate_mm = max(all_map_points, key=lambda x: x[1])[1]

    # Define origin point
    map_origin = (min_x_coordinate_mm, min_y_coordinate_mm)

    # Calculate the grid's width and height
    grid_width, grid_height = calculate_grid_resolution(min_x_coordinate_mm, min_y_coordinate_mm, max_x_coordinate_mm, max_y_coordinate_mm, occ_grid_res)    

    if add_padding:
        # Pad the obstacles throughout the map
        high_res_occupancy_grid, padded_occupancy_grid = pad_occupancy_grid(all_map_points, min_x_coordinate_mm, min_y_coordinate_mm, max_x_coordinate_mm, max_y_coordinate_mm, padding_grid_res)

        # Convert the high-resolution padded occupancy grid to the final desired grid resolution
        occupancy_grid = downsample_occupancy_grid(padded_occupancy_grid, padding_grid_res, occ_grid_res)

    else:
        # Generate occupancy grid
        occupancy_grid = gen_occupancy_grid(all_map_points, grid_width, grid_height, map_origin, occ_grid_res)

    if FILTER_BOUNDARIES:
        # Filter boundaries of the map
        filter_map_boundaries(occupancy_grid)

        # Remove untraversable spaces in the occupancy grid
        filter_untraversable_space(occupancy_grid)

    return occupancy_grid, map_origin

def gen_dock_pos(x, y, heading, distance_to_dock=750):
    delta_x = distance_to_dock * math.cos(math.radians(heading))
    delta_y = distance_to_dock * math.sin(math.radians(heading))

    dock_x = x + delta_x
    dock_y = y + delta_y

    dock_heading = math.degrees(math.atan2(y - dock_y, x - dock_x))
    dock_heading = math.degrees(math.atan2(-delta_y, -delta_x))

    print("x = ", dock_x)
    print("y = ", dock_y)
    print("theta = ", dock_heading)
    return dock_x, dock_y, dock_heading

def gen_loose_hilbert_path(occ_grid, occ_cell_size, path_cell_size):

    occ_grid_width, occ_grid_height = occ_grid.shape
    scanned_area = np.zeros((occ_grid_width, occ_grid_height), dtype=np.int8)

    # find the size of the path planning grid
    scale_factor = path_cell_size / occ_cell_size
    scale_factor_2 = max(1, scale_factor // 2)
    path_grid_width = ceil(occ_grid_width / scale_factor)
    path_grid_height = ceil(occ_grid_height / scale_factor)

    # generate hilbert curve linked list
    path = generate_hilbert_curve_linked_list(path_grid_width, path_grid_height, scale_factor)

    # traverse LL
    current_node = path.head
    previous_node = None
    while current_node:

        x = current_node.coordinate[0]
        y = current_node.coordinate[1]

        # check if occupancy grid is occupied at the goalpoint
        if occ_grid[floor(x), floor(y)]:

            # check if there is a valid unoccupied space up to 1/2 of the path planning cell size away
            #xmin, xmax = int(max(0, floor(x)-scale_factor_2)), int(min(occ_grid_width, floor(x)+scale_factor_2+1))
            #ymin, ymax = int(max(0, floor(y)-scale_factor_2)), int(min(occ_grid_height, floor(y)+scale_factor_2+1))
            #search_area = occ_grid[xmin:xmax, ymin:ymax]
            search_area = get_search_region(occ_grid, x, y, scale_factor)
            adjust = spiral_search_matrix_zero(search_area)

            # if waypoint can be moved successfully, adjust its coordinates
            if adjust:
                x += adjust[0]
                y += adjust[1]
                current_node.coordinate = (x, y)

            # if waypoint was not able to be moved, remove it
            else:
                next_node = path.remove_current_node(current_node, previous_node)
                current_node = next_node
                continue

        # remove the waypoint if it is too close to an already existing waypoint
        if REMOVE_REDUNDANT_GOALPOINTS:
            if scanned_area[floor(x), floor(y)]:
                next_node = path.remove_current_node(current_node, previous_node)
                current_node = next_node
                continue

        # set the area around the waypoint to already scanned, up to 1/2 of the path planning cell size away
        xmin, xmax = int(max(0, floor(x)-scale_factor_2)), int(min(occ_grid_width, floor(x)+scale_factor_2+1))
        ymin, ymax = int(max(0, floor(y)-scale_factor_2)), int(min(occ_grid_width, floor(y)+scale_factor_2+1))
        scanned_area[xmin:xmax, ymin:ymax] = 1

        previous_node = current_node
        current_node = current_node.next

    # optimize the path length using short-window pruned permutation trees
    if DISTANCE_OPTIMIZATION:
        
        # create initial path from beginning of list up to max window size
        subpath = [path.head]
        while (len(subpath) < DO_MAX_WINDOW_SIZE) and all(subpath):
            subpath.append(subpath[-1].next)

        # while path is valid (all nodes exist)
        while all(subpath):

            # get coordinates of all nodes in path
            coordinate_list = []
            for i in range(len(subpath)):
                coordinate_list.append(subpath[i].coordinate)

            # make pruned permutation tree to find shortest subpath
            tree = PathPermTree()
            tree.make_pruned_tree(coordinate_list)
            new_order = tree.best_path

            # rearrange subpath nodes in the LL
            for i in range(len(new_order)-1):
                subpath[new_order[i]].next = subpath[new_order[i+1]]

            # shift the sliding window by 1 node
            for i in range(len(subpath)):
                subpath[i] = subpath[i].next

    # remove redundant nodes that lie in the middle of a straight line
    if SIMPLIFY_STRAIGHT_LINES:

        previous_node = path.head
        current_node = previous_node.next
        next_node = current_node.next

        # slide window along path
        while previous_node and current_node and next_node:

            px = previous_node.coordinate[0]
            py = previous_node.coordinate[1]
            x = current_node.coordinate[0]
            y = current_node.coordinate[1]
            nx = next_node.coordinate[0]
            ny = next_node.coordinate[1]

            # check if all nodes lie on a vertical or horizontal line, with a tolerance of up to 2 occupancy grid tiles
            x_line = abs(px-x)<3 and abs(nx-x)<3 and abs(nx-px)<3
            y_line = abs(py-y)<3 and abs(ny-y)<3 and abs(ny-py)<3

            # check if nodes are lined up sequentially (i.e. current node must be between next and previous nodes)
            x_seq = (px<x<nx) or (px>x>nx)
            y_seq = (py<y<ny) or (py>y>ny)

            # if nodes lie on a line and are sequential, the middle waypoint can be removed
            if (x_line and y_seq) or (y_line and x_seq):
                path.remove_current_node(current_node, previous_node)
                current_node = current_node.next
                next_node = current_node.next
            else:
                previous_node = previous_node.next
                current_node = previous_node.next
                next_node = current_node.next

    return path




def gen_hilbert_path(occupancy_grid, occ_grid_resolution):
    coordinates_indices = []

    occupied_indices = np.argwhere(occupancy_grid)
    coordinates_indices.extend(map(tuple, occupied_indices))
    
    #(min_x_unoccupied, max_x_unoccupied, min_y_unoccupied, max_y_unoccupied) = find_zero_indices(filled_binary_grid)
    #grid_width = max_x_unoccupied - min_x_unoccupied

    grid_width, grid_height = occupancy_grid.shape

    scale = 5  # Adjust scale for better visualization
    base_hilbert_linked_list = generate_hilbert_curve_linked_list(grid_width, grid_height, scale)
    unmodified_linked_list = generate_hilbert_curve_linked_list(grid_width, grid_height, scale)

    for occupied_coordinate in coordinates_indices:
        base_hilbert_linked_list.remove_node(occupied_coordinate)

    path_length_units = base_hilbert_linked_list.calculate_total_path_length()


    print("Total Path Length =", path_length_units, "units")

    print("Total Path Length =", path_length_units * occ_grid_resolution, "mm")

    return base_hilbert_linked_list

def generate_rf_map(map_parser):
    # Generate new map file to modify
    print(map_parser.org_map_path)
    print(map_parser.output_path_dir)
    print(map_parser.map_name)
    copy_file(map_parser, map_parser.org_map_path, map_parser.output_path_dir, map_parser.map_name)

    # Read output map file lines
    map_file_lines = read_file(map_parser.output_map_file_path)

    # Add complete coverage path goal points and dock search goal position
    add_ccp_to_map(map_parser.ccp_list, map_parser.occ_grid_res, map_parser.map_origin, map_file_lines, map_parser.lines_start_line)

    # Add dock search goal point to RF Survey Map file if exists
    if map_parser.dock_goal:
        add_dock_goal(*map_parser.dock_goal, map_file_lines, map_parser.lines_start_line)

    # Export update map text data to RF Survey Map file
    export_map_file(map_parser.output_map_file_path, map_file_lines)

    print(f"RF Survey Map File successfully generated at: {map_parser.output_map_file_path}")


    
def convert_to_raw_coordinate(grid_coordinate, grid_resolution, origin):
    index_x, index_y = grid_coordinate
    
    # Convert indices to coordinates in millimeters
    coordinate_dx = index_x * grid_resolution
    coordinate_dy = index_y * grid_resolution

    # Calculate the coordinates in millimeters
    x_mm = origin[0] + coordinate_dx + 0.5 * grid_resolution
    y_mm = origin[1] + coordinate_dy + 0.5 * grid_resolution

    return int(x_mm), int(y_mm)    




def plot_map_with_path(map_name, map_points, map_lines, path_linked_list, grid_resolution, origin, qframe):
    """
    Plots the map with points, lines, and overlays the robot's path inside a QFrame.
    """

    if qframe.layout() is None:
        qframe.setLayout(QVBoxLayout())  # Create a new layout if missing
    else:
        while qframe.layout().count():
            item = qframe.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    # ax.set_aspect(2)        # Y-axis is 2× taller than X-axis

    points_x, points_y = map(list, zip(*map_points)) if map_points else ([], [])

    lines_x, lines_y = map_lines if map_lines else ([], [])

    labels = []

    if PLOT_POINTS and points_x and points_y:
        ax.plot(points_x, points_y, '.', color='Black', markersize=1, label="LiDAR Points")
        labels.append("LiDAR Points")

    if PLOT_LINES and lines_x and lines_y:
        for i in range(len(lines_x)):
            label = "Lines" if i == 0 else None  # Avoid duplicate labels
            ax.plot(lines_x[i], lines_y[i], color='Black', label=label)
            if i == 0:
                labels.append("Lines")

    if path_linked_list and len(path_linked_list) > 1:
        cmap = plt.get_cmap('inferno')
        norm = mcolors.Normalize(vmin=0, vmax=len(path_linked_list) - 1)

        current_node = path_linked_list.head
        i = 0  # Index for colormap scaling
        while current_node and current_node.next:
            color = cmap(floor(i))  # Get color from colormap
            x1, y1 = convert_to_raw_coordinate(current_node.coordinate, grid_resolution, origin)
            x2, y2 = convert_to_raw_coordinate(current_node.next.coordinate, grid_resolution, origin)
            label = "Path" if i == 0 else None  # Avoid duplicate labels
            ax.plot((x1, x2), (y1, y2), color=color, linewidth=2, label=label) #plot line
            ax.plot(x1, y1, 's', color=color) #plot point
            if i == 0:
                labels.append("Path")
            current_node = current_node.next
            i += 224/len(path_linked_list)  # Increment index for color mapping

        #  Mark Start and End Points
        start_x, start_y = convert_to_raw_coordinate(path_linked_list.head.coordinate, grid_resolution, origin)
        end_x, end_y = convert_to_raw_coordinate(current_node.coordinate, grid_resolution, origin)  # Last node's coordinates

        ax.scatter(start_x, start_y, color=cmap(norm(0)), s=150, marker='*', label="Start Position")
        ax.scatter(end_x, end_y, color=cmap(224), s=150, marker='D', label="End Position")
        labels.extend(["Start Position", "End Position"])

    # print(f" Legend Items: {labels}")

    if labels:
        ax.legend(loc='upper left', fontsize=10, frameon=True, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
   

    plt.title(f"Map: {map_name} with RF Survey Path")
    plt.xlabel("x-coordinate (mm)")
    plt.ylabel("y-coordinate (mm)")

    canvas = FigureCanvas(fig)
    qframe.layout().addWidget(canvas)

    # Close figure to free memory
    #plt.close(fig)


# # Helper function to read Omron's map file and return all the map file's lines
# def read_map_file_rf(map_file_path):
#     # Open and read map file from path
#     DOCK_POS_WORD = "DockPos"

#     with open(map_file_path, 'r') as file:
#         input_file_lines = file.readlines()

#     first_forbidden_area = True

#     forbidden_area_line_list = []

#     # Read map file and get the first line index of each data segment type
#     data_start_line, lines_start_line, forbidden_start_line, goals_start_line, goals_line_end = get_seg_start(input_file_lines, first_forbidden_area, forbidden_area_line_list)

#     # Get map lines segment from the loaded map file
#     lines_text = input_file_lines[lines_start_line+1:data_start_line]

#     # Get map points text segment from the loaded map file
#     points_text = input_file_lines[data_start_line + 1::]

#     # Get forbidden region if it exists
#     if forbidden_start_line != -1:
#         forbidden_area_text = [input_file_lines[i] for i in forbidden_area_line_list]

#     # Get goal points (specifically DockPos) if exists
#     if goals_start_line != -1:
#         goals_text = input_file_lines[goals_start_line : goals_line_end + 1]
    
#     # Initialize list of map 2D points
#     map_points = []

#     # Initialize list of map lines
#     lines_x = []
#     lines_y = []
    
#     # Initialize list of forbidden region data (angle, top-left & bottom-right corners)
#     forbidden_areas = []

#     # Initialize a dict for goal point(s) represented by their
#     goal_points = {}

#     # Initialize variable for dock pos goal point
#     dock_pos = None

#     # Extract 2D map points from text segment
#     for text_line in points_text:
#         # Note: populated `x` and `y` variables instead of a list
#         x, y = text_line.strip().split(' ')
#         map_points.append((int(x), int(y)))
    
#     # Extract map lines from text segment
#     for text_line in lines_text:
#         x1, y1, x2, y2 = text_line.strip().split(' ')
#         lines_y.append([int(y1), int(y2)])
#         lines_x.append([int(x1), int(x2)])
    
#     # Extract forbidden region points
#     if forbidden_start_line != -1:
#         for text_line in forbidden_area_text:
#             text_line = text_line.split()
#             forbidden_areas.append([float(text_line[4]), (int(text_line[-4]), int(text_line[-3])), (int(text_line[-2]), int(text_line[-1]))])
    
#     # Extract all goal points from text segment
#     if goals_start_line != -1:
#         for text_line in goals_text:
#             # Extract goal point name
#             goal_name = text_line.split(' ')[-1].split('"')[1]
#             x, y, heading = text_line.split(' ')[2:5]
#             # Add goal point and its coordinates to dict
#             goal_points[goal_name] = [int(x), int(y), float(heading)]

#             # Check if goal point is "DockPos"
#             if goal_name == DOCK_POS_WORD:
#                 dock_pos = [int(x), int(y), float(heading)]

#     # Combine line elements into a list
#     map_lines = [lines_x, lines_y]

#     return map_points, map_lines, forbidden_areas, goal_points, dock_pos, lines_start_line

# def plot_map_with_path(map_name, map_points, map_lines, path_linked_list, grid_resolution, origin, qframe):
#     """
#     Plots the map with points, lines, and overlays the robot's path inside a QFrame.

#     Args:
#         map_name (str): Name of the map.
#         map_points (list of tuples): List of (x, y) coordinates representing points.
#         map_lines (tuple of lists): Tuple containing lists of x and y coordinates for lines.
#         path_linked_list (LinkedList): Linked List containing the path nodes.
#         grid_resolution (float): Grid resolution.
#         origin (tuple): Origin coordinate (x, y).
#         qframe (QFrame): The QFrame widget where the plot should be displayed.
#     """

#     #  Ensure QFrame has a layout
#     if qframe.layout() is None:
#         qframe.setLayout(QVBoxLayout())  # Create a new layout if missing
#     else:
#         #  Clear existing widgets in layout
#         while qframe.layout().count():
#             item = qframe.layout().takeAt(0)
#             if item.widget():
#                 item.widget().deleteLater()

#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.set_aspect('equal')

#     # Unpack map points
#     points_x, points_y = map(list, zip(*map_points)) if map_points else ([], [])

#     # Unpack map lines
#     lines_x, lines_y = map_lines if map_lines else ([], [])

#     # Plot map points
#     if PLOT_POINTS:
#         ax.plot(points_x, points_y, '.', color='Black', markersize=1, label="LiDAR Points")

#     # Plot map lines
#     if PLOT_LINES:
#         for i in range(len(lines_x)):
#             ax.plot(lines_x[i], lines_y[i], color='Black', label="Lines" if i == 0 else "")

#     # Plot path with colormap
#     if path_linked_list and len(path_linked_list) > 1:
#         cmap = plt.get_cmap('inferno')
#         norm = mcolors.Normalize(vmin=0, vmax=len(path_linked_list) - 1)

#         current_node = path_linked_list.head
#         i = 0  # Index for colormap scaling
#         while current_node and current_node.next:
#             color = cmap(norm(i))  # Get color from colormap
#             x1, y1 = convert_to_raw_coordinate(current_node.coordinate, grid_resolution, origin)
#             x2, y2 = convert_to_raw_coordinate(current_node.next.coordinate, grid_resolution, origin)
#             ax.plot((x1, x2), (y1, y2), color=color, linewidth=2)
#             current_node = current_node.next
#             i += 1  # Increment index for color mapping

#         # Mark Start and End Points
#         start_x, start_y = convert_to_raw_coordinate(path_linked_list.head.coordinate, grid_resolution, origin)
#         end_x, end_y = convert_to_raw_coordinate(current_node.coordinate, grid_resolution, origin)  # Last node's coordinates

#         ax.scatter(start_x, start_y, color=cmap(norm(0)), s=150, marker='*', label="Start Position")  # Green circle
#         ax.scatter(end_x, end_y, color=cmap(norm(len(path_linked_list)-1)), s=150, marker='D', label="End Position")  # Red 'X'


#     if DISPLAY_LEGEND:
#         handles, labels = ax.get_legend_handles_labels()
#         if labels:  # ✅ Ensure there are labels before adding the legend
#             ax.legend(loc='upper right', fontsize=10, frameon=True)

#     # Add title and labels
#     plt.title(f"Map: {map_name} with RF Survey Path")
#     plt.xlabel("x-coordinate (mm)")
#     plt.ylabel("y-coordinate (mm)")
  
#     # Embed the Matplotlib figure inside QFrame using a FigureCanvas
#     canvas = FigureCanvas(fig)
#     qframe.layout().addWidget(canvas)

#     #Close figure to free memory
#     plt.close(fig)




    
# def plot_rf_map_with_path(map_name, map_points, map_lines, path_linked_list, grid_resolution, origin):
#     """
#     Plots the map with points and lines, then overlays the robot's expected path.

#     Args:
#         map_name (str): Name of the map.
#         map_points (list of tuples): List of (x, y) coordinates representing points.
#         map_lines (tuple of lists): Tuple containing lists of x and y coordinates for lines.
#         path_linked_list (LinkedList): Linked List containing the path nodes.
#     """

#     # Unpack map points
#     points_x, points_y = map(list, zip(*map_points))

#     # Unpack map lines
#     lines_x, lines_y = map_lines

#     fig, ax = plt.subplots()
#     ax.set_aspect('equal')

#     # Plot map points (optional)
#     if PLOT_POINTS:
#         ax.plot(points_x, points_y, '.', color='Black', markersize=1, label="LiDAR Points")  

#     # Plot map lines
#     if PLOT_LINES:
#         for i in range(len(lines_x)):
#             ax.plot(lines_x[i], lines_y[i], color='Black', label="Lines" if i == 0 else "")

#     # Plot path with colormap
#     if path_linked_list and len(path_linked_list) > 1:
#         cmap = plt.get_cmap('inferno')
#         norm = mcolors.Normalize(vmin=0, vmax=len(path_linked_list) - 1)

#         current_node = path_linked_list.head
#         i = 0  # Index for colormap scaling
#         while current_node and current_node.next:
#             color = cmap(norm(i))  # Get color from colormap
#             x1, y1 = convert_to_raw_coordinate(current_node.coordinate, grid_resolution, origin)
#             x2, y2 = convert_to_raw_coordinate(current_node.next.coordinate, grid_resolution, origin)
#             ax.plot((x1, x2), (y1, y2), color=color, linewidth=2)
#             current_node = current_node.next
#             i += 1  # Increment index for color mapping

#         # Mark Start and End Points
#         start_x, start_y = convert_to_raw_coordinate(path_linked_list.head.coordinate, grid_resolution, origin)
#         end_x, end_y = convert_to_raw_coordinate(current_node.coordinate, grid_resolution, origin)  # Last node's coordinates

#         ax.scatter(start_x, start_y, color=cmap(norm(0)), s=150, marker='*', label="Start Position")  # Green circle
#         ax.scatter(end_x, end_y, color=cmap(norm(len(path_linked_list)-1)), s=150, marker='D', label="End Position")  # Red 'X'


#     # Hide axes if needed
#     if DISPLAY_AXES is False:
#         plt.axis('off')

#     if DISPLAY_LEGEND:
#         ax.legend()

#     # Add title and labels
#     plt.title(f"Map: {map_name} with RF Survey Path")
#     plt.xlabel("x-coordinate (mm)")
#     plt.ylabel("y-coordinate (mm)")

#     plt.show()



# def plot_rf_map_with_path_reg(map_name, map_points, map_lines, path_linked_list, grid_resolution, origin):
#     """
#     Plots the map with points and lines, then overlays the robot's expected path.

#     Args:
#         map_name (str): Name of the map.
#         map_points (list of tuples): List of (x, y) coordinates representing points.
#         map_lines (tuple of lists): Tuple containing lists of x and y coordinates for lines.
#         path_linked_list (LinkedList): Linked List containing the path nodes.
#     """

#     # Unpack map points
#     points_x, points_y = map(list, zip(*map_points))

#     # Unpack map lines
#     lines_x, lines_y = map_lines

#     fig, ax = plt.subplots()
#     ax.set_aspect('equal')

#     # Plot map points (optional)
#     if PLOT_POINTS:
#         ax.plot(points_x, points_y, '.', color='Black', markersize=1, label="LiDAR Points")  

#     # Plot map lines
#     if PLOT_LINES:
#         for i in range(len(lines_x)):
#             ax.plot(lines_x[i], lines_y[i], color='Black', label="Lines" if i == 0 else "")

#     # Plot path with colormap
#     if path_linked_list and len(path_linked_list) > 1:
#         cmap = plt.get_cmap('inferno')
#         norm = mcolors.Normalize(vmin=0, vmax=len(path_linked_list) - 1)

#         current_node = path_linked_list.head
#         i = 0  # Index for colormap scaling
#         while current_node and current_node.next:
#             color = cmap(norm(i))  # Get color from colormap
#             x1, y1 = convert_to_raw_coordinate(current_node.coordinate, grid_resolution, origin)
#             x2, y2 = convert_to_raw_coordinate(current_node.next.coordinate, grid_resolution, origin)
#             ax.plot((x1, x2), (y1, y2), color=color, linewidth=2)
#             current_node = current_node.next
#             i += 1  # Increment index for color mapping

#         # Mark Start and End Points
#         start_x, start_y = convert_to_raw_coordinate(path_linked_list.head.coordinate, grid_resolution, origin)
#         end_x, end_y = convert_to_raw_coordinate(current_node.coordinate, grid_resolution, origin)  # Last node's coordinates

#         ax.scatter(start_x, start_y, color=cmap(norm(0)), s=150, marker='*', label="Start Position")  # Green circle
#         ax.scatter(end_x, end_y, color=cmap(norm(len(path_linked_list)-1)), s=150, marker='D', label="End Position")  # Red 'X'


#     # Hide axes if needed
#     if DISPLAY_AXES is False:
#         plt.axis('off')

#     if DISPLAY_LEGEND:
#         ax.legend()

#     # Add title and labels
#     plt.title(f"Map: {map_name} with RF Survey Path")
#     plt.xlabel("x-coordinate (mm)")
#     plt.ylabel("y-coordinate (mm)")

#     plt.show()
