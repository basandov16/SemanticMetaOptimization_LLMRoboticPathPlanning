import os
import numpy as np
import turtle
import shutil
from math import sin, cos, radians, ceil, floor, dist
# from LinkedList import LinkedList
from matplotlib import pyplot as plt
from collections import deque
from scipy import ndimage
from scipy import stats
# from itertools import permutations

DATA_WORD = 'DATA\n'
LINES_WORD = 'LINES\n'
FORBIDDEN_WORD = 'Cairn: ForbiddenArea'
GOALHEADING_WORD = 'Cairn: GoalWithHeading'
GOAL_WORD = 'Cairn: Goal'
DOCK_GOAL_WORD = 'WiseBot-Dock'

def rotate_point_around_center(point, center, angle):

    angle = radians(angle)

    x,y = point
    center_x, center_y = center

    xp = (x - center_x) * cos(angle) - (y - center_y) * sin(angle) + center_x
    yp = (x - center_x) * sin(angle) + (y - center_y) * cos(angle) + center_y

    return (xp, yp)

def rotate_list_around_center(points, center, angle):

    rotated_list = []

    for point in points:
        rotated_list.append(rotate_point_around_center(point, center, angle))

    return rotated_list

def get_centroid_of_rectangle(corner, opposite):

    x1, y1 = corner
    x2, y2 = opposite

    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    return (center_x, center_y)

def interpolate_across_rectangle(corner, opposite, spacing):
    x1, y1 = corner
    x2, y2 = opposite
    
    # Ensure x1, y1 is the bottom-left and x2, y2 is the top-right
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    
    grid = []
    
    # Generate grid points
    x = min_x
    while x <= max_x:
        y = min_y
        while y <= max_y:
            grid.append((x, y))
            y += spacing
        x += spacing
    
    return grid

def rotate_and_interpolate_forbidden_area_element(forbidden_area_element, spacing):
    angle, corner, opposite = forbidden_area_element
    rectangle_corner_list = [corner, opposite]

    rectangle_corner_list = rotate_list_around_center(rectangle_corner_list, (0,0), angle)

    centroid = get_centroid_of_rectangle(rectangle_corner_list[0], rectangle_corner_list[1])

    rectangle_corner_list = rotate_list_around_center(rectangle_corner_list, centroid, 90 - angle)

    covering_points = interpolate_across_rectangle(rectangle_corner_list[0], rectangle_corner_list[1], spacing)

    covering_points = rectangle_corner_list + covering_points
    
    covering_points = rotate_list_around_center(covering_points, centroid, -(90 - angle))
    
    return covering_points

def copy_file(map_parser, source_path, destination_dir, new_name):
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

    # 7) Copy the file
    shutil.copy2(source_path, destination_path)

    print(f"File copied to {destination_path}")
    map_parser.Final_map_destination = destination_path


def find_zero_indices(arr):
    # Find the indices of all zeros in the array
    zero_indices = np.argwhere(arr == 0)

    if zero_indices.size == 0:
        # If there are no zeros in the array, return None or some other indication
        return None

    # Extract the first (x) and second (y) dimensions from the zero indices
    first_indices = zero_indices[:, 0]
    second_indices = zero_indices[:, 1]

    # Find the lowest and highest indices for the first dimension
    x = np.min(first_indices)
    X = np.max(first_indices)

    # Find the lowest and highest indices for the second dimension
    y = np.min(second_indices)
    Y = np.max(second_indices)

    return (x, X, y, Y)

def sgn(x):
    return (x > 0) - (x < 0)

def gilbert2d(x, y, ax, ay, bx, by, linked_list):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids.
    """
    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay))  # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by))  # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(w):
            linked_list.append((x, y))
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(h):
            linked_list.append((x, y))
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax // 2, ay // 2)
    (bx2, by2) = (bx // 2, by // 2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        gilbert2d(x, y, ax2, ay2, bx, by, linked_list)
        gilbert2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, linked_list)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        gilbert2d(x, y, bx2, by2, ax2, ay2, linked_list)
        gilbert2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, linked_list)
        gilbert2d(x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby),
                  -bx2, -by2, -(ax - ax2), -(ay - ay2), linked_list)

def plot_linked_list(linked_list, scale, width, height, animate):
    screen = turtle.Screen()
    screen.setup(width=width * scale, height=height * scale)
    drawing_turtle = turtle.Turtle()
    drawing_turtle.speed(0)  # Set the speed to the maximum
    drawing_turtle.penup()
    drawing_turtle.hideturtle()

    if not animate:
        screen.tracer(0, 0)  # Turn off automatic screen updates

    # Move turtle to the bottom left corner
    drawing_turtle.goto(-width * scale / 2, -height * scale / 2)
    drawing_turtle.pendown()

    current_node = linked_list.head
    if current_node:
        drawing_turtle.goto(current_node.coordinate[0] * scale - width * scale / 2,
                            current_node.coordinate[1] * scale - height * scale / 2)
        drawing_turtle.pendown()
        current_node = current_node.next

    while current_node:
        drawing_turtle.goto(current_node.coordinate[0] * scale - width * scale / 2,
                            current_node.coordinate[1] * scale - height * scale / 2)
        current_node = current_node.next

    screen.update()  # Update the screen manually
    turtle.done()

def generate_hilbert_curve_linked_list(width, height, scale):

    path = LinkedList()

    # generate the coordinates for the hilbert curve
    if width >= height:
        gilbert2d(0, 0, width, 0, 0, height, path)
    else:
        gilbert2d(0, 0, 0, height, width, 0, path)

    # multiply coordinates by scale factor
    node = path.head
    while node:
        node.coordinate = (node.coordinate[0]*scale, node.coordinate[1]*scale)
        node = node.next

    return path

def get_search_region(occ_grid, x, y, scale_factor):
    # get dims of occupancy grid
    occ_grid_width = occ_grid.shape[0]
    occ_grid_height = occ_grid.shape[1]
    scale_factor_2 = max(1, scale_factor // 2)

    # get search region bounds
    xmin, xmax = int(max(0, floor(x)-scale_factor_2)), int(min(occ_grid_width, floor(x)+scale_factor_2+1))
    ymin, ymax = int(max(0, floor(y)-scale_factor_2)), int(min(occ_grid_height, floor(y)+scale_factor_2+1))
    search_area = occ_grid[xmin:xmax, ymin:ymax]

    # correctly pad the search region so the waypoint is in the center
    x_left_pad = int(-(floor(x)-scale_factor_2)) if xmin == 0 else 0
    x_right_pad = int(floor(x) + scale_factor_2 + 1 - occ_grid_width) if xmax == occ_grid_width else 0
    y_left_pad = int(-(floor(y)-scale_factor_2)) if ymin == 0 else 0
    y_right_pad = int(floor(y) + scale_factor_2 + 1 - occ_grid_height) if ymax == occ_grid_height else 0
    search_area = np.pad(search_area, [(x_left_pad, x_right_pad), (y_left_pad, y_right_pad)], 'constant', constant_values=(-1, -1))

    return search_area

# search matrix spirally from the center for the first zero value
def spiral_search_matrix_zero(A):

    # A must be an odd-sized square matrix
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] % 2 == 1
    # sz = max(A.shape[0], A.shape[1])
    # if sz % 2 == 0:
    #     sz += 1

    # xpad = sz-A.shape[0]
    # ypad = sz-A.shape[1]
    # A = np.pad(A, [(floor(xpad/2), ceil(xpad/2)), (floor(ypad/2), ceil(ypad/2))], 'constant', constant_values=(-1, -1))

    dir = (0,1) # (0,1)-UP, (1,0)-RIGHT, (0,-1)-DOWN, (-1,0)-LEFT
    i = A.shape[0] // 2
    j = A.shape[1] // 2
    center = (i, j)
    n = 1
    ctr = 0

    # perform spiral pattern search
    while(is_valid_index(A, (i,j))):
        
        # search n cells in a particular direction
        for k in range(n):
            if A[i, j] == 0:
                return((i-center[0], j-center[1]))
            i += dir[0]
            j += dir[1]
        
        # switch direction
        dir = (dir[1], -dir[0])
        
        # increment n every 2 iterations of the outer loop
        if ctr % 2 == 0:
            n += 1
        ctr += 1
        
    return None
        

# check if an index in matrix A is valid
def is_valid_index(A, idx):
    if min(idx) < 0:
        return False
    for i in range(len(idx)):
        if idx[i] >= A.shape[i]:
            return False
    return True

def plot_binary_grid(grid, label):
    grid = np.transpose(grid)

    plt.figure(figsize= (10,10))
    # Plot the binary grid
    plt.imshow(grid, cmap='binary_r', interpolation='nearest')
    plt.title(label)
    plt.xlabel('Cell Number (X))')
    plt.ylabel('Cell Number (Y)')
    plt.gca().invert_yaxis()
    plt.show()

def plot_padding(grid_a, grid_b):
    grid_a = np.transpose(grid_a)
    grid_b = np.transpose(grid_b)
    
    plt.subplot(1,2,1)
    plt.imshow(grid_a, cmap='binary_r', interpolation='nearest')
    plt.title("Before Padding")
    plt.xlabel('X Dimension')
    plt.ylabel('Y Dimension')
    plt.gca().invert_yaxis()
    plt.subplot(1,2,2)
    plt.imshow(grid_b, cmap='binary_r', interpolation='nearest')
    plt.title("After Padding")
    plt.xlabel('X Dimension')
    plt.ylabel('Y Dimension')    
    plt.gca().invert_yaxis()
    plt.show()

def save_binary_grid(grid, filename):
    np.savetxt(filename + 'csv', grid, delimiter=',', fmt='%d')
    plt.savefig(filename + '.pdf')
    plt.savefig(filename + '.svg')
    plt.savefig(filename + '.png')

def is_valid_grid_space(grid, x, y):
    
    rows, cols = grid.shape

    return 0 <= x < rows and 0 <= y < cols

def is_traversible_grid_space(grid, x, y):

    return grid[x, y] == 0


def add_goal_point(coordinate_pos, origin_mm, file_lines, goal_count, grid_resolution, lines_start_line):
    index_x = coordinate_pos[0]
    index_y = coordinate_pos[1]
    
    # Convert indices to coordinates in millimeters
    coordinate_dx = index_x * grid_resolution
    coordinate_dy = index_y * grid_resolution

    # Calculate the coordinates in millimeters
    x_mm = origin_mm[0] + coordinate_dx + 0.5 * grid_resolution
    y_mm = origin_mm[1] + coordinate_dy + 0.5 * grid_resolution

    x_mm = int(x_mm)
    y_mm = int(y_mm)

    formatted_string = f'Cairn: Goal {x_mm} {y_mm} 0.000000 "" ICON "{goal_count}"'

    file_lines.insert(lines_start_line, formatted_string + '\n')

    return (x_mm, y_mm)
 
# Helper function to prompt the user to select a map file from the available ones
def prompt_select_map(maps_list):
    selected_map = ''

    # Get number of available map files
    num_maps = len(maps_list)
    
    if num_maps == 0:
        print("No maps are available in the specified directory.\nPlease update path or add a map file to the original maps directory.")
        exit()
        
    # Select the available map if only 
    if num_maps == 1:
        print(f"Map file {maps_list[0]} is the only file available and has been selected accordingly.")
        selected_map = maps_list[0]

    # Prompt the user to select the desired map
    else:
        print("\nPlease select a map file from the following by entering their key:")
        for i, map_file in enumerate(maps_list):
            print(f"{i}:\t\t{map_file}")    
        
        selection = -1
        while selection not in range(num_maps):
            print(f"Please enter a number between 0 and {num_maps-1}")
            user_input = input()
            if user_input.isnumeric():
                selection = int(user_input)
                
                if selection not in range(num_maps):
                    print("Selected number is out of range.\n")
                else:
                    selected_map = maps_list[selection]
            else:
                print("Error: Input is not a number!\n")
    
    print(f"Selected map file: {selected_map}")
    
    return selected_map

# Helper function to get the start line index of each data segment (i.e., points, lines, and forbidden region)
def get_seg_start(map_file_lines, first_forbidden_area, forbidden_area_line_list):
    # NOTE: Initialized map file's line index variables here and converted to regular variables
    # Initialize map file's data line offsets
    data_start_line = lines_start_line = forbidden_start_line = -1
    goals_start_line = -1
    goals_end_line = -1

    for i, line in enumerate(map_file_lines):
        if 'FeedbackHandler' in line:
            print("line",line)
        # check if string present on a current_node line
        #print(row.find(word))
        # find() method returns -1 if the value is not found,
        # if found it returns index of the first occurrence of the substring
        if line.find(DATA_WORD) != -1:
            print('DATA section exists in file')
            data_start_line = i
            print('DATA section start line number:', data_start_line)

        if line.find(LINES_WORD) != -1:
            print('LINES section exists in file')
            lines_start_line = i
            print('LINES section start line number:', lines_start_line)

        if line.find(FORBIDDEN_WORD) != -1:
            if first_forbidden_area:
                first_forbidden_area = False
                forbidden_start_line = i

            forbidden_area_line_list.append(i)
            print(f"forbidden area found: start line number {i}")
        
        if line.find(GOAL_WORD) != -1:
            # Set goals start line if not yet found
            if goals_start_line == -1:
                goals_start_line = i

            goals_end_line = i
            
    return data_start_line, lines_start_line, forbidden_start_line, goals_start_line, goals_end_line

# Helper function to calculate the resolution of the occupancy grid based on the maximum and mininimum x and y coordinates (works for zero-cell referencing only)
def calculate_grid_resolution(min_x, min_y, max_x, max_y, grid_resolution):
    # Calculate Occupancy Grid Dimensions
    grid_width = ceil((max_x - min_x)/grid_resolution)
    grid_height = ceil((max_y - min_y)/grid_resolution)

    # Check if we need to add an extra cell for inclusive bounds when max points are at a multiple of the grid's cell resolution
    if (max_x - min_x) % grid_resolution == 0:
        grid_width += 1

    if (max_y - min_y) % grid_resolution == 0:
        grid_height += 1

    return grid_width, grid_height

# Helper function to calculate the resolution of the occupancy grid based on the maximum and mininimum x and y coordinates
def calculate_grid_resolutionv1(min_x, min_y, max_x, max_y, grid_resolution):
    # Calculate Occupancy Grid Dimensions # NOTE: added 0.5*grid_resolution when using occupancy grid cell-center referencing
    grid_width = ceil((max_x - min_x + 0.5 * grid_resolution)/grid_resolution)
    grid_height = ceil((max_y - min_y + 0.5 * grid_resolution)/grid_resolution)

    # Check if we need to add an extra cell for inclusive bounds when max points are at a multiple of the grid's cell resolution
    if (max_x - min_x + 0.5 * grid_resolution) % grid_resolution == 0:
        grid_width += 1

    if (max_y - min_y + 0.5 * grid_resolution) % grid_resolution == 0:
        grid_height += 1

    return grid_width, grid_height    

# Function to generate an occupancy grid from a list of 2D points - References the start of each grid cell
def gen_occupancy_grid(points, grid_width, grid_height, origin, grid_resolution):
    
    # Create occupancy grid matrix
    occupancy_grid = np.zeros((grid_width, grid_height), dtype=np.int8)

    for point in points:
        # Remove offset from each point and find the index of the corresponding occupancy grid cell
        point_index_x = floor((point[0] - origin[0])/grid_resolution)
        point_index_y = floor((point[1] - origin[1])/grid_resolution)

        # Verify that the indices are within bounds of the grid dimensions
        assert 0 <= point_index_x < grid_width and 0 <= point_index_y < grid_height

        # Update the corresponding occupancy grid cell
        occupancy_grid[point_index_x, point_index_y] = -1
        
    return occupancy_grid

# Function to dialate/pad the obstacles within the map at high occupancy grid resolution
def pad_occupancy_grid(points, min_x, min_y, max_x, max_y, padd_grid_resolution):
    grid_width, grid_height = calculate_grid_resolution(min_x, min_y, max_x, max_y, padd_grid_resolution)

    occupancy_grid = gen_occupancy_grid(points, grid_width, grid_height, (min_x, min_y), padd_grid_resolution)

    padded_occupancy_grid = dilate_grid(occupancy_grid)

    return occupancy_grid, padded_occupancy_grid

def dilate_grid_v2(occ_grid, occ_grid_res, padding_amount):

    grid_size = occ_grid.shape
    padded_occ_grid = np.zeros(grid_size, dtype=np.int8)

    # Number of grid cells to pad by
    assert occ_grid_res > 0
    n = ceil(padding_amount / occ_grid_res)

    # Square-pad each occupied cell by n cells
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if occ_grid[i][j] == -1:

                i_min = max(0, i-n)
                i_max = min(i+n+1, grid_size[0])
                j_min = max(0, j-n)
                j_max = min(j+n+1, grid_size[1])

                padded_occ_grid[i_min:i_max, j_min:j_max] = -1

    return padded_occ_grid

def dilate_grid(occupancy_grid):
    grid_size = occupancy_grid.shape
    padded_occupancy_grid = np.zeros(grid_size, dtype=np.int8)

    # Parse occupancy grid and padd each instance (3x3 adjacent blocks)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            if occupancy_grid[i][j] == -1:
                # Check if index is 0 in either dimension (function works if either index is max)
                if i == 0:
                    if j == 0:
                        padded_occupancy_grid[i:i+2, j:j+2] = -1
                    else:
                        padded_occupancy_grid[i:i+2, j-1:j+2] = -1
                elif j == 0:
                    padded_occupancy_grid[i-1:i+2, j:j+2] = -1
                else:
                    padded_occupancy_grid[i-1:i+2, j-1:j+2] = -1

    return padded_occupancy_grid

def downsample_occupancy_grid(high_res_grid, high_res_resolution, low_res_resolution):
    """
    Converts a high-resolution occupancy grid to a lower resolution one.
    
    Parameters:
        - high_res_grid: 2D numpy array with the high-resolution occupancy grid
        - high_res_resolution: Integer resolution of the high-resolution grid in mm
        - low_res_resolution: Integer resolution of the low-resolution grid in mm
    
    Returns:
        - low_res_grid: 2D numpy array with the lower-resolution occupancy grid
    """
    # Calculate the downsampling factor
    downsample_factor = low_res_resolution / high_res_resolution

    # Get high-resolution's grid dimensions
    high_res_height, high_res_width = high_res_grid.shape
    
    # Calculate the low-resolution occupancy grid size
    low_res_height = ceil(high_res_height / downsample_factor)
    low_res_width = ceil(high_res_width / downsample_factor)

    # Initialize the lower resolution occupancy grid
    low_res_grid = np.zeros((low_res_height, low_res_width), dtype=np.int8)    
    
    # Downscale the high-resolution grid to low-resolution
    for i in range(low_res_height):
        for j in range(low_res_width):
            # Extract the corresponding high resolution cells
            high_res_subgrid = high_res_grid[
                round(i*downsample_factor):round((i+1)*downsample_factor),
                round(j*downsample_factor):round((j+1)*downsample_factor)
            ]
            
            # Determine the occupancy state of the lower resolution cell
            if np.any(high_res_subgrid == -1):  # If any cell is occupied
                low_res_grid[i, j] = -1
            else:
                low_res_grid[i, j] = 0
    
    return low_res_grid

def get_output_filename(input_map_file, grid_res, padding, padding_grid_res, path_plan_algorithm):
    output_map_file = input_map_file.split('.')[0] + '_' + str(grid_res)
    output_map_file += "_padding_" + str(padding_grid_res) + "_" + path_plan_algorithm + '.map' if padding else "_" + path_plan_algorithm + '.map'

    return output_map_file

# Implementation: Call by reference
def filter_map_boundaries(occupancy_grid):
    # Find shortest dimension (width vs height)
    height, width = occupancy_grid.shape

    # vertical map (filter horizontally)
    if height >= width:
        for i in range(height):
            row = occupancy_grid[i, :]
            
            # Check if no points exist in given row
            if row.sum() == 0:
                occupancy_grid[i, :] = -1

            # Filter map boundaries based on position of the obstacles
            else:
                # if row.sum() < 0:
                min_point_index, max_point_index = get_vector_boundaries(row)

                # Fill occupancy grid's row beyond these points
                occupancy_grid[i, 0:min_point_index] = -1
                occupancy_grid[i, max_point_index+1:] = -1
    
    # horizontal map (filter vertically)
    else:
        for j in range(width):
            col = occupancy_grid[:, j]
                
            # Check if no points exist in given column
            if col.sum() == 0:
                occupancy_grid[:, j] = -1
            
            # Filter map boundaries based on position of the obstacles
            else:
                # if col.sum() < 0:
                min_point_index, max_point_index = get_vector_boundaries(col)

                # Fill occupancy grid's column beyond these points
                occupancy_grid[0:min_point_index, j] = -1
                occupancy_grid[max_point_index+1:, j] = -1 

def get_vector_boundaries(vector):
    min_point_index = 0
    max_point_index = len(vector)

    # Update point indices
    # Find min point
    for i, value in enumerate(vector):
        if value == -1:
            min_point_index = i
            break

    # Find max point
    for i, value in reversed(list(enumerate(vector))):
        if value == -1:
            max_point_index = i
            break
    
    return min_point_index, max_point_index

# Function to filter out the background and any untraversable space (instead of flood filling)
def filter_untraversable_space(grid, visualize = False):
    # Label connected free spaces (0 = free space, 1 = obstacle)
    # structure = np.ones((3, 3), dtype=int)  # 8-connectivity for region growing
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]]) # 4-connectivity (cross-shaped)

    labeled_grid, _ = ndimage.label(grid == 0, structure=structure)

    if visualize:
        visualize_cells(labeled_grid)
    
    # determine which is the background label using the border of the map
    top_row = labeled_grid[0, :]
    bottom_row = labeled_grid[-1, :]
    top_col  = labeled_grid[:, 0]
    bottom_col = labeled_grid[:, -1]
    map_border = np.concatenate([top_row, bottom_row, top_col, bottom_col])
    
    # Find the background's label index
    background_label = stats.mode(map_border)[0]
    # print("BG LABEL:")
    # print(background_label)

    # Find the largest region not including the background
    mask_grid = labeled_grid.copy()

    # mask_grid[mask_grid == 0] = 0

    max_label_idx = -1
    max_value = 0
    num_labels = len(np.unique(mask_grid))

    for label_idx in range(num_labels):
        if label_idx == background_label:
            continue

        num_cells = (mask_grid == label_idx).sum()

        if num_cells > max_value:
            max_value = num_cells
            max_label_idx = label_idx
        
    # filter the occupancy grid to only keep the traversable map
    grid[mask_grid != max_label_idx] = -1
    grid[mask_grid == max_label_idx] = 0


def visualize_cells(grid, cells=None):
    grid = np.transpose(grid)
    # color_map = colors.Colormap("test", num_regions)

    plt.figure(figsize= (2,8))
    # Plot the binary grid
    plt.imshow(grid, cmap = 'plasma', interpolation='nearest')
    plt.title("Cell Decomposition")
    plt.xlabel('X Dimension')
    plt.ylabel('Y Dimension')
    plt.gca().invert_yaxis()
    plt.show()

def read_file(map_file_path):
    with open(map_file_path, 'r') as file:
        return file.readlines()

# Function to add complete coverage path coordinate list to map file text lines
def add_ccp_to_map(ccp_list, grid_resolution, origin_mm, file_lines, lines_start_line):
    goal_count = 0
    current_node = ccp_list.head
    while current_node:
        add_goal_point(current_node.coordinate, origin_mm, file_lines, goal_count, grid_resolution, lines_start_line)
        # Update goal count
        goal_count += 1
        if current_node.next is None:
            print("None, end of list")
        current_node = current_node.next

# Function to add a dock search goal point to map file's text
def add_dock_goal(dock_x, dock_y, dock_heading, file_lines, lines_start_line):
    formatted_string = f'Cairn: DockLynx {dock_x} {dock_y} {dock_heading} "Autonomous docking goal point position to enable autonomous charging of WiseBot." ICON "{DOCK_GOAL_WORD}"'
    # print(formatted_string)
    file_lines.insert(lines_start_line, formatted_string + '\n')


def visualize_path(linked_list, scale, grid_dims):
    grid_width, grid_height = grid_dims
    plot_linked_list(linked_list, scale, grid_width, grid_height, animate = True)        

# Function to generate map points from forbidden areas based on the occupancy grid resolution
def gen_forbidden_area_pts(forbidden_areas, occ_grid_res):
    # Initialize list for generate map points
    map_points = []
    # Parse forbidden areas and generate map points based on the specified resolution
    for forbidden_area in forbidden_areas:
        point_list = rotate_and_interpolate_forbidden_area_element(forbidden_area, occ_grid_res)
        for point in point_list:
            map_points.append(point)
    return map_points



# def export_map_file(output_map_file_path, map_file_lines):
#     """Export map data to a file, ensuring the directory exists first."""

#     output_map_file_path = os.path.abspath(output_map_file_path)

#     output_dir = os.path.dirname(output_map_file_path)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)  

#     try:
#         with open(output_map_file_path, 'w+', encoding="utf-8") as file:
#             file.writelines(map_file_lines)
#         print(f" Map file successfully exported: {output_map_file_path}")

#     except OSError as e:
#         print(f" Error writing file: {e}")


def export_map_file(output_map_file_path, map_file_lines):
    with open(output_map_file_path, 'w+') as file:
        file.writelines(map_file_lines)