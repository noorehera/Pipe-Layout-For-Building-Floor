import numpy as np
import cv2
from PIL import Image
import io

def process_image(image_data, grid_size=(30, 30)):
    """
    Process an uploaded image into a grid layout.
    
    Args:
        image_data: Image data (bytes or file-like object)
        grid_size (tuple): Size of the grid (height, width)
        
    Returns:
        dict: Layout information including grid, inlet, and outlet
    """
    # Convert image data to numpy array
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        image = Image.open(image_data)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to match grid size
    image = image.resize((grid_size[1] * 10, grid_size[0] * 10))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Threshold to binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Resize to grid size
    grid = np.zeros(grid_size, dtype=np.int8)
    
    # Convert binary image to grid (0 = free space, 1 = obstacle)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Get the corresponding region in the binary image
            region = binary[i*10:(i+1)*10, j*10:(j+1)*10]
            # If more than 50% of pixels are black, mark as obstacle
            if np.mean(region) < 127:
                grid[i, j] = 1
    
    # Add walls (border obstacles)
    grid[0, :] = 1  # Top wall
    grid[-1, :] = 1  # Bottom wall
    grid[:, 0] = 1  # Left wall
    grid[:, -1] = 1  # Right wall
    
    # Look for green (inlet) and red (outlet) pixels in the original image
    inlet = None
    outlet = None
    
    # Define color ranges in RGB
    green_lower = np.array([0, 100, 0])
    green_upper = np.array([100, 255, 100])
    red_lower = np.array([100, 0, 0])
    red_upper = np.array([255, 100, 100])
    
    # Find inlet (green)
    green_mask = cv2.inRange(img_array, green_lower, green_upper)
    green_pixels = np.where(green_mask > 0)
    if len(green_pixels[0]) > 0:
        # Calculate average position of green pixels
        y = int(np.mean(green_pixels[0]) / 10)
        x = int(np.mean(green_pixels[1]) / 10)
        inlet = (min(max(y, 1), grid_size[0]-2), min(max(x, 1), grid_size[1]-2))
        grid[inlet[0], inlet[1]] = 0  # Ensure inlet is not an obstacle
    
    # Find outlet (red)
    red_mask = cv2.inRange(img_array, red_lower, red_upper)
    red_pixels = np.where(red_mask > 0)
    if len(red_pixels[0]) > 0:
        # Calculate average position of red pixels
        y = int(np.mean(red_pixels[0]) / 10)
        x = int(np.mean(red_pixels[1]) / 10)
        outlet = (min(max(y, 1), grid_size[0]-2), min(max(x, 1), grid_size[1]-2))
        grid[outlet[0], outlet[1]] = 0  # Ensure outlet is not an obstacle
    
    # If inlet or outlet not found, set default positions
    if inlet is None:
        inlet = (1, 1)
        grid[inlet[0], inlet[1]] = 0
    
    if outlet is None:
        outlet = (grid_size[0]-2, grid_size[1]-2)
        grid[outlet[0], outlet[1]] = 0
    
    # Clear a small area around inlet and outlet to ensure they're accessible
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny_inlet, nx_inlet = inlet[0] + dy, inlet[1] + dx
            ny_outlet, nx_outlet = outlet[0] + dy, outlet[1] + dx
            
            # Check bounds for inlet
            if 0 < ny_inlet < grid_size[0]-1 and 0 < nx_inlet < grid_size[1]-1:
                grid[ny_inlet, nx_inlet] = 0
            
            # Check bounds for outlet
            if 0 < ny_outlet < grid_size[0]-1 and 0 < nx_outlet < grid_size[1]-1:
                grid[ny_outlet, nx_outlet] = 0
    
    return {
        "grid": grid.tolist(),
        "inlet": inlet,
        "outlet": outlet,
        "grid_size": grid_size
    }

def visualize_processed_image(layout):
    """
    Visualize the processed image layout.
    
    Args:
        layout (dict): Layout information including grid, inlet, and outlet
        
    Returns:
        numpy.ndarray: Visualization image
    """
    grid = np.array(layout["grid"])
    inlet = layout["inlet"]
    outlet = layout["outlet"]
    
    # Create RGB image (white background)
    height, width = grid.shape
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw obstacles (black)
    for i in range(height):
        for j in range(width):
            if grid[i, j] == 1:
                img[i, j] = [0, 0, 0]
    
    # Draw inlet (green)
    if inlet:
        img[inlet[0], inlet[1]] = [0, 255, 0]
    
    # Draw outlet (red)
    if outlet:
        img[outlet[0], outlet[1]] = [255, 0, 0]
    
    # Draw pipe path if available (blue)
    if "pipe_path" in layout and layout["pipe_path"]:
        for pos in layout["pipe_path"]:
            y, x = pos
            if (y, x) != inlet and (y, x) != outlet:
                img[y, x] = [0, 0, 255]
    
    # Resize for better visualization
    img_resized = cv2.resize(img, (width * 10, height * 10), interpolation=cv2.INTER_NEAREST)
    
    return img_resized

def generate_sample_image(grid_size=(30, 30)):
    """
    Generate a sample image for testing.
    
    Args:
        grid_size (tuple): Size of the grid (height, width)
        
    Returns:
        numpy.ndarray: Sample image
    """
    # Create a white background
    img = np.ones((grid_size[0] * 10, grid_size[1] * 10, 3), dtype=np.uint8) * 255
    
    # Add border (black)
    img[0:10, :] = [0, 0, 0]  # Top
    img[-10:, :] = [0, 0, 0]  # Bottom
    img[:, 0:10] = [0, 0, 0]  # Left
    img[:, -10:] = [0, 0, 0]  # Right
    
    # Add some obstacles (black rectangles)
    # Obstacle 1
    img[50:100, 50:150] = [0, 0, 0]
    # Obstacle 2
    img[150:200, 100:200] = [0, 0, 0]
    # Obstacle 3
    img[100:250, 220:240] = [0, 0, 0]
    
    # Add inlet (green)
    img[20:30, 20:30] = [0, 255, 0]
    
    # Add outlet (red)
    img[grid_size[0]*10-30:grid_size[0]*10-20, grid_size[1]*10-30:grid_size[1]*10-20] = [255, 0, 0]
    
    return img 