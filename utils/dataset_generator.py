import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
from pathlib import Path

class LayoutGenerator:
    """
    Generates synthetic room layouts with obstacles, inlets, and outlets
    for pipe layout optimization.
    """
    
    def __init__(self, grid_size=(50, 50), obstacle_density=0.1, min_obstacles=5, max_obstacles=15):
        """
        Initialize the layout generator.
        
        Args:
            grid_size (tuple): Size of the grid (height, width)
            obstacle_density (float): Density of obstacles (0-1)
            min_obstacles (int): Minimum number of obstacles
            max_obstacles (int): Maximum number of obstacles
        """
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        
    def generate_layout(self):
        """
        Generate a random room layout with obstacles, inlet, and outlet.
        
        Returns:
            dict: Layout information including grid, inlet, outlet, and obstacles
        """
        # Initialize empty grid (0 = free space, 1 = obstacle)
        grid = np.zeros(self.grid_size, dtype=np.int8)
        
        # Generate walls (border obstacles)
        grid[0, :] = 1  # Top wall
        grid[-1, :] = 1  # Bottom wall
        grid[:, 0] = 1  # Left wall
        grid[:, -1] = 1  # Right wall
        
        # Generate random obstacles (rectangular rooms, furniture, etc.)
        num_obstacles = random.randint(self.min_obstacles, self.max_obstacles)
        obstacles = []
        
        for _ in range(num_obstacles):
            # Random obstacle size
            height = random.randint(2, max(3, int(self.grid_size[0] * 0.2)))
            width = random.randint(2, max(3, int(self.grid_size[1] * 0.2)))
            
            # Random obstacle position (not on the border)
            top = random.randint(2, self.grid_size[0] - height - 2)
            left = random.randint(2, self.grid_size[1] - width - 2)
            
            # Add obstacle to grid
            grid[top:top+height, left:left+width] = 1
            obstacles.append({"top": top, "left": left, "height": height, "width": width})
        
        # Generate inlet and outlet positions
        # Instead of placing them on walls, place them near walls but in free space
        side = random.choice(["top", "bottom", "left", "right"])
        
        if side == "top":
            inlet_x = random.randint(1, self.grid_size[1] - 2)
            inlet_y = 1  # One cell below the top wall
            outlet_x = random.randint(1, self.grid_size[1] - 2)
            outlet_y = self.grid_size[0] - 2  # One cell above the bottom wall
        elif side == "bottom":
            inlet_x = random.randint(1, self.grid_size[1] - 2)
            inlet_y = self.grid_size[0] - 2  # One cell above the bottom wall
            outlet_x = random.randint(1, self.grid_size[1] - 2)
            outlet_y = 1  # One cell below the top wall
        elif side == "left":
            inlet_x = 1  # One cell to the right of the left wall
            inlet_y = random.randint(1, self.grid_size[0] - 2)
            outlet_x = self.grid_size[1] - 2  # One cell to the left of the right wall
            outlet_y = random.randint(1, self.grid_size[0] - 2)
        else:  # right
            inlet_x = self.grid_size[1] - 2  # One cell to the left of the right wall
            inlet_y = random.randint(1, self.grid_size[0] - 2)
            outlet_x = 1  # One cell to the right of the left wall
            outlet_y = random.randint(1, self.grid_size[0] - 2)
        
        # Ensure inlet and outlet are not obstacles
        grid[inlet_y, inlet_x] = 0
        grid[outlet_y, outlet_x] = 0
        
        # Clear a small area around inlet and outlet to ensure they're accessible
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny_inlet, nx_inlet = inlet_y + dy, inlet_x + dx
                ny_outlet, nx_outlet = outlet_y + dy, outlet_x + dx
                
                # Check bounds for inlet
                if 0 <= ny_inlet < self.grid_size[0] and 0 <= nx_inlet < self.grid_size[1]:
                    grid[ny_inlet, nx_inlet] = 0
                
                # Check bounds for outlet
                if 0 <= ny_outlet < self.grid_size[0] and 0 <= nx_outlet < self.grid_size[1]:
                    grid[ny_outlet, nx_outlet] = 0
        
        return {
            "grid": grid.tolist(),
            "inlet": (inlet_y, inlet_x),
            "outlet": (outlet_y, outlet_x),
            "obstacles": obstacles,
            "grid_size": self.grid_size
        }
    
    def visualize_layout(self, layout, save_path=None, show=True):
        """
        Visualize the generated layout.
        
        Args:
            layout (dict): Layout information
            save_path (str, optional): Path to save the visualization
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        grid = np.array(layout["grid"])
        inlet = layout["inlet"]
        outlet = layout["outlet"]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot grid
        ax.imshow(grid, cmap='binary', interpolation='nearest')
        
        # Plot inlet and outlet
        ax.plot(inlet[1], inlet[0], 'go', markersize=10, label='Inlet')
        ax.plot(outlet[1], outlet[0], 'ro', markersize=10, label='Outlet')
        
        ax.set_title('Room Layout')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def generate_dataset(self, num_samples, output_dir='data'):
        """
        Generate a dataset of room layouts.
        
        Args:
            num_samples (int): Number of samples to generate
            output_dir (str): Directory to save the dataset
            
        Returns:
            list: List of generated layouts
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        layouts = []
        
        for i in range(num_samples):
            layout = self.generate_layout()
            layouts.append(layout)
            
            # Save layout as JSON
            with open(os.path.join(output_dir, f'layout_{i}.json'), 'w') as f:
                json.dump(layout, f)
            
            # Save visualization
            self.visualize_layout(
                layout, 
                save_path=os.path.join(output_dir, f'layout_{i}.png'),
                show=False
            )
        
        return layouts


if __name__ == "__main__":
    # Example usage
    generator = LayoutGenerator(grid_size=(30, 30), obstacle_density=0.15)
    layout = generator.generate_layout()
    generator.visualize_layout(layout)
    
    # Generate a small dataset
    # generator.generate_dataset(5, output_dir='../data') 