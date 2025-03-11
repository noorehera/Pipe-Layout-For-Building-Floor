import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx

class PipeRouter:
    """
    A* pathfinding algorithm for optimizing pipe layouts.
    """
    
    def __init__(self, bend_penalty=10, spacing_factor=2):
        """
        Initialize the pipe router.
        
        Args:
            bend_penalty (float): Penalty for each bend in the pipe
            spacing_factor (int): Minimum spacing from obstacles
        """
        self.bend_penalty = bend_penalty
        self.spacing_factor = spacing_factor
        self.directions = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1)    # right
        ]
    
    def _heuristic(self, a, b):
        """
        Manhattan distance heuristic for A* algorithm.
        
        Args:
            a (tuple): First point (y, x)
            b (tuple): Second point (y, x)
            
        Returns:
            float: Manhattan distance between points
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _is_valid_position(self, pos, grid, visited, start=None, goal=None):
        """
        Check if a position is valid (within grid bounds, not an obstacle, not visited).
        
        Args:
            pos (tuple): Position to check (y, x)
            grid (numpy.ndarray): Grid representation of the room
            visited (set): Set of visited positions
            start (tuple, optional): Start position (inlet)
            goal (tuple, optional): Goal position (outlet)
            
        Returns:
            bool: True if position is valid, False otherwise
        """
        y, x = pos
        height, width = grid.shape
        
        # Check if position is within grid bounds
        if y < 0 or y >= height or x < 0 or x >= width:
            return False
        
        # Allow start and goal positions even if they are obstacles
        if start is not None and pos == start:
            return True
        if goal is not None and pos == goal:
            return True
        
        # Check if position is an obstacle
        if grid[y, x] == 1:
            return False
        
        # Check if position has been visited
        if pos in visited:
            return False
        
        return True
    
    def _check_spacing(self, pos, grid, start=None, goal=None):
        """
        Check if a position maintains minimum spacing from obstacles.
        
        Args:
            pos (tuple): Position to check (y, x)
            grid (numpy.ndarray): Grid representation of the room
            start (tuple, optional): Start position (inlet)
            goal (tuple, optional): Goal position (outlet)
            
        Returns:
            bool: True if spacing is maintained, False otherwise
        """
        # Always allow start and goal positions
        if start is not None and pos == start:
            return True
        if goal is not None and pos == goal:
            return True
            
        if self.spacing_factor <= 1:
            return True
            
        y, x = pos
        height, width = grid.shape
        
        # Check surrounding cells based on spacing factor
        for dy in range(-self.spacing_factor + 1, self.spacing_factor):
            for dx in range(-self.spacing_factor + 1, self.spacing_factor):
                ny, nx = y + dy, x + dx
                
                # Skip the position itself
                if dy == 0 and dx == 0:
                    continue
                
                # Skip positions outside grid bounds
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                
                # Skip start and goal positions
                if start is not None and (ny, nx) == start:
                    continue
                if goal is not None and (ny, nx) == goal:
                    continue
                
                # If an obstacle is too close, return False
                if grid[ny, nx] == 1:
                    return False
        
        return True
    
    def _get_direction(self, a, b):
        """
        Get the direction from point a to point b.
        
        Args:
            a (tuple): First point (y, x)
            b (tuple): Second point (y, x)
            
        Returns:
            tuple: Direction vector (dy, dx)
        """
        dy = b[0] - a[0]
        dx = b[1] - a[1]
        
        if dy != 0:
            dy = dy // abs(dy)
        if dx != 0:
            dx = dx // abs(dx)
            
        return (dy, dx)
    
    def find_path(self, layout):
        """
        Find the optimal pipe path using A* algorithm.
        
        Args:
            layout (dict): Layout information including grid, inlet, and outlet
            
        Returns:
            list: List of positions representing the optimal pipe path
        """
        grid = np.array(layout["grid"])
        start = layout["inlet"]
        goal = layout["outlet"]
        
        # Ensure inlet and outlet are accessible
        grid_copy = grid.copy()
        if start[0] >= 0 and start[0] < grid.shape[0] and start[1] >= 0 and start[1] < grid.shape[1]:
            grid_copy[start[0], start[1]] = 0
        if goal[0] >= 0 and goal[0] < grid.shape[0] and goal[1] >= 0 and goal[1] < grid.shape[1]:
            grid_copy[goal[0], goal[1]] = 0
        
        # Initialize data structures
        open_set = []
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        # Push start node to open set
        heapq.heappush(open_set, (f_score[start], start))
        
        # Initialize direction for bend detection
        current_direction = None
        
        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)
            
            # If goal is reached, reconstruct path
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Add current node to closed set
            closed_set.add(current)
            
            # Explore neighbors
            for direction in self.directions:
                dy, dx = direction
                neighbor = (current[0] + dy, current[1] + dx)
                
                # Skip invalid neighbors
                if not self._is_valid_position(neighbor, grid_copy, closed_set, start, goal):
                    continue
                
                # Skip neighbors that don't maintain spacing
                if not self._check_spacing(neighbor, grid_copy, start, goal):
                    continue
                
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + 1
                
                # Add bend penalty if direction changes
                if current in came_from:
                    prev = came_from[current]
                    prev_direction = self._get_direction(prev, current)
                    new_direction = self._get_direction(current, neighbor)
                    
                    if prev_direction != new_direction:
                        tentative_g_score += self.bend_penalty
                
                # If neighbor is not in g_score or has a better score
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    
                    # Add neighbor to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        return None
    
    def optimize_layout(self, layout):
        """
        Optimize the pipe layout for a given room layout.
        
        Args:
            layout (dict): Layout information including grid, inlet, and outlet
            
        Returns:
            dict: Updated layout with optimized pipe path
        """
        # Try with current settings
        path = self.find_path(layout)
        
        # If no path found, try with reduced spacing factor
        if path is None and self.spacing_factor > 1:
            print(f"No path found with spacing factor {self.spacing_factor}, trying with reduced spacing...")
            original_spacing = self.spacing_factor
            self.spacing_factor = 1
            path = self.find_path(layout)
            self.spacing_factor = original_spacing
        
        # If still no path, try with direct line (ignoring obstacles)
        if path is None:
            print("No path found with standard A* search, trying direct path...")
            start = layout["inlet"]
            goal = layout["outlet"]
            
            # Create a simple direct path
            path = self._create_direct_path(start, goal)
        
        if path:
            layout["pipe_path"] = path
            return layout
        else:
            print("No valid path found!")
            # Return layout with empty pipe path
            layout["pipe_path"] = []
            return layout
    
    def _create_direct_path(self, start, goal):
        """
        Create a direct path between start and goal, ignoring obstacles.
        This is a fallback method when A* cannot find a path.
        
        Args:
            start (tuple): Start position (y, x)
            goal (tuple): Goal position (y, x)
            
        Returns:
            list: List of positions representing a direct path
        """
        path = [start]
        current = start
        
        while current != goal:
            # Get direction to goal
            dy = goal[0] - current[0]
            dx = goal[1] - current[1]
            
            # Normalize direction
            if dy != 0:
                dy = dy // abs(dy)
            if dx != 0:
                dx = dx // abs(dx)
            
            # Move in x direction first, then y direction
            if dx != 0:
                next_pos = (current[0], current[1] + dx)
                current = next_pos
                path.append(current)
            elif dy != 0:
                next_pos = (current[0] + dy, current[1])
                current = next_pos
                path.append(current)
        
        return path
    
    def visualize_solution(self, layout, save_path=None, show=True):
        """
        Visualize the optimized pipe layout.
        
        Args:
            layout (dict): Layout information including grid, inlet, outlet, and pipe_path
            save_path (str, optional): Path to save the visualization
            show (bool): Whether to display the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        grid = np.array(layout["grid"])
        inlet = layout["inlet"]
        outlet = layout["outlet"]
        pipe_path = layout.get("pipe_path", [])
        
        # Create a copy of the grid for visualization
        vis_grid = grid.copy()
        
        # Mark pipe path on the grid
        for pos in pipe_path:
            y, x = pos
            vis_grid[y, x] = 2  # 2 represents pipe
        
        # Create custom colormap
        colors = ['white', 'black', 'blue']
        cmap = ListedColormap(colors)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot grid with pipe path
        ax.imshow(vis_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=2)
        
        # Plot inlet and outlet
        ax.plot(inlet[1], inlet[0], 'go', markersize=10, label='Inlet')
        ax.plot(outlet[1], outlet[0], 'ro', markersize=10, label='Outlet')
        
        # Count bends in the pipe path
        bends = 0
        for i in range(1, len(pipe_path) - 1):
            prev_dir = self._get_direction(pipe_path[i-1], pipe_path[i])
            next_dir = self._get_direction(pipe_path[i], pipe_path[i+1])
            if prev_dir != next_dir:
                bends += 1
        
        # Calculate pipe length
        pipe_length = len(pipe_path) - 1 if pipe_path else 0
        
        ax.set_title(f'Optimized Pipe Layout\nLength: {pipe_length}, Bends: {bends}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig


if __name__ == "__main__":
    # Example usage
    from utils.dataset_generator import LayoutGenerator
    
    # Generate a sample layout
    generator = LayoutGenerator(grid_size=(30, 30))
    layout = generator.generate_layout()
    
    # Optimize pipe layout
    router = PipeRouter(bend_penalty=10)
    optimized_layout = router.optimize_layout(layout)
    
    # Visualize the solution
    router.visualize_solution(optimized_layout) 