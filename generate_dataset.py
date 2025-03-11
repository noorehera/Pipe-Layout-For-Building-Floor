#!/usr/bin/env python3
"""
Script to generate a synthetic dataset of room layouts for pipe routing optimization.
"""

import argparse
from utils.dataset_generator import LayoutGenerator
from models.pipe_router import PipeRouter
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate a synthetic dataset for pipe layout optimization')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to generate')
    parser.add_argument('--grid_size', type=int, default=30, help='Size of the grid (N x N)')
    parser.add_argument('--min_obstacles', type=int, default=5, help='Minimum number of obstacles')
    parser.add_argument('--max_obstacles', type=int, default=15, help='Maximum number of obstacles')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--with_solutions', action='store_true', help='Generate solutions along with layouts')
    parser.add_argument('--bend_penalty', type=float, default=10, help='Penalty for each bend in the pipe')
    parser.add_argument('--spacing_factor', type=int, default=1, help='Minimum spacing from obstacles')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    layouts_dir = os.path.join(args.output_dir, 'layouts')
    solutions_dir = os.path.join(args.output_dir, 'solutions')
    
    Path(layouts_dir).mkdir(parents=True, exist_ok=True)
    if args.with_solutions:
        Path(solutions_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize layout generator
    generator = LayoutGenerator(
        grid_size=(args.grid_size, args.grid_size),
        min_obstacles=args.min_obstacles,
        max_obstacles=args.max_obstacles
    )
    
    # Initialize pipe router if solutions are requested
    if args.with_solutions:
        router = PipeRouter(
            bend_penalty=args.bend_penalty,
            spacing_factor=args.spacing_factor
        )
    
    print(f"Generating {args.num_samples} samples...")
    
    # Generate layouts and solutions
    for i in range(args.num_samples):
        # Generate layout
        layout = generator.generate_layout()
        
        # Save layout visualization
        generator.visualize_layout(
            layout,
            save_path=os.path.join(layouts_dir, f'layout_{i}.png'),
            show=False
        )
        
        # Generate and save solution if requested
        if args.with_solutions:
            # Optimize pipe layout
            optimized_layout = router.optimize_layout(layout)
            
            # Save solution visualization
            router.visualize_solution(
                optimized_layout,
                save_path=os.path.join(solutions_dir, f'solution_{i}.png'),
                show=False
            )
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{args.num_samples} samples")
    
    print(f"Dataset generation complete. Saved to {args.output_dir}")

if __name__ == "__main__":
    main() 