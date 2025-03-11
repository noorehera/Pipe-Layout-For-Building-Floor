#!/usr/bin/env python3
"""
Streamlit web application for pipe layout optimization.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import tempfile
from PIL import Image
import io
import cv2
from utils.dataset_generator import LayoutGenerator
from models.pipe_router import PipeRouter
from utils.image_processor import process_image, visualize_processed_image, generate_sample_image

# Set page configuration
st.set_page_config(
    page_title="Pipe Layout Optimizer",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTitle {
        font-weight: bold;
        color: #2C3E50;
    }
    .stHeader {
        font-weight: bold;
        color: #2C3E50;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔧 Pipe Layout Optimizer")
st.markdown("Generate optimized pipe layouts for building floors based on room layouts and inlet/outlet locations.")

# Sidebar
st.sidebar.header("Configuration")

# Grid size
grid_size = st.sidebar.slider("Grid Size", min_value=10, max_value=50, value=30, step=5)

# Optimization parameters
st.sidebar.subheader("Optimization Parameters")
bend_penalty = st.sidebar.slider("Bend Penalty", min_value=1, max_value=20, value=10, step=1)
spacing_factor = st.sidebar.slider("Spacing Factor", min_value=1, max_value=3, value=1, step=1)

# Layout generation parameters
st.sidebar.subheader("Layout Generation")
min_obstacles = st.sidebar.slider("Min Obstacles", min_value=1, max_value=10, value=5, step=1)
max_obstacles = st.sidebar.slider("Max Obstacles", min_value=5, max_value=20, value=10, step=1)

# Input method selection
input_method = st.sidebar.radio(
    "Input Method",
    ["Generate Random Layout", "Draw Layout", "Upload Layout", "Upload Image"]
)

# Reset button
st.sidebar.markdown("---")
if st.sidebar.button("Reset Application"):
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Main content
if input_method == "Generate Random Layout":
    st.header("Generate Random Layout")
    
    if st.button("Generate New Layout"):
        # Initialize layout generator
        generator = LayoutGenerator(
            grid_size=(grid_size, grid_size),
            min_obstacles=min_obstacles,
            max_obstacles=max_obstacles
        )
        
        # Generate layout
        layout = generator.generate_layout()
        
        # Store layout in session state
        st.session_state.layout = layout
        
        # Visualize layout
        fig = generator.visualize_layout(layout, show=False)
        st.pyplot(fig)
        plt.close(fig)
    
    # Optimize button (outside the if statement for Generate New Layout)
    if "layout" in st.session_state and st.button("Optimize Pipe Layout"):
        # Initialize pipe router
        router = PipeRouter(
            bend_penalty=bend_penalty,
            spacing_factor=spacing_factor
        )
        
        # Optimize layout
        optimized_layout = router.optimize_layout(st.session_state.layout)
        
        # Store optimized layout in session state
        st.session_state.optimized_layout = optimized_layout
        
        # Visualize solution
        fig = router.visualize_solution(optimized_layout, show=False)
        st.pyplot(fig)
        plt.close(fig)
        
        # Display statistics
        pipe_path = optimized_layout.get("pipe_path", [])
        pipe_length = len(pipe_path) - 1 if pipe_path else 0
        
        # Count bends
        bends = 0
        for i in range(1, len(pipe_path) - 1):
            prev_dir = router._get_direction(pipe_path[i-1], pipe_path[i])
            next_dir = router._get_direction(pipe_path[i], pipe_path[i+1])
            if prev_dir != next_dir:
                bends += 1
        
        st.info(f"Pipe Length: {pipe_length}, Bends: {bends}")
    
    # Display current layout if available
    if "layout" in st.session_state and "optimized_layout" not in st.session_state:
        st.subheader("Current Layout")
        st.write("Click 'Optimize Pipe Layout' to generate the optimal pipe route.")
        generator = LayoutGenerator()
        fig = generator.visualize_layout(st.session_state.layout, show=False)
        st.pyplot(fig)
        plt.close(fig)
    
    # Display optimized layout if available
    if "optimized_layout" in st.session_state:
        st.subheader("Optimized Pipe Layout")
        router = PipeRouter()
        fig = router.visualize_solution(st.session_state.optimized_layout, show=False)
        st.pyplot(fig)
        plt.close(fig)

elif input_method == "Draw Layout":
    st.header("Draw Layout")
    st.markdown("""
    **Instructions:**
    - Click and drag to draw obstacles (walls, furniture, etc.)
    - Click once to place inlet (green)
    - Click again to place outlet (red)
    - Click 'Optimize Pipe Layout' to generate the optimal pipe route
    """)
    
    # Initialize canvas
    canvas_size = min(700, grid_size * 20)
    
    # Create a blank canvas
    canvas = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    canvas.fill(255)  # White background
    
    # Draw grid lines
    for i in range(0, grid_size + 1, 1):
        # Horizontal lines
        cv2.line(canvas, (0, i), (grid_size, i), (200, 200, 200), 1)
        # Vertical lines
        cv2.line(canvas, (i, 0), (i, grid_size), (200, 200, 200), 1)
    
    # Convert to PIL Image
    canvas_pil = Image.fromarray(canvas)
    
    # Display canvas
    st.image(canvas_pil, width=canvas_size)
    
    st.markdown("**Note:** Drawing functionality is limited in this demo. For a full implementation, consider using Streamlit's drawing components or JavaScript integration.")
    
    # Placeholder for manual input
    st.subheader("Manual Input")
    
    # Create grid representation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Inlet Position (y, x)**")
        inlet_y = st.number_input("Inlet Y", min_value=0, max_value=grid_size-1, value=0)
        inlet_x = st.number_input("Inlet X", min_value=0, max_value=grid_size-1, value=1)
    
    with col2:
        st.markdown("**Outlet Position (y, x)**")
        outlet_y = st.number_input("Outlet Y", min_value=0, max_value=grid_size-1, value=grid_size-1)
        outlet_x = st.number_input("Outlet X", min_value=0, max_value=grid_size-1, value=grid_size-2)
    
    st.markdown("**Add Obstacles**")
    
    # Initialize obstacles list
    if "obstacles" not in st.session_state:
        st.session_state.obstacles = []
    
    # Add obstacle form
    with st.form("add_obstacle"):
        st.markdown("Add a rectangular obstacle:")
        obs_col1, obs_col2 = st.columns(2)
        
        with obs_col1:
            top = st.number_input("Top (y)", min_value=1, max_value=grid_size-2, value=5)
            height = st.number_input("Height", min_value=1, max_value=grid_size-2, value=5)
        
        with obs_col2:
            left = st.number_input("Left (x)", min_value=1, max_value=grid_size-2, value=5)
            width = st.number_input("Width", min_value=1, max_value=grid_size-2, value=5)
        
        submitted = st.form_submit_button("Add Obstacle")
        
        if submitted:
            st.session_state.obstacles.append({
                "top": top,
                "left": left,
                "height": height,
                "width": width
            })
    
    # Display current obstacles
    if st.session_state.obstacles:
        st.markdown("**Current Obstacles:**")
        for i, obs in enumerate(st.session_state.obstacles):
            st.text(f"Obstacle {i+1}: Top={obs['top']}, Left={obs['left']}, Height={obs['height']}, Width={obs['width']}")
        
        if st.button("Clear Obstacles"):
            st.session_state.obstacles = []
    
    # Create layout from manual input
    if st.button("Create Layout"):
        # Initialize empty grid
        grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        
        # Add walls (border)
        grid[0, :] = 1  # Top wall
        grid[-1, :] = 1  # Bottom wall
        grid[:, 0] = 1  # Left wall
        grid[:, -1] = 1  # Right wall
        
        # Add obstacles
        for obs in st.session_state.obstacles:
            top = obs["top"]
            left = obs["left"]
            height = obs["height"]
            width = obs["width"]
            
            grid[top:top+height, left:left+width] = 1
        
        # Ensure inlet and outlet are not obstacles
        grid[inlet_y, inlet_x] = 0
        grid[outlet_y, outlet_x] = 0
        
        # Create layout dictionary
        layout = {
            "grid": grid.tolist(),
            "inlet": (inlet_y, inlet_x),
            "outlet": (outlet_y, outlet_x),
            "obstacles": st.session_state.obstacles,
            "grid_size": (grid_size, grid_size)
        }
        
        # Store layout in session state
        st.session_state.layout = layout
        
        # Clear any previous optimized layout
        if "optimized_layout" in st.session_state:
            del st.session_state.optimized_layout
        
        # Visualize layout
        generator = LayoutGenerator()
        fig = generator.visualize_layout(layout, show=False)
        st.pyplot(fig)
        plt.close(fig)
    
    # Display current layout if available
    if "layout" in st.session_state and "optimized_layout" not in st.session_state:
        st.subheader("Current Layout")
        st.write("Click 'Optimize Pipe Layout' to generate the optimal pipe route.")
        generator = LayoutGenerator()
        fig = generator.visualize_layout(st.session_state.layout, show=False)
        st.pyplot(fig)
        plt.close(fig)
    
    # Optimize button
    if "layout" in st.session_state and st.button("Optimize Pipe Layout"):
        # Initialize pipe router
        router = PipeRouter(
            bend_penalty=bend_penalty,
            spacing_factor=spacing_factor
        )
        
        # Optimize layout
        optimized_layout = router.optimize_layout(st.session_state.layout)
        
        # Store optimized layout in session state
        st.session_state.optimized_layout = optimized_layout
        
        # Visualize solution
        fig = router.visualize_solution(optimized_layout, show=False)
        st.pyplot(fig)
        plt.close(fig)
        
        # Display statistics
        pipe_path = optimized_layout.get("pipe_path", [])
        pipe_length = len(pipe_path) - 1 if pipe_path else 0
        
        # Count bends
        bends = 0
        for i in range(1, len(pipe_path) - 1):
            prev_dir = router._get_direction(pipe_path[i-1], pipe_path[i])
            next_dir = router._get_direction(pipe_path[i], pipe_path[i+1])
            if prev_dir != next_dir:
                bends += 1
        
        st.info(f"Pipe Length: {pipe_length}, Bends: {bends}")
    
    # Display optimized layout if available
    if "optimized_layout" in st.session_state:
        st.subheader("Optimized Pipe Layout")
        router = PipeRouter()
        fig = router.visualize_solution(st.session_state.optimized_layout, show=False)
        st.pyplot(fig)
        plt.close(fig)

elif input_method == "Upload Layout":
    st.header("Upload Layout")
    
    uploaded_file = st.file_uploader("Upload a layout JSON file", type=["json"])
    
    if uploaded_file is not None:
        try:
            # Load layout from JSON
            layout = json.load(uploaded_file)
            
            # Store layout in session state
            st.session_state.layout = layout
            
            # Visualize layout
            generator = LayoutGenerator()
            fig = generator.visualize_layout(layout, show=False)
            st.pyplot(fig)
            plt.close(fig)
        
        except Exception as e:
            st.error(f"Error loading layout: {str(e)}")
    
    # Optimize button
    if "layout" in st.session_state and st.button("Optimize Pipe Layout"):
        # Initialize pipe router
        router = PipeRouter(
            bend_penalty=bend_penalty,
            spacing_factor=spacing_factor
        )
        
        # Optimize layout
        optimized_layout = router.optimize_layout(st.session_state.layout)
        
        # Store optimized layout in session state
        st.session_state.optimized_layout = optimized_layout
        
        # Visualize solution
        fig = router.visualize_solution(optimized_layout, show=False)
        st.pyplot(fig)
        plt.close(fig)
        
        # Display statistics
        pipe_path = optimized_layout.get("pipe_path", [])
        pipe_length = len(pipe_path) - 1 if pipe_path else 0
        
        # Count bends
        bends = 0
        for i in range(1, len(pipe_path) - 1):
            prev_dir = router._get_direction(pipe_path[i-1], pipe_path[i])
            next_dir = router._get_direction(pipe_path[i], pipe_path[i+1])
            if prev_dir != next_dir:
                bends += 1
        
        st.info(f"Pipe Length: {pipe_length}, Bends: {bends}")
        
        # Download button for optimized layout
        optimized_json = json.dumps(optimized_layout)
        st.download_button(
            label="Download Optimized Layout",
            data=optimized_json,
            file_name="optimized_layout.json",
            mime="application/json"
        )
    
    # Display optimized layout if available
    if "optimized_layout" in st.session_state:
        st.subheader("Optimized Pipe Layout")
        router = PipeRouter()
        fig = router.visualize_solution(st.session_state.optimized_layout, show=False)
        st.pyplot(fig)
        plt.close(fig)

elif input_method == "Upload Image":
    st.header("Upload Image")
    st.markdown("""
    **Instructions:**
    - Upload a JPG or PNG image of a room layout
    - Black pixels will be interpreted as obstacles
    - Green pixels will be interpreted as the inlet
    - Red pixels will be interpreted as the outlet
    - White pixels will be interpreted as free space
    """)
    
    # Option to generate a sample image
    if st.button("Generate Sample Image"):
        # Generate sample image
        sample_img = generate_sample_image(grid_size=(grid_size, grid_size))
        
        # Display sample image
        st.image(sample_img, caption="Sample Image", use_column_width=True)
        
        # Save sample image to session state for processing
        st.session_state.sample_img = sample_img
        
        # Process the sample image
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(sample_img.astype('uint8'))
        
        # Save to a BytesIO object
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Process the image
        layout = process_image(img_byte_arr, grid_size=(grid_size, grid_size))
        
        # Store layout in session state
        st.session_state.layout = layout
        
        # Display the processed image
        st.subheader("Processed Layout")
        processed_img = visualize_processed_image(layout)
        st.image(processed_img, caption="Processed Layout", use_column_width=True)
        
        # Visualize layout using the standard visualizer
        generator = LayoutGenerator()
        fig = generator.visualize_layout(layout, show=False)
        st.pyplot(fig)
        plt.close(fig)
    
    # Option to upload an image
    uploaded_file = st.file_uploader("Or upload your own image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Process the uploaded image
            layout = process_image(uploaded_file, grid_size=(grid_size, grid_size))
            
            # Store layout in session state
            st.session_state.layout = layout
            
            # Display the processed image
            st.subheader("Processed Layout")
            processed_img = visualize_processed_image(layout)
            st.image(processed_img, caption="Processed Layout", use_column_width=True)
            
            # Visualize layout using the standard visualizer
            generator = LayoutGenerator()
            fig = generator.visualize_layout(layout, show=False)
            st.pyplot(fig)
            plt.close(fig)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Optimize button
    if "layout" in st.session_state and st.button("Optimize Pipe Layout"):
        # Initialize pipe router
        router = PipeRouter(
            bend_penalty=bend_penalty,
            spacing_factor=spacing_factor
        )
        
        # Optimize layout
        optimized_layout = router.optimize_layout(st.session_state.layout)
        
        # Store optimized layout in session state
        st.session_state.optimized_layout = optimized_layout
        
        # Visualize solution
        fig = router.visualize_solution(optimized_layout, show=False)
        st.pyplot(fig)
        plt.close(fig)
        
        # Display statistics
        pipe_path = optimized_layout.get("pipe_path", [])
        pipe_length = len(pipe_path) - 1 if pipe_path else 0
        
        # Count bends
        bends = 0
        for i in range(1, len(pipe_path) - 1):
            prev_dir = router._get_direction(pipe_path[i-1], pipe_path[i])
            next_dir = router._get_direction(pipe_path[i], pipe_path[i+1])
            if prev_dir != next_dir:
                bends += 1
        
        st.info(f"Pipe Length: {pipe_length}, Bends: {bends}")
        
        # Display processed image with pipe path
        st.subheader("Optimized Pipe Layout (Image View)")
        processed_img = visualize_processed_image(optimized_layout)
        st.image(processed_img, caption="Optimized Pipe Layout", use_column_width=True)
        
        # Download button for optimized layout
        optimized_json = json.dumps(optimized_layout)
        st.download_button(
            label="Download Optimized Layout",
            data=optimized_json,
            file_name="optimized_layout.json",
            mime="application/json"
        )
    
    # Display optimized layout if available
    if "optimized_layout" in st.session_state:
        st.subheader("Optimized Pipe Layout")
        router = PipeRouter()
        fig = router.visualize_solution(st.session_state.optimized_layout, show=False)
        st.pyplot(fig)
        plt.close(fig)
        
        # Display processed image with pipe path
        st.subheader("Optimized Pipe Layout (Image View)")
        processed_img = visualize_processed_image(st.session_state.optimized_layout)
        st.image(processed_img, caption="Optimized Pipe Layout", use_column_width=True)

# Footer
st.markdown("---")
st.markdown("Pipe Layout Optimizer - AI-powered tool for optimizing pipe layouts in building floors")

if __name__ == "__main__":
    # This will be executed when the script is run directly
    pass 