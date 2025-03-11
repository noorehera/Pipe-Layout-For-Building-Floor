# Pipe Layout Optimizer

An AI-powered tool for generating optimized 2D pipe layouts for building floors based on room layouts and inlet/outlet locations.

## Features

- Synthetic dataset generation for training and testing
- A* pathfinding algorithm for optimal pipe routing
- Constraints handling (obstacle avoidance, spacing rules)
- Interactive web interface using Streamlit
- Visualization of generated layouts

## Project Structure

```
.
├── data/                  # Synthetic dataset storage
├── models/                # AI model implementation
├── utils/                 # Utility functions
├── app.py                 # Streamlit web application
├── generate_dataset.py    # Dataset generation script
└── requirements.txt       # Project dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Generate a synthetic dataset:
```bash
python generate_dataset.py
```

2. Run the Streamlit web application:
```bash
streamlit run app.py
```

3. Use the web interface to:
   - Upload or draw a room layout
   - Specify inlet and outlet locations
   - Generate an optimized pipe layout
   - Visualize the results

## Implementation Details

- Room layouts are represented as grid-based maps
- A* algorithm is used for pathfinding with custom heuristics
- Constraints include obstacle avoidance and minimizing pipe bends
- Visualization is done using Matplotlib and OpenCV 