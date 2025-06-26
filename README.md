# Spatial Semantic Pointer Visualization

An interactive web application for visualizing Spatial Semantic Pointer (SSP) transformations and analyzing their properties in real-time.

## Features

### 📊 **Three Interactive Visualizations**

1. **3D SSP Trajectory** - Shows how an SSP moves through 3D space as it's raised to different powers
   - Interactive 3D visualization with mouse controls (drag to rotate, scroll to zoom)
   - Real-time animation showing the SSP trajectory
   - Connection line from origin to current point
   - Background points showing the complete trajectory

2. **Parallel Coordinates** - Displays how each dimension of the SSP changes with exponent variations
   - Line chart showing up to 20 dimensions simultaneously
   - Real-time updates as the exponent changes

3. **Similarity Analysis** - Compares the similarity between original and powered SSPs
   - Dot product and cosine similarity curves
   - Configurable exponent range for analysis

### 🎛️ **Interactive Controls**

- **Exponent Slider** (-10 to 10): Controls the power to which the SSP is raised
- **SSP Dimension** (3 to 50): Sets the dimensionality of the generated SSP
- **Similarity Range** (±5 to ±100): Controls the range for similarity analysis
- **Animation Speed** (0.1x to 2.0x): Adjusts the speed of the automatic animation

### 📈 **Real-time Statistics**

- **Norm**: Euclidean norm of the current SSP
- **Mean**: Average value across all dimensions
- **Standard Deviation**: Measure of value spread

## How to Use

1. **Open the Website**: Simply open `index.html` in any modern web browser
2. **Interact with Controls**: Use the sliders to adjust parameters and see real-time updates
3. **Explore 3D View**: 
   - Drag to rotate the 3D visualization
   - Scroll to zoom in/out
   - Watch the animated trajectory
4. **Analyze Patterns**: Observe how different parameters affect the SSP behavior

## Technical Implementation

The website is built as a **standalone application** using:

- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Core SSP algorithms reimplemented from Python
- **Three.js**: 3D visualization and animation
- **Chart.js**: 2D plotting for parallel coordinates and similarity analysis

### SSP Algorithm Implementation

The core SSP functions have been reimplemented in JavaScript:

- `makeGoodUnitary()`: Generates normalized unitary SSPs
- `powerSSP()`: Raises SSPs to arbitrary powers with phase and amplitude transformations
- Statistical functions: norm, mean, standard deviation, dot product, cosine similarity

## Browser Compatibility

Works in all modern browsers that support:
- ES6 JavaScript features
- WebGL (for 3D visualization)
- HTML5 Canvas (for 2D charts)

## Academic Context

This visualization tool demonstrates the mathematical properties of Spatial Semantic Pointers, which are used in:
- Vector Symbolic Architectures
- Cognitive modeling
- Neural representation learning
- Spatial reasoning systems

The visualizations help understand how SSP transformations preserve and modify spatial relationships through mathematical operations.

---

**Note**: This is a standalone web application - no server setup or Python installation required!