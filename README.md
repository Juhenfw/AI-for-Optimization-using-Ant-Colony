# Route Optimization for District Offices (TSP) using Ant Colony Optimization

*[🇮🇩 Baca dalam Bahasa Indonesia (Read in Indonesian)](README-id.md)*

---

This repository contains a Python script aimed at solving the *Traveling Salesman Problem* (TSP) to find the most efficient routing between district offices (*Kantor Kecamatan*).

Utilizing the **Ant Colony Optimization (ACO)** algorithm, this script calculates the optimal route and provides a visual representation of the path with the minimum travel distance. The program has been dynamically designed so it can be easily adapted for various cities such as Surabaya, Sidoarjo, and Depok.

## Key Features

- **Multi-City Configuration (Dynamic)**: Equipped with a centralized configuration block. Simply change the `CITY_NAME` variable, and the program will automatically adjust the coordinate readings and map images.
- **Smart Image Detector & Safe Saving**: The system intelligently detects the availability of maps in various formats (`.png` or `.jpg`) and automatically creates the output folder structure (`assets/results/`) if it does not exist to prevent errors.
- **High-Performance Optimization Algorithm**: Uses a NumPy-vectorized *Ant Colony Solver*, ensuring the search for the best route runs swiftly.
- **Map & Route Visualization**: Generates a point-to-point route image that is directly overlaid onto the city's original map.
- **Analysis and Statistics**: Provides statistical table calculations using Pandas to compare the performance of purely converged algorithm results against time-constrained (scheduled) ones.

## Technologies Used

- **Python 3.x**: Main programming language.
- **NumPy**: Accelerates mathematical operations and ant movements (vectorization) within the algorithm.
- **Pandas**: Used for comparative statistical computing analysis on the final optimization results.
- **Matplotlib**: The main engine for rendering coordinates and routing paths over the map image.

## How to Use

1. **Install Dependencies**: Ensure you have installed all supporting libraries. You can install them via terminal:
   ```bash
   pip install numpy pandas matplotlib ipython
   ```
2. **City Configuration**: Open the main script (e.g., `src/route_optimization.py`). At the very top, locate the **PENGATURAN UTAMA** (MAIN SETTINGS) block and change the variable value according to the city you want to analyze:
   ```python
   CITY_NAME = 'Surabaya' # Options: 'Surabaya', 'Sidoarjo', 'Depok'
   ```
3. **Execute the Program**: Run the script via terminal or an IDE (such as VS Code / Jupyter).
4. **Check the Results**: The initial mapping image (`Kecamatan.png`) and the route optimization result (`path.png`) will automatically be saved neatly inside the `assets/results/` directory.

## Visual Examples

*(Original Map of Surabaya)*  
![Peta Kantor Kecamatan Surabaya](assets/maps/Location_District_Office_Surabaya.png)

*(Example of Optimal TSP Path)*  
![Jalur Optimal](assets/results/path.png)

## License

This project is licensed under the **Apache License, Version 2.0**[cite: 6] - see the [LICENSE](LICENSE) file for details on usage, reproduction, and distribution terms.