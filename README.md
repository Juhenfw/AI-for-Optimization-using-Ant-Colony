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

## How to Run

1. **Install Dependencies**: Ensure you have installed all supporting libraries. If you have set up a virtual environment, you can install them via terminal using:
   ```bash
   pip install -r requirements.txt
   ```
   *(Or manually: `pip install numpy pandas matplotlib ipython argparse`)*

2. **Execute the Program**: You can run the optimization script directly from the terminal with various dynamic options using CLI arguments. No need to edit the code manually!
   ```bash
   # Run with default parameters (City: Surabaya, Ants: 64, Iterations: 100)
   python src/route_optimization.py

   # Run for a different city with custom parameters
   python src/route_optimization.py --city Sidoarjo --ants 100 --iterations 200

   # Adjust ACO hyperparameters (alpha: pheromone weight, beta: distance weight)
   python src/route_optimization.py --city Depok --alpha 1.5 --beta 2.0
   
   # Run and generate additional statistical analysis
   python src/route_optimization.py --city Surabaya --stats

   # Display help menu to see all available flags
   python src/route_optimization.py --help
   ```

3. **Check the Results**: The initial mapping image (`Kecamatan.png`) and the route optimization result (`path.png`) will automatically be saved neatly inside the `assets/results/` directory.

## Visual Examples

*(Original Map of Surabaya)*  
![Peta Kantor Kecamatan Surabaya](assets/maps/Location_District_Office_Surabaya.png)

*(Example of Optimal TSP Path)*  
![Jalur Optimal](assets/results/path.png)

## License

This project is licensed under the **Apache License, Version 2.0** - see the [LICENSE](LICENSE) file for details on usage, reproduction, and distribution terms.