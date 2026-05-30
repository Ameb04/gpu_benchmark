# GPU Benchmarking

This project provides a comprehensive benchmarking and analysis framework for GPU operations. It includes implementations using naive methods, NumPy, and PyTorch, and offers detailed performance measurement and visualization through a PyQt5-based dashboard.

## Features

- Benchmarking GPU-related operations with multiple implementations:
  - Naive (manual) implementation
  - NumPy-based optimized operations
  - PyTorch-based GPU acceleration
- Supports key GPU operations including:  
  - Matrix multiplication (matmul)  
  - Matrix inversion (matinv)  
  - Distance calculations (dist)  
- Performance measurement using execution time and memory usage
- Detailed analysis and visualization via a PyQt5 GUI dashboard
- Data management and result export functionality
- Support for comparing multiple methods across different matrix sizes

## Project Structure
```
.  
│  
├── main.py              # Main GUI application  
├── proj.py              # Core benchmarking functions  
├── gpu_monitor.py       # GPU monitoring utility  
├── analyse.py           # Data analysis and visualization  
├── results.csv          # Benchmark results  
└── gpu_log.csv          # GPU monitoring logs
```


## Prerequisites

- Python 3.10 or higher
- Required Python packages (install via pip):

  ```bash
  pip install numpy pandas matplotlib seaborn pyqt5 pytorch
  ```

## How to run

Run Dashboard.py to perform benchmarking, storing data, and visualasing data and analysing:

  ```bash
  python Dashboard.py
  ```
Results are saved in results.csv and can be reloaded for later analysis.

## Contributing
If you find any bugs or have ideas to make this project better, just send a pull request.

## License
This project is licensed under the MIT License.

## Authors
AmirMasoud Ebrahimi, 
Alireza Mohandesi, 
Sajjad Agheli, 
AmirReza Jafari
