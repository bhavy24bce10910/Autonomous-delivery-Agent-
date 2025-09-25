# Autonomous Delivery Agent

**CSA2001 - Fundamentals of AI and ML - Project 1**

An intelligent delivery agent that navigates 2D grid cities using multiple pathfinding algorithms including BFS, UCS, A*, and Simulated Annealing.

## 🚀 Features

- **Multiple Algorithms**: BFS, Uniform Cost Search, A*, Simulated Annealing
- **Dynamic Replanning**: Handles moving obstacles and changing environments
- **Performance Analysis**: Comprehensive comparison of algorithms
- **CLI Interface**: Easy-to-use command-line interface
- **Visualization**: Grid map display and path visualization
- **Comprehensive Testing**: Multiple test scenarios and maps

## 📁 Project Structure

```
autonomous-delivery-agent/
├── delivery_agent.py          # Main application
├── maps/                      # Test maps
│   ├── small_map.txt          # 5x5 basic test map
│   ├── medium_map.txt         # 10x10 performance test map
│   ├── large_map.txt          # 20x20 scalability test map
│   └── dynamic_map.txt        # 8x8 dynamic obstacles map
├── experimental_results.csv   # Performance comparison results
├── analysis_report.txt        # Detailed analysis and findings
├── requirements.md            # Installation and system requirements
└── README.md                  # This file
```

## 🛠 Installation

No external dependencies required! Uses only Python standard library.

**Requirements:**
- Python 3.7 or higher

**Setup:**
```bash
git clone <repository-url>
cd autonomous-delivery-agent
```

## 🎮 Usage

### Basic Commands

```bash
# Run A* algorithm on small map
python delivery_agent.py --map maps/small_map.txt --start 0,0 --goal 4,4 --algorithm a_star

# Compare all algorithms on medium map
python delivery_agent.py --map maps/medium_map.txt --start 0,0 --goal 9,9 --compare

# Demonstrate dynamic replanning
python delivery_agent.py --map maps/dynamic_map.txt --start 0,0 --goal 7,7 --demo-dynamic

# Show map visualization
python delivery_agent.py --map maps/large_map.txt --start 0,0 --goal 19,19 --algorithm a_star --show-map
```

### Command Line Options

- `--map`: Map file to load (required)
- `--start`: Start position as "row,col" (required)
- `--goal`: Goal position as "row,col" (required)
- `--algorithm`: Choose algorithm (bfs, ucs, a_star, simulated_annealing)
- `--compare`: Compare all algorithms
- `--demo-dynamic`: Demonstrate dynamic replanning
- `--show-map`: Display map visualization

### Example Output

```
Map loaded successfully from maps/small_map.txt
Map size: 5x5
Agent setup complete. Start: (0, 0), Goal: (4, 4)

Running A_STAR...
✓ Path found!
Path length: 9 steps
Path cost: 8
Nodes expanded: 20
Execution time: 0.0002 seconds
Path: (0,0) -> (1,0) -> (2,0) -> (3,0) -> (4,0) -> (4,1) -> (4,2) -> (4,3) -> (4,4)
```

## 🧪 Algorithms

### 1. Breadth-First Search (BFS)
- **Type**: Uninformed search
- **Optimality**: Optimal for unit costs
- **Use Case**: When step count matters more than terrain cost

### 2. Uniform Cost Search (UCS)
- **Type**: Uninformed search
- **Optimality**: Always optimal
- **Use Case**: When optimality is crucial regardless of computation time

### 3. A* Search
- **Type**: Informed search
- **Heuristic**: Manhattan distance
- **Optimality**: Optimal with admissible heuristic
- **Use Case**: Best general-purpose pathfinding algorithm

### 4. Simulated Annealing
- **Type**: Local search
- **Optimality**: Near-optimal
- **Use Case**: Dynamic environments with frequent changes

## 📊 Performance Results

| Algorithm | Small Map | Medium Map | Large Map | Dynamic Map |
|-----------|-----------|------------|-----------|-------------|
| BFS       | Fast      | Good       | Slower    | Basic       |
| UCS       | Optimal   | Optimal    | Optimal   | Good        |
| A*        | Fast+Opt  | Fast+Opt   | Fast+Opt  | Excellent   |
| Sim. Ann. | Variable  | Variable   | Variable  | Adaptive    |

*Detailed results available in `experimental_results.csv`*

## 🎯 Key Features

### Dynamic Replanning
- Real-time obstacle detection
- Automatic path recalculation
- Support for time-based moving obstacles
- Performance comparison between original and replanned paths

### Environment Modeling
- Variable terrain costs (1-5)
- Static obstacles (cost = 0)
- Dynamic/moving obstacles
- Time-based obstacle scheduling

### Performance Metrics
- Path cost optimization
- Node expansion efficiency
- Execution time measurement
- Success rate tracking

## 🧩 Map Format

Maps use a simple text format:
```
5 5
1 1 1 1 1
1 3 0 1 1
1 0 2 1 1
1 1 1 4 1
1 1 1 1 1
DYNAMIC_OBSTACLES
0 3 1
1 3 2
2 3 3
```

Where:
- First line: `rows cols`
- Grid values: terrain costs (0 = obstacle, ≥1 = passable)
- Optional dynamic obstacles with time steps

## 🔬 Experimental Analysis

The project includes comprehensive experimental analysis covering:
- Algorithm efficiency comparison
- Scalability across different map sizes
- Optimality guarantees
- Dynamic environment adaptation
- Performance recommendations

See `analysis_report.txt` for detailed findings.

## 🏆 Project Highlights

- ✅ **Complete Implementation**: All required algorithms working
- ✅ **Dynamic Replanning**: Proof-of-concept with logging
- ✅ **Multiple Test Maps**: 4 different scenarios
- ✅ **Performance Analysis**: Comprehensive experimental results
- ✅ **CLI Interface**: Professional command-line tool
- ✅ **Documentation**: Thorough README and requirements
- ✅ **Reproducibility**: Deterministic results with clear instructions

## 🤝 Contributing

This is an academic project for CSA2001. The implementation follows the project requirements:
- Rational agent design maximizing delivery efficiency
- Multiple search algorithms with performance comparison
- Dynamic replanning capabilities
- Comprehensive testing and analysis

## 📝 License

Academic project for educational purposes.

---

**Author**: [Your Name]  
**Course**: CSA2001 - Fundamentals of AI and ML  
**Project**: Autonomous Delivery Agent (Project 1)
