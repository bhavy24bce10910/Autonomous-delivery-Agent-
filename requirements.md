# Requirements

## System Requirements

- Python 3.7 or higher
- No external dependencies required (uses only standard library modules)

## Python Modules Used

The following standard library modules are used:
- `os` - Operating system interface
- `heapq` - Heap queue algorithm (priority queue)
- `random` - Generate random numbers
- `time` - Time-related functions
- `json` - JSON encoder and decoder
- `argparse` - Command-line argument parsing
- `sys` - System-specific parameters and functions
- `collections.deque` - Double-ended queue
- `typing` - Support for type hints
- `math` - Mathematical functions

## Installation

No installation required beyond Python standard library. All code uses built-in modules.

## Files Structure

```
project/
├── delivery_agent.py          # Main application
├── maps/                      # Map files directory
│   ├── small_map.txt         # Small test map
│   ├── medium_map.txt        # Medium test map
│   ├── large_map.txt         # Large test map
│   └── dynamic_map.txt       # Map with dynamic obstacles
├── experimental_results.csv   # Experimental results
├── analysis_report.txt       # Performance analysis
└── requirements.md           # This file
```

## Usage

### Basic Usage

```bash
# Run specific algorithm
python delivery_agent.py --map maps/small_map.txt --start 0,0 --goal 4,4 --algorithm a_star

# Compare all algorithms
python delivery_agent.py --map maps/medium_map.txt --start 0,0 --goal 9,9 --compare

# Demonstrate dynamic replanning
python delivery_agent.py --map maps/dynamic_map.txt --start 0,0 --goal 7,7 --demo-dynamic

# Show map visualization
python delivery_agent.py --map maps/large_map.txt --start 0,0 --goal 19,19 --algorithm a_star --show-map
```

### Map File Format

Map files should follow this format:
```
rows cols
cost1 cost2 cost3 ...
cost4 cost5 cost6 ...
...
DYNAMIC_OBSTACLES
time_step row col
time_step row col
...
```

Where:
- `rows cols` defines the grid dimensions
- Each subsequent line contains terrain costs for that row
- Cost of 0 indicates an impassable obstacle
- Costs >= 1 indicate traversable terrain with movement cost
- Optional DYNAMIC_OBSTACLES section defines time-based moving obstacles

## Testing

The project includes comprehensive test maps and experimental results:

1. **Small Map (5x5)** - Basic functionality testing
2. **Medium Map (10x10)** - Performance comparison
3. **Large Map (20x20)** - Scalability testing  
4. **Dynamic Map (8x8)** - Dynamic replanning demonstration

## Algorithms Implemented

1. **BFS (Breadth-First Search)** - Uninformed search, finds shortest path in steps
2. **UCS (Uniform Cost Search)** - Uninformed search, finds optimal path by cost
3. **A* Search** - Informed search with Manhattan distance heuristic
4. **Simulated Annealing** - Local search with random restarts for dynamic replanning

## Performance Metrics

The system tracks and reports:
- Path cost (total terrain cost)
- Path length (number of steps)
- Nodes expanded (search efficiency)
- Execution time (performance)
- Success rate (path finding reliability)
