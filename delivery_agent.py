#!/usr/bin/env python3
"""
Autonomous Delivery Agent
CSA2001 - Fundamentals of AI and ML - Project 1

This module implements an autonomous delivery agent that navigates a 2D grid city
to deliver packages using various pathfinding algorithms.
"""

import os
import heapq
import random
import time
import json
import argparse
import sys
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
import math

class DeliveryAgent:
    """
    Autonomous delivery agent that can navigate 2D grid environments using
    different pathfinding algorithms.
    """

    def __init__(self, grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Initialize the delivery agent

        Args:
            grid: 2D list representing the environment (terrain costs)
            start: Starting position (row, col)
            goal: Goal position (row, col)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.goal = goal
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        self.dynamic_obstacles = set()
        self.statistics = {
            'nodes_expanded': 0,
            'path_cost': 0,
            'execution_time': 0.0
        }

    def reset_stats(self):
        """Reset statistics for new search"""
        self.statistics = {
            'nodes_expanded': 0,
            'path_cost': 0,
            'execution_time': 0.0
        }

    def is_valid(self, row: int, col: int) -> bool:
        """Check if position is valid and not blocked"""
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row][col] > 0 and
                (row, col) not in self.dynamic_obstacles)

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int, int]]:
        """Get valid neighbors with their costs"""
        neighbors = []
        row, col = pos

        for dr, dc in self.directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid(new_row, new_col):
                cost = self.grid[new_row][new_col]
                neighbors.append((new_row, new_col, cost))

        return neighbors

    def bfs(self) -> Tuple[Optional[List[Tuple[int, int]]], Dict]:
        """Breadth-First Search implementation"""
        start_time = time.time()
        self.reset_stats()

        queue = deque([(self.start, [self.start])])
        visited = set([self.start])

        while queue:
            current, path = queue.popleft()
            self.statistics['nodes_expanded'] += 1

            if current == self.goal:
                # Calculate path cost
                path_cost = 0
                for i in range(1, len(path)):
                    row, col = path[i]
                    path_cost += self.grid[row][col]

                self.statistics['path_cost'] = path_cost
                self.statistics['execution_time'] = time.time() - start_time
                return path, self.statistics

            for next_row, next_col, cost in self.get_neighbors(current):
                next_pos = (next_row, next_col)
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))

        self.statistics['execution_time'] = time.time() - start_time
        return None, self.statistics

    def uniform_cost_search(self) -> Tuple[Optional[List[Tuple[int, int]]], Dict]:
        """Uniform Cost Search implementation"""
        start_time = time.time()
        self.reset_stats()

        # Priority queue: (cost, current_pos, path)
        heap = [(0, self.start, [self.start])]
        visited = set()

        while heap:
            current_cost, current_pos, path = heapq.heappop(heap)

            if current_pos in visited:
                continue

            visited.add(current_pos)
            self.statistics['nodes_expanded'] += 1

            if current_pos == self.goal:
                self.statistics['path_cost'] = current_cost
                self.statistics['execution_time'] = time.time() - start_time
                return path, self.statistics

            for next_row, next_col, move_cost in self.get_neighbors(current_pos):
                next_pos = (next_row, next_col)
                if next_pos not in visited:
                    new_cost = current_cost + move_cost
                    new_path = path + [next_pos]
                    heapq.heappush(heap, (new_cost, next_pos, new_path))

        self.statistics['execution_time'] = time.time() - start_time
        return None, self.statistics

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def a_star_search(self) -> Tuple[Optional[List[Tuple[int, int]]], Dict]:
        """A* Search implementation"""
        start_time = time.time()
        self.reset_stats()

        # Priority queue: (f_cost, g_cost, current_pos, path)
        heap = [(0, 0, self.start, [self.start])]
        visited = set()
        g_costs = {self.start: 0}

        while heap:
            f_cost, g_cost, current_pos, path = heapq.heappop(heap)

            if current_pos in visited:
                continue

            visited.add(current_pos)
            self.statistics['nodes_expanded'] += 1

            if current_pos == self.goal:
                self.statistics['path_cost'] = g_cost
                self.statistics['execution_time'] = time.time() - start_time
                return path, self.statistics

            for next_row, next_col, move_cost in self.get_neighbors(current_pos):
                next_pos = (next_row, next_col)
                if next_pos not in visited:
                    new_g_cost = g_cost + move_cost

                    if next_pos not in g_costs or new_g_cost < g_costs[next_pos]:
                        g_costs[next_pos] = new_g_cost
                        h_cost = self.manhattan_distance(next_pos, self.goal)
                        f_cost = new_g_cost + h_cost
                        new_path = path + [next_pos]
                        heapq.heappush(heap, (f_cost, new_g_cost, next_pos, new_path))

        self.statistics['execution_time'] = time.time() - start_time
        return None, self.statistics

    def simulated_annealing(self, max_iterations: int = 1000, initial_temp: float = 100.0) -> Tuple[Optional[List[Tuple[int, int]]], Dict]:
        """Simulated Annealing for local search with replanning"""
        start_time = time.time()
        self.reset_stats()

        # Start with a random path or simple path
        current_path = self._generate_random_path()
        if not current_path:
            return None, self.statistics

        current_cost = self._calculate_path_cost(current_path)
        best_path = current_path[:]
        best_cost = current_cost

        temperature = initial_temp

        for iteration in range(max_iterations):
            self.statistics['nodes_expanded'] += 1

            # Generate neighbor solution
            neighbor_path = self._get_neighbor_solution(current_path)
            if not neighbor_path:
                continue

            neighbor_cost = self._calculate_path_cost(neighbor_path)

            # Accept or reject the neighbor
            if neighbor_cost < current_cost or random.random() < math.exp(-(neighbor_cost - current_cost) / temperature):
                current_path = neighbor_path
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_path = current_path[:]
                    best_cost = current_cost

            # Cool down
            temperature *= 0.995

            if temperature < 0.01:
                break

        self.statistics['path_cost'] = best_cost
        self.statistics['execution_time'] = time.time() - start_time

        return best_path if self._is_valid_path(best_path) else None, self.statistics

    def _generate_random_path(self) -> Optional[List[Tuple[int, int]]]:
        """Generate a random valid path from start to goal"""
        path = [self.start]
        current = self.start
        visited = set([self.start])

        for _ in range(self.rows * self.cols):
            if current == self.goal:
                return path

            neighbors = self.get_neighbors(current)
            valid_neighbors = [(r, c) for r, c, _ in neighbors if (r, c) not in visited]

            if not valid_neighbors:
                # Backtrack or restart
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                    visited.remove(current)
                    continue
                else:
                    return None

            # Choose neighbor closest to goal
            next_pos = min(valid_neighbors, key=lambda pos: self.manhattan_distance(pos, self.goal))
            path.append(next_pos)
            visited.add(next_pos)
            current = next_pos

        return None

    def _calculate_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """Calculate the total cost of a path"""
        if not path or len(path) < 2:
            return float('inf')

        cost = 0
        for i in range(1, len(path)):
            row, col = path[i]
            if not self.is_valid(row, col):
                return float('inf')
            cost += self.grid[row][col]

        return cost

    def _get_neighbor_solution(self, path: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """Generate a neighbor solution by modifying the current path"""
        if len(path) < 3:
            return path

        new_path = path[:]

        # Randomly select a segment to modify
        start_idx = random.randint(1, len(path) - 2)
        end_idx = min(start_idx + random.randint(1, 3), len(path) - 1)

        # Try to find alternative path for this segment
        segment_start = path[start_idx - 1]
        segment_end = path[end_idx]

        # Simple local modification: try different intermediate points
        intermediate_positions = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row = segment_start[0] + dr
                new_col = segment_start[1] + dc
                if self.is_valid(new_row, new_col):
                    intermediate_positions.append((new_row, new_col))

        if intermediate_positions:
            chosen_intermediate = random.choice(intermediate_positions)
            new_path = (path[:start_idx] + 
                       [chosen_intermediate] + 
                       path[end_idx:])

        return new_path if self._is_valid_path(new_path) else path

    def _is_valid_path(self, path: List[Tuple[int, int]]) -> bool:
        """Check if a path is valid"""
        if not path or path[0] != self.start or path[-1] != self.goal:
            return False

        for i in range(len(path)):
            if not self.is_valid(path[i][0], path[i][1]):
                return False

        # Check if consecutive positions are adjacent
        for i in range(1, len(path)):
            prev_row, prev_col = path[i-1]
            curr_row, curr_col = path[i]
            if abs(prev_row - curr_row) + abs(prev_col - curr_col) != 1:
                return False

        return True

    def add_dynamic_obstacle(self, pos: Tuple[int, int]):
        """Add a dynamic obstacle"""
        self.dynamic_obstacles.add(pos)

    def remove_dynamic_obstacle(self, pos: Tuple[int, int]):
        """Remove a dynamic obstacle"""
        self.dynamic_obstacles.discard(pos)

    def replan_with_dynamic_obstacles(self, algorithm: str = "a_star") -> Tuple[Optional[List[Tuple[int, int]]], Dict]:
        """Replan when dynamic obstacles are detected"""
        print(f"Dynamic obstacle detected! Replanning with {algorithm}...")

        if algorithm == "bfs":
            return self.bfs()
        elif algorithm == "ucs":
            return self.uniform_cost_search()
        elif algorithm == "a_star":
            return self.a_star_search()
        elif algorithm == "simulated_annealing":
            return self.simulated_annealing()
        else:
            return self.a_star_search()


class GridMap:
    """
    Represents a 2D grid environment with terrain costs and obstacles.
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.grid = [[1 for _ in range(cols)] for _ in range(rows)]
        self.static_obstacles = set()
        self.dynamic_obstacles = {}  # {time_step: set of positions}

    def set_terrain_cost(self, row: int, col: int, cost: int):
        """Set terrain cost for a cell"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row][col] = cost

    def add_static_obstacle(self, row: int, col: int):
        """Add static obstacle (impassable)"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row][col] = 0
            self.static_obstacles.add((row, col))

    def add_dynamic_obstacle(self, row: int, col: int, time_step: int):
        """Add dynamic obstacle at specific time step"""
        if time_step not in self.dynamic_obstacles:
            self.dynamic_obstacles[time_step] = set()
        self.dynamic_obstacles[time_step].add((row, col))

    def get_dynamic_obstacles_at_time(self, time_step: int) -> Set[Tuple[int, int]]:
        """Get dynamic obstacles at specific time step"""
        return self.dynamic_obstacles.get(time_step, set())

    def save_to_file(self, filename: str):
        """Save grid map to file"""
        with open(filename, 'w') as f:
            f.write(f"{self.rows} {self.cols}\n")
            for row in self.grid:
                f.write(" ".join(map(str, row)) + "\n")

            # Save dynamic obstacles
            f.write("DYNAMIC_OBSTACLES\n")
            for time_step, obstacles in self.dynamic_obstacles.items():
                for row, col in obstacles:
                    f.write(f"{time_step} {row} {col}\n")

    @classmethod
    def load_from_file(cls, filename: str):
        """Load grid map from file"""
        with open(filename, 'r') as f:
            lines = f.readlines()

        rows, cols = map(int, lines[0].strip().split())
        grid_map = cls(rows, cols)

        # Load grid
        for i in range(1, rows + 1):
            row_data = list(map(int, lines[i].strip().split()))
            grid_map.grid[i-1] = row_data

        # Load dynamic obstacles if they exist
        dynamic_start = -1
        for i, line in enumerate(lines):
            if line.strip() == "DYNAMIC_OBSTACLES":
                dynamic_start = i + 1
                break

        if dynamic_start != -1:
            for i in range(dynamic_start, len(lines)):
                if lines[i].strip():
                    time_step, row, col = map(int, lines[i].strip().split())
                    grid_map.add_dynamic_obstacle(row, col, time_step)

        return grid_map

    def print_grid(self):
        """Print the grid for visualization"""
        for row in self.grid:
            print(" ".join(f"{cell:2d}" for cell in row))


class DeliveryAgentCLI:
    """
    Command-line interface for the delivery agent.
    """

    def __init__(self):
        self.agent = None
        self.grid_map = None

    def load_map(self, map_file: str):
        """Load map from file"""
        try:
            self.grid_map = GridMap.load_from_file(map_file)
            print(f"Map loaded successfully from {map_file}")
            print(f"Map size: {self.grid_map.rows}x{self.grid_map.cols}")
            return True
        except Exception as e:
            print(f"Error loading map: {e}")
            return False

    def setup_agent(self, start: Tuple[int, int], goal: Tuple[int, int]):
        """Setup the delivery agent"""
        if not self.grid_map:
            print("Please load a map first!")
            return False

        # Validate start and goal positions
        if not (0 <= start[0] < self.grid_map.rows and 0 <= start[1] < self.grid_map.cols):
            print(f"Invalid start position: {start}")
            return False

        if not (0 <= goal[0] < self.grid_map.rows and 0 <= goal[1] < self.grid_map.cols):
            print(f"Invalid goal position: {goal}")
            return False

        if self.grid_map.grid[start[0]][start[1]] == 0:
            print(f"Start position {start} is blocked!")
            return False

        if self.grid_map.grid[goal[0]][goal[1]] == 0:
            print(f"Goal position {goal} is blocked!")
            return False

        self.agent = DeliveryAgent(self.grid_map.grid, start, goal)
        print(f"Agent setup complete. Start: {start}, Goal: {goal}")
        return True

    def run_algorithm(self, algorithm: str, show_path: bool = True):
        """Run specified algorithm"""
        if not self.agent:
            print("Please setup agent first!")
            return None, None

        print(f"\nRunning {algorithm.upper()}...")

        if algorithm == "bfs":
            path, stats = self.agent.bfs()
        elif algorithm == "ucs":
            path, stats = self.agent.uniform_cost_search()
        elif algorithm == "a_star":
            path, stats = self.agent.a_star_search()
        elif algorithm == "simulated_annealing":
            path, stats = self.agent.simulated_annealing()
        else:
            print(f"Unknown algorithm: {algorithm}")
            return None, None

        # Print results
        if path:
            print(f"✓ Path found!")
            print(f"Path length: {len(path)} steps")
            print(f"Path cost: {stats['path_cost']}")
            print(f"Nodes expanded: {stats['nodes_expanded']}")
            print(f"Execution time: {stats['execution_time']:.4f} seconds")

            if show_path:
                print("Path:", " -> ".join(f"({r},{c})" for r, c in path))
        else:
            print("✗ No path found!")
            print(f"Nodes expanded: {stats['nodes_expanded']}")
            print(f"Execution time: {stats['execution_time']:.4f} seconds")

        return path, stats

    def demonstrate_dynamic_replanning(self):
        """Demonstrate dynamic replanning capability"""
        if not self.agent or not self.grid_map:
            print("Please load map and setup agent first!")
            return

        print("\n" + "="*50)
        print("DYNAMIC REPLANNING DEMONSTRATION")
        print("="*50)

        # First, find initial path
        print("Finding initial path with A*...")
        initial_path, initial_stats = self.agent.a_star_search()

        if not initial_path:
            print("Cannot find initial path!")
            return

        print(f"Initial path found: {len(initial_path)} steps, cost: {initial_stats['path_cost']}")
        print("Initial path:", " -> ".join(f"({r},{c})" for r, c in initial_path[:5]) + "...")

        # Simulate dynamic obstacle appearing
        if len(initial_path) > 3:
            obstacle_pos = initial_path[len(initial_path)//2]  # Place obstacle in middle of path
            print(f"\n🚧 DYNAMIC OBSTACLE DETECTED at position {obstacle_pos}!")

            # Add dynamic obstacle
            self.agent.add_dynamic_obstacle(obstacle_pos)

            # Replan
            print("Replanning with A*...")
            new_path, new_stats = self.agent.replan_with_dynamic_obstacles("a_star")

            if new_path:
                print(f"✓ New path found: {len(new_path)} steps, cost: {new_stats['path_cost']}")
                print("New path:", " -> ".join(f"({r},{c})" for r, c in new_path[:5]) + "...")

                # Compare paths
                print(f"\nComparison:")
                print(f"Original: {len(initial_path)} steps, cost {initial_stats['path_cost']}")
                print(f"Replanned: {len(new_path)} steps, cost {new_stats['path_cost']}")
                print(f"Additional cost due to obstacle: {new_stats['path_cost'] - initial_stats['path_cost']}")
            else:
                print("✗ No alternative path found!")

        # Simulate time-based dynamic obstacles
        print(f"\n⏰ SIMULATING TIME-BASED DYNAMIC OBSTACLES...")

        # Check if map has dynamic obstacles
        if self.grid_map.dynamic_obstacles:
            for time_step in range(3):
                dynamic_obs = self.grid_map.get_dynamic_obstacles_at_time(time_step)
                if dynamic_obs:
                    print(f"Time {time_step}: Dynamic obstacles at {dynamic_obs}")

                    # Clear previous dynamic obstacles
                    self.agent.dynamic_obstacles.clear()

                    # Add current time step obstacles
                    for obs_pos in dynamic_obs:
                        self.agent.add_dynamic_obstacle(obs_pos)

                    # Replan
                    path, stats = self.agent.replan_with_dynamic_obstacles("a_star")
                    if path:
                        print(f"  ✓ Replanned path: {len(path)} steps, cost: {stats['path_cost']}")
                    else:
                        print(f"  ✗ No path available at time {time_step}")

    def compare_algorithms(self):
        """Compare all algorithms on current map"""
        if not self.agent:
            print("Please setup agent first!")
            return

        algorithms = ["bfs", "ucs", "a_star", "simulated_annealing"]
        results = {}

        print("\n" + "="*60)
        print("ALGORITHM COMPARISON")
        print("="*60)

        for alg in algorithms:
            print(f"\nTesting {alg.upper()}...")
            path, stats = self.run_algorithm(alg, show_path=False)
            results[alg] = {
                'path_found': path is not None,
                'path_length': len(path) if path else 0,
                'path_cost': stats['path_cost'],
                'nodes_expanded': stats['nodes_expanded'],
                'execution_time': stats['execution_time']
            }

        # Print comparison table
        print(f"\n{'Algorithm':<20} {'Found':<6} {'Length':<8} {'Cost':<8} {'Nodes':<8} {'Time(s)':<10}")
        print("-" * 68)

        for alg, result in results.items():
            found = "Yes" if result['path_found'] else "No"
            length = result['path_length'] if result['path_found'] else "-"
            cost = result['path_cost'] if result['path_found'] else "-"
            nodes = result['nodes_expanded']
            time_val = f"{result['execution_time']:.4f}"

            print(f"{alg.upper():<20} {found:<6} {length:<8} {cost:<8} {nodes:<8} {time_val:<10}")


def run_demo():
    """
    Run a demonstration of the delivery agent.
    This function can be called directly from IDEs without command line arguments.
    """
    print("="*60)
    print("AUTONOMOUS DELIVERY AGENT - DEMO MODE")
    print("="*60)

    # Create CLI instance
    cli = DeliveryAgentCLI()

    # Load a demo map
    map_file = "maps/small_map.txt"
    if not cli.load_map(map_file):
        print("Error: Could not load demo map. Make sure maps/small_map.txt exists.")
        return

    # Show map
    print("\nDemo Map:")
    cli.grid_map.print_grid()

    # Setup agent
    start = (0, 0)
    goal = (4, 4)
    if not cli.setup_agent(start, goal):
        return

    # Run comparison of all algorithms
    cli.compare_algorithms()

    # Test dynamic replanning if dynamic map exists
    print("\n" + "="*60)
    print("TESTING DYNAMIC REPLANNING")
    print("="*60)

    dynamic_map = "maps/dynamic_map.txt"
    if cli.load_map(dynamic_map):
        if cli.setup_agent((0, 0), (7, 7)):
            cli.demonstrate_dynamic_replanning()


def main():
    """Main CLI function"""
    # Check if running in interactive mode (like Spyder)
    if hasattr(sys, 'ps1') or sys.flags.interactive:
        print("Running in interactive mode. Starting demo...")
        run_demo()
        return 0

    # Check if no arguments provided (running from IDE without args)
    if len(sys.argv) == 1:
        print("No command line arguments provided. Running demo mode...")
        run_demo()
        return 0

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Autonomous Delivery Agent")
    parser.add_argument("--map", required=True, help="Map file to load")
    parser.add_argument("--start", required=True, help="Start position (row,col)")
    parser.add_argument("--goal", required=True, help="Goal position (row,col)")
    parser.add_argument("--algorithm", choices=["bfs", "ucs", "a_star", "simulated_annealing"], 
                       help="Algorithm to run")
    parser.add_argument("--compare", action="store_true", help="Compare all algorithms")
    parser.add_argument("--demo-dynamic", action="store_true", help="Demonstrate dynamic replanning")
    parser.add_argument("--show-map", action="store_true", help="Show the map")

    try:
        args = parser.parse_args()
    except SystemExit:
        print("\nError parsing arguments. Running demo mode instead...")
        run_demo()
        return 0

    # Create CLI instance
    cli = DeliveryAgentCLI()

    # Load map
    if not cli.load_map(args.map):
        return 1

    # Show map if requested
    if args.show_map:
        print("\nMap visualization:")
        cli.grid_map.print_grid()

    # Parse start and goal
    try:
        start = tuple(map(int, args.start.split(',')))
        goal = tuple(map(int, args.goal.split(',')))
    except:
        print("Error: Start and goal must be in format 'row,col'")
        return 1

    # Setup agent
    if not cli.setup_agent(start, goal):
        return 1

    # Run requested operations
    if args.compare:
        cli.compare_algorithms()
    elif args.demo_dynamic:
        cli.demonstrate_dynamic_replanning()
    elif args.algorithm:
        cli.run_algorithm(args.algorithm)
    else:
        print("Please specify --algorithm, --compare, or --demo-dynamic")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        if e.code != 0:
            print("\nFalling back to demo mode...")
            run_demo()
