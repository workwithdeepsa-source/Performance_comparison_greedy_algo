import heapq  # Ensure heapq is imported
from collections import defaultdict
import time
import matplotlib.pyplot as plt


# UnionFind class for Kruskal's MST
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1


# Kruskal's MST
def kruskal_mst(graph, vertices):
    try:
        edges = sorted(graph, key=lambda x: x[2])  # Sort edges by weight
        uf = UnionFind(vertices)
        mst = []
        mst_cost = 0

        for u, v, weight in edges:
            if uf.find(u) != uf.find(v):
                uf.union(u, v)
                mst.append((u, v, weight))
                mst_cost += weight

        return mst, mst_cost
    except Exception as e:
        print(f"Error in Kruskal's MST: {e}")
        return None, 0


# Fractional Knapsack
def fractional_knapsack(values, weights, capacity):
    try:
        items = [(values[i] / weights[i], values[i], weights[i]) for i in range(len(values))]
        items.sort(reverse=True, key=lambda x: x[0])  # Sort by value-to-weight ratio

        total_value = 0
        for ratio, value, weight in items:
            if capacity >= weight:
                total_value += value
                capacity -= weight
            else:
                total_value += ratio * capacity
                break

        return total_value
    except Exception as e:
        print(f"Error in Fractional Knapsack: {e}")
        return 0


# Activity Selection
def activity_selection(start, finish):
    try:
        activities = sorted(zip(start, finish), key=lambda x: x[1])  # Sort by finish time
        selected_activities = [activities[0]]  # Start with the first activity
        last_finish = activities[0][1]

        for i in range(1, len(activities)):
            if activities[i][0] >= last_finish:
                selected_activities.append(activities[i])
                last_finish = activities[i][1]

        return selected_activities
    except Exception as e:
        print(f"Error in Activity Selection: {e}")
        return []


# Huffman Coding
class HuffmanNode:
    def __init__(self, freq, char=None):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(char_freq):
    try:
        heap = [HuffmanNode(freq, char) for char, freq in char_freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(left.freq + right.freq)
            merged.left = left
            merged.right = right
            heapq.heappush(heap, merged)

        return heap[0]  # Root node of the Huffman Tree
    except Exception as e:
        print(f"Error in building Huffman Tree: {e}")
        return None


def generate_huffman_codes(root, current_code, codes):
    if root is None:
        return

    if root.char is not None:
        codes[root.char] = current_code
        return

    generate_huffman_codes(root.left, current_code + "0", codes)
    generate_huffman_codes(root.right, current_code + "1", codes)


def huffman_coding(char_freq):
    try:
        root = build_huffman_tree(char_freq)
        if root is None:
            return {}
        codes = {}
        generate_huffman_codes(root, "", codes)
        return codes
    except Exception as e:
        print(f"Error in Huffman Coding: {e}")
        return {}


# Job Sequencing with Deadlines
class Job:
    def __init__(self, job_id, deadline, profit):
        self.job_id = job_id
        self.deadline = deadline
        self.profit = profit


def job_sequencing_with_deadlines(jobs, max_deadline):
    try:
        jobs.sort(key=lambda x: x.profit, reverse=True)  # Sort by profit in descending order
        result = [None] * max_deadline
        total_profit = 0

        for job in jobs:
            for slot in range(min(max_deadline, job.deadline) - 1, -1, -1):
                if result[slot] is None:
                    result[slot] = job.job_id
                    total_profit += job.profit
                    break

        return result, total_profit
    except Exception as e:
        print(f"Error in Job Sequencing with Deadlines: {e}")
        return [], 0


# Comparison Function with Plotting
def compare_all_algorithms():
    # Larger input data for more measurable execution times

    # Data for Kruskal's MST (expanded for more complexity)
    graph = [(i, j, (i + j) % 10 + 1) for i in range(100) for j in range(i + 1, 100)]
    vertices = 100

    # Data for Fractional Knapsack (expanded to larger set of items)
    values = [i * 10 for i in range(1, 101)]
    weights = [i * 2 for i in range(1, 101)]
    capacity = 5000

    # Data for Activity Selection (larger input)
    start = [i for i in range(1, 101)]
    finish = [i + 1 for i in range(1, 101)]

    # Data for Huffman Coding (expanded frequencies)
    char_freq = {chr(97 + i): (i + 1) * 5 for i in range(26)}

    # Data for Job Sequencing with Deadlines (expanded set)
    jobs = [Job(chr(97 + i), i % 10, (i + 1) * 10) for i in range(100)]
    max_deadline = 10

    # Dictionary to store execution times
    times = {}

    # Kruskal's MST
    start_time = time.perf_counter()
    mst, mst_cost = kruskal_mst(graph, vertices)
    times['Kruskal'] = time.perf_counter() - start_time

    # Fractional Knapsack
    start_time = time.perf_counter()
    knapsack_value = fractional_knapsack(values, weights, capacity)
    times['Knapsack'] = time.perf_counter() - start_time

    # Activity Selection
    start_time = time.perf_counter()
    selected_activities = activity_selection(start, finish)
    times['Activity Selection'] = time.perf_counter() - start_time

    # Huffman Coding
    start_time = time.perf_counter()
    huffman_codes = huffman_coding(char_freq)
    times['Huffman Coding'] = time.perf_counter() - start_time

    # Job Sequencing with Deadlines
    start_time = time.perf_counter()
    scheduled_jobs, total_profit = job_sequencing_with_deadlines(jobs, max_deadline)
    times['Job Sequencing'] = time.perf_counter() - start_time

    # Print the execution times in the terminal
    for algorithm, exec_time in times.items():
        print(f"{algorithm}: {exec_time:.6f} seconds")

    # Plotting Execution Times as a Line Plot
    algorithms = list(times.keys())
    execution_times = list(times.values())
    plt.figure(figsize=(10, 6))
    plt.plot(algorithms, execution_times, marker='o', color='skyblue', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Algorithms')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison of Greedy Algorithms')
    plt.grid(True)
    plt.show()


# Execute the comparison with line plot
compare_all_algorithms()