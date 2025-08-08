from multiprocessing import Manager, Process, Lock, Queue
import time
import random

def worker(graph, task_queue, shared_distances, lock):
    while True:
        try:
            current_distance, current_node = task_queue.get(timeout=0.1)
        except:
            return  # No more work

        if current_distance > shared_distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            with lock:
                if distance < shared_distances[neighbor]:
                    shared_distances[neighbor] = distance
                    task_queue.put((distance, neighbor))

def parallel_dijkstra(graph, start_node, num_processes):
    start_time = time.perf_counter()

    manager = Manager()
    lock = manager.Lock()
    shared_distances = manager.dict({node: float('inf') for node in graph})
    shared_distances[start_node] = 0
    task_queue = manager.Queue()
    task_queue.put((0, start_node))

    processes = []
    for _ in range(num_processes):
        p = Process(target=worker, args=(graph, task_queue, shared_distances, lock))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    elapsed = time.perf_counter() - start_time
    return dict(shared_distances), elapsed

def generate_random_graph(num_nodes, avg_degree=4, max_weight=10):
    """
    Generate a connected undirected weighted graph.
    avg_degree ~ how many edges per node (on average)
    """
    nodes = [str(i) for i in range(num_nodes)]
    graph = {node: {} for node in nodes}

    # Ensure connectedness by making a random spanning tree first
    for i in range(1, num_nodes):
        neighbor = str(random.randint(0, i - 1))
        weight = random.randint(1, max_weight)
        graph[str(i)][neighbor] = weight
        graph[neighbor][str(i)] = weight

    # Add extra random edges
    num_extra_edges = int(num_nodes * avg_degree / 2) - (num_nodes - 1)
    for _ in range(num_extra_edges):
        a, b = random.sample(nodes, 2)
        if b not in graph[a]:  # no duplicate edges
            weight = random.randint(1, max_weight)
            graph[a][b] = weight
            graph[b][a] = weight

    return graph

if __name__ == '__main__':
    # Example graph
    graph2 = {
        'A': {'B': 1, 'P': 4, 'V': 7},
        'B': {'A': 1, 'R': 4, 'C': 3, 'E': 7, 'L': 9, 'D': 8, 'Q': 5},
        'C': {'B': 3, 'F': 4, 'K': 3},
        'D': {'B': 8, 'M': 6},
        'E': {'B': 7, 'L': 3, 'H': 3},
        'F': {'C': 4, 'J': 7, 'G': 9},
        'G': {'F': 9, 'J': 5, 'H': 4, 'I': 3},
        'H': {'E': 3, 'G': 5, 'O': 6},
        'I': {'G': 3, 'P': 5},
        'J': {'G': 5, 'T': 1, 'U': 8, 'F': 7},
        'K': {'C': 3, 'S': 2},
        'L': {'E': 3, 'B': 9, 'M': 5, 'N': 2},
        'M': {'D': 6, 'L': 5},
        'N': {'L': 2},
        'O': {'H': 6},
        'P': {'I': 5, 'U': 6, 'A': 4},
        'Q': {'B': 5},
        'R': {'B': 4},
        'S': {'K': 2, 'T': 1},
        'T': {'S': 1, 'J': 1, 'U': 11},
        'U': {'T': 11, 'J': 8, 'P': 6},
        'V': {'A': 7}
    }

    """
    distances, elapsed = parallel_dijkstra(graph2, 'A', 4)
    print("Distances:", distances)
    print("Time:", elapsed)
    """

    # Parallel is slower than sequential...waiting for Professor's input 
    
    graph = generate_random_graph(50000, avg_degree=6)
    start_node = "0"

    distances, elapsed = parallel_dijkstra(graph, '0', 4)
    #print("Distances:", distances)
    print("Time:", elapsed)
