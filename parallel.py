import threading
import queue
import time
import random

import threading
import queue

def parallel(graph, start_node, num_threads=6):
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0

    pq = queue.PriorityQueue()
    pq.put((0, start_node))

    node_locks = {node: threading.Lock() for node in graph}

    def worker():
        while True:
            dist_u, u = pq.get()
            if u is None:  # sentinel to stop
                pq.task_done()
                break

            if dist_u > distances[u]:
                pq.task_done()
                continue

            for v, w in graph[u].items():
                new_dist = dist_u + w
                if new_dist < distances[v]:
                    with node_locks[v]:
                        if new_dist < distances[v]:
                            distances[v] = new_dist
                            pq.put((new_dist, v))

            pq.task_done()

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    pq.join()

    # Send sentinel None tasks to stop workers
    for _ in range(num_threads):
        pq.put((float('inf'), None))

    for t in threads:
        t.join()

    return distances

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

if __name__ == "__main__":
    graph = generate_random_graph(100000, avg_degree=6)
    start_node = "0"
    start_time = time.perf_counter()
    shortest_paths = parallel(graph, start_node)
    end_time = time.perf_counter()
    #print(f"Shortest paths from {start_node}: {shortest_paths}") 
    print("Compute time:", end_time - start_time)
