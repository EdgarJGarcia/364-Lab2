import multiprocessing as mp
import heapq
import time
import random
import queue  

def local_dijkstra(subgraph, start_nodes, dist, inbound_queue, outbound_queues, node_to_part, part_id):
    heap = []
    # Initialize heap with known start distances in this partition
    for node in start_nodes:
        heapq.heappush(heap, (dist[node], node))

    while True:
        # Process any boundary updates from other partitions safely
        while True:
            try:
                node, new_dist = inbound_queue.get_nowait()
            except queue.Empty:
                break
            if new_dist < dist[node]:
                dist[node] = new_dist
                heapq.heappush(heap, (new_dist, node))

        if not heap:
            break  # No local work

        cur_dist, cur_node = heapq.heappop(heap)
        if cur_dist > dist[cur_node]:
            continue  # Outdated

        for nbr, w in subgraph[cur_node].items():
            new_dist = cur_dist + w
            if new_dist < dist.get(nbr, float('inf')):
                dist[nbr] = new_dist
                if node_to_part[nbr] == part_id:
                    # neighbor inside partition, push locally
                    heapq.heappush(heap, (new_dist, nbr))
                else:
                    # neighbor in another partition: send update
                    outbound_queues[node_to_part[nbr]].put((nbr, new_dist))

def partition_graph(graph, num_parts):
    nodes = list(graph.keys())
    size = len(nodes) // num_parts
    partitions = []
    for i in range(num_parts):
        part_nodes = nodes[i*size:(i+1)*size] if i < num_parts-1 else nodes[i*size:]
        subgraph = {n: graph[n] for n in part_nodes}
        partitions.append(subgraph)
    return partitions

def parallel_partitioned_dijkstra(graph, start_node, num_parts=4, max_iters=10):
    start_time = time.perf_counter()
    # Partition graph
    partitions = partition_graph(graph, num_parts)

    # Map node -> partition
    node_to_part = {}
    for i, part in enumerate(partitions):
        for node in part:
            node_to_part[node] = i

    # Shared distances dicts per partition (use Manager)
    manager = mp.Manager()
    dists = [manager.dict({node: float('inf') for node in part}) for part in partitions]
    # Set start distance
    dists[node_to_part[start_node]][start_node] = 0

    # Queues for boundary updates between partitions
    queues = [manager.Queue() for _ in range(num_parts)]

    processes = []
    for i in range(num_parts):
        # Start nodes for local processing = nodes with dist < inf in partition
        start_nodes = [node for node, dist in dists[i].items() if dist < float('inf')]
        p = mp.Process(target=local_dijkstra,
                       args=(partitions[i], start_nodes, dists[i], queues[i], queues, node_to_part, i))
        processes.append(p)
        p.start()

    for _ in range(max_iters):
        # Wait for all to finish local Dijkstra round
        for p in processes:
            p.join(timeout=1)

        # Check if any queues have messages safely
        has_updates = False
        for q in queues:
            try:
                q.get_nowait()
                has_updates = True
                break
            except queue.Empty:
                continue

        if not has_updates:
            break  # converged, no more updates

        # Restart processes for next iteration if needed
        processes = []
        for i in range(num_parts):
            start_nodes = []
            while True:
                try:
                    node, dist_val = queues[i].get_nowait()
                except queue.Empty:
                    break
                if dist_val < dists[i].get(node, float('inf')):
                    dists[i][node] = dist_val
                    start_nodes.append(node)
            p = mp.Process(target=local_dijkstra,
                           args=(partitions[i], start_nodes, dists[i], queues[i], queues, node_to_part, i))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    # Merge distances from all partitions
    final_dist = {}
    for dist_dict in dists:
        final_dist.update(dict(dist_dict))

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return final_dist, elapsed_time

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
    # Doesn't scale well, max around 11,500
    graph = generate_random_graph(10000, avg_degree=6)
    start_node = "0"

    shortest_paths, compute_time = parallel_partitioned_dijkstra(graph, start_node, num_parts=2, max_iters=5)
    #print("Shortest distances:", shortest_paths)
    print("Time:", compute_time)
