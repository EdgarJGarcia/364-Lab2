import heapq

def sequential(graph, start_node):
    # Dijkstra's algorithm
    distances = {node: float('inf') for node in graph} 
    distances[start_node] = 0  

    priority_queue = [(0, start_node)]  
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances 

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
} 

start_node = 'A'
shortest_paths = sequential(graph, start_node)
print(f"Shortest paths from {start_node}: {shortest_paths}") 
