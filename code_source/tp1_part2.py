
import numpy as np

#################### Implimentation du graphe ###############
graph = {
    'S': {'A': 1, 'G': 12},
    'A': {'B': 3, 'C': 1},
    'B': {'D': 3},
    'C': {'D': 1, 'G': 2},
    'D': {'G': 3},
    'G': {}
}

################### La matrice d'adjacence ##################


nodes = ['S', 'A', 'B', 'C', 'D', 'G']
size = len(nodes)
adj_matrix = np.full((size, size), float('0'))  # Initialise avec un coût infini

# Remplissage de la matrice d'adjacence
for i, node in enumerate(nodes):
    for neighbor, cost in graph.get(node, {}).items():
        j = nodes.index(neighbor)
        adj_matrix[i][j] = cost

print("Matrice d'adjacence:")
print(adj_matrix)

########################### Implimentation des algorithmes d'exploration ###################
########################### a) Exploration en graphe en largeur d’abord (BFS) ########################
from collections import deque

def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = set()

    while queue:
        path = queue.popleft()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor in sorted(graph[node]):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return None

print("Chemin BFS:", bfs(graph, 'S', 'G'))



############################## b) Exploration en graphe à coût uniforme (UCS) ##########################
import heapq

def ucs(graph, start, goal):
    queue = [(0, start, [])]
    visited = set()

    while queue:
        (cost, node, path) = heapq.heappop(queue)

        if node in visited:
            continue
        visited.add(node)

        path = path + [node]

        if node == goal:
            return path, cost

        for neighbor, edge_cost in sorted(graph[node].items()):
            if neighbor not in visited:
                heapq.heappush(queue, (cost + edge_cost, neighbor, path))

    return None

ucs_path, ucs_cost = ucs(graph, 'S', 'G')
print("Chemin UCS:", ucs_path, "Coût:", ucs_cost)



############################ c) Exploration en graphe en profondeur d’abord (DFS) ########################################
def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        (node, path) = stack.pop()

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            for neighbor in sorted(graph[node], reverse=True):
                stack.append((neighbor, path + [neighbor]))

    return None

print("Chemin DFS:", dfs(graph, 'S', 'G'))

#################### d) Exploration en graphe A* avec une heuristique cohérente ############### 
heuristic = {
    'S': 7, 'A': 6, 'B': 2, 'C': 1, 'D': 1, 'G': 0
}

def a_star(graph, start, goal, heuristic):
    queue = [(heuristic[start], 0, start, [])]
    visited = set()

    while queue:
        (f_cost, cost, node, path) = heapq.heappop(queue)

        if node in visited:
            continue
        visited.add(node)

        path = path + [node]

        if node == goal:
            return path, cost

        for neighbor, edge_cost in sorted(graph[node].items()):
            if neighbor not in visited:
                g_cost = cost + edge_cost
                f_cost = g_cost + heuristic[neighbor]
                heapq.heappush(queue, (f_cost, g_cost, neighbor, path))

    return None

a_star_path, a_star_cost = a_star(graph, 'S', 'G', heuristic)
print("Chemin A*:", a_star_path, "Coût:", a_star_cost)

######################## Afichage des chemins ##############
import networkx as nx
import matplotlib.pyplot as plt

# Création du graphe
G = nx.DiGraph()
for node, neighbors in graph.items():
    for neighbor, cost in neighbors.items():
        G.add_edge(node, neighbor, weight=cost)

# Positionnement des nœuds
pos = nx.spring_layout(G)

# Couleurs des chemins
path_colors = {
    'BFS': ('blue', bfs(graph, 'S', 'G')),
    'UCS': ('green', ucs_path),
    'DFS': ('orange', dfs(graph, 'S', 'G')),
    'A*': ('red', a_star_path)
}

# Dessiner le graphe de base
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10, font_weight="bold")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G[u][v]['weight'] for u, v in G.edges})

# Dessiner les chemins de chaque méthode en couleurs différentes
for method, (color, path) in path_colors.items():
    if path:
        edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color=color, width=2, label=method)

plt.legend(handles=[plt.Line2D([0], [0], color=color, lw=4) for color, _ in path_colors.values()],
           labels=path_colors.keys(), loc='upper left')
plt.title("Différents chemins d'exploration dans le graphe")
plt.show()
