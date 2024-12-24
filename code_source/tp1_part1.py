import numpy as np
import heapq
import networkx as nx
import matplotlib.pyplot as plt

################################## PARTIE 1 #############################################
graph = {
    'S': {'A': 1, 'B': 2},
    'A': {'C': 1},
    'B': {'C': 1},
    'C': {'G': 2},
}

heuristic = {
    'S': 3,
    'A': 3,
    'B': 1,
    'C': 0,
    'G': 0,
}
############################La matrice d'adjacence#######################


# Liste des sommets dans l'ordre S, A, B, C, G
nodes = ['S', 'A', 'B', 'C', 'G']
n = len(nodes)
adjacency_matrix = np.full((n, n), float('0'))  # Initialise avec 'inf' pour les non-arcs

# Remplir la matrice d'adjacence en fonction du graphe
for i, node in enumerate(nodes):
    for neighbor, cost in graph.get(node, {}).items():
        j = nodes.index(neighbor)
        adjacency_matrix[i][j] = cost

print("Matrice d'adjacence:")
print(adjacency_matrix)

################### Application de l'algorithme A #########################



def a_star(graph, heuristic, start, goal):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic[start], 0, start, []))  # (f_score, g_score, current_node, path)
    visited = set()

    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        # Ajoute le noeud actuel au chemin
        path = path + [current]

        # Vérifie si nous avons atteint l'état but
        if current == goal:
            return path, g_score

        # Explore les voisins
        for neighbor, cost in graph.get(current, {}).items():
            if neighbor not in visited:
                new_g_score = g_score + cost
                new_f_score = new_g_score + heuristic[neighbor]
                heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, path))

    return None, float('inf')  # Retourne un coût infini si aucun chemin n'est trouvé

# Exécuter A* du sommet initial S vers le sommet but G
path, cost = a_star(graph, heuristic, 'S', 'G')
print("Ordre de développement des états et chemin trouvé avec A* :", path)
print("Coût total du chemin :", cost)


####################### Le graph #####################



# Créer un graphe orienté avec NetworkX
G = nx.DiGraph()

# Ajouter les arcs et les coûts
for node, neighbors in graph.items():
    for neighbor, cost in neighbors.items():
        G.add_edge(node, neighbor, weight=cost)

# Extraire le chemin trouvé
edges_in_path = [(path[i], path[i+1]) for i in range(len(path) - 1)]

# Définir les positions des noeuds
pos = nx.spring_layout(G)

# Dessiner les noeuds et les arêtes
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10, font_weight="bold")
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G[u][v]['weight'] for u, v in G.edges}, font_color="black")

# Dessiner le chemin en rouge
nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color="red", width=2)

plt.title("Chemin trouvé avec A* en rouge")
plt.show()
