import networkx as nx

G = nx.Graph()
G.add_nodes_from(range(5))
G.add_edge(0, 1, w = 1)

print(G)