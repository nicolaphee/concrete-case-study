import matplotlib.pyplot as plt
import networkx as nx
import os

# Directory di salvataggio immagini
img_dir = "01_domain_knowledge_plots/"
os.makedirs(img_dir, exist_ok=True)

# Variabili con effetto semplificato (solo ↑ / ↓ / ±)
variables_icons = {
    "CementComponent": "↑",
    "BlastFurnaceSlag": "↑",
    "FlyAshComponent": "↑",
    "WaterComponent": "↓",
    "SuperplasticizerComponent": "↑",
    "CoarseAggregateComponent": "±",
    "FineAggregateComponent": "±",
    "AgeInDays": "↑"
}

# Nuovo grafo
G = nx.DiGraph()
G.add_node("Resistenza Calcestruzzo", style="filled", color="lightgray")

# Aggiunta archi
for var, symbol in variables_icons.items():
    G.add_node(var)
    G.add_edge(var, "Resistenza Calcestruzzo", label=symbol)

# Layout circolare con nodo centrale al centro
pos = nx.circular_layout(G)
pos["Resistenza Calcestruzzo"] = [0, 0]

# Disegno del grafo
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="lightyellow", edgecolors="black")
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", )

# Etichette sugli archi (solo simboli)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=12,
    font_weight="bold",
    bbox=dict(alpha=0.0, color="white"),
)

plt.title("Mappa concettuale: Resistenza del calcestruzzo", fontsize=14)
plt.axis("off")

# Salvataggio in PNG
plt.savefig(os.path.join(img_dir,"mappa_concettuale_calcestruzzo_icons.png"), format="png", dpi=300, bbox_inches="tight")

variables_short = {
    "CementComponent": "↑ Resistenza (troppo → ritiro)",
    "BlastFurnaceSlag": "↑ Resistenza lungo termine",
    "FlyAshComponent": "↑ Resistenza 28+ giorni",
    "WaterComponent": "Troppa acqua → ↓ Resistenza",
    "SuperplasticizerComponent": "Permette basso w/c → ↑ Resistenza",
    "CoarseAggregateComponent": "Rigidità, eccesso → ↓ Interfacce",
    "FineAggregateComponent": "↑ Densità, troppo → ↓ Resistenza",
    "AgeInDays": "Più giorni → ↑ Resistenza"
}

# Nuovo grafo
G = nx.DiGraph()
G.add_node("Resistenza Calcestruzzo", style="filled", color="lightgray")

# Aggiunta archi
for var, symbol in variables_short.items():
    G.add_node(var)
    G.add_edge(var, "Resistenza Calcestruzzo", label=symbol)

# Layout circolare con nodo centrale al centro
pos = nx.circular_layout(G)
pos["Resistenza Calcestruzzo"] = [0, 0]

# Disegno del grafo
plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="lightyellow", edgecolors="black")
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", )

# Etichette sugli archi (solo simboli)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=12,
    font_weight="bold",
    bbox=dict(alpha=0.0, color="white"),
)

plt.title("Mappa concettuale: Resistenza del calcestruzzo", fontsize=14)
plt.axis("off")

# Salvataggio in PNG
plt.savefig(os.path.join(img_dir,"mappa_concettuale_calcestruzzo.png"), format="png", dpi=300, bbox_inches="tight")