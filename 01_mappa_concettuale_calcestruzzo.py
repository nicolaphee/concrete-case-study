import matplotlib.pyplot as plt
import networkx as nx
import os
from utils.functions import save_plot

# Directory di salvataggio immagini
img_dir = "01_domain_knowledge_plots/"
os.makedirs(img_dir, exist_ok=True)

variables_short = {
    "CementComponent": "↑ Resistenza \n (troppo → ritiro)",
    "BlastFurnaceSlag": "↑ Resistenza \n lungo termine",
    "FlyAshComponent": "↑ Resistenza \n 28+ giorni",
    "WaterComponent": "Troppa acqua → \n ↓ Resistenza",
    "SuperplasticizerComponent": "Permette basso w/c → \n ↑ Resistenza",
    "CoarseAggregateComponent": "Rigidità, eccesso → \n ↓ Interfacce",
    "FineAggregateComponent": "↑ Densità, troppo → \n ↓ Resistenza",
    "AgeInDays": "Più giorni → \n ↑ Resistenza"
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
save_plot("mappa_concettuale_calcestruzzo.png", img_dir=img_dir)




# ================================
# Nuova mappa concettuale: Variabili ingegnerizzate
# ================================
variables_engineered = {
    "W/C": "↓ se alto \n → porosità ↑",
    "Binder": "↑ reazioni \n → Strength ↑",
    "SCM%": "Ottimale 20–40% \n → Strength ↑",
    "Aggregates/Paste": "Troppo alto → poca pasta \n → Strength ↓",
    "Superplasticizer/Binder": "Ottimale → meno acqua \n → Strength ↑",
    "Age (cat)": "Crescita rapida 28d \n → poi saturazione"
}

# Nuovo grafo per variabili ingegnerizzate
G2 = nx.DiGraph()
G2.add_node("Resistenza Calcestruzzo", style="filled", color="lightgray")

for var, desc in variables_engineered.items():
    G2.add_node(var)
    G2.add_edge(var, "Resistenza Calcestruzzo", label=desc)

pos2 = nx.circular_layout(G2)
pos2["Resistenza Calcestruzzo"] = [0, 0]

plt.figure(figsize=(12, 8))
nx.draw_networkx_nodes(G2, pos2, node_size=2500, node_color="lightyellow", edgecolors="black")
nx.draw_networkx_labels(G2, pos2, font_size=9, font_weight="bold")
nx.draw_networkx_edges(G2, pos2, arrowstyle="->", arrowsize=5, edge_color="gray")

edge_labels2 = nx.get_edge_attributes(G2, 'label')
nx.draw_networkx_edge_labels(
    G2, pos2,
    edge_labels=edge_labels2,
    font_size=10,
    font_weight="bold",
    bbox=dict(alpha=0.0, color="white"),
)

plt.title("Mappa concettuale: Variabili ingegnerizzate", fontsize=14)
plt.axis("off")
plt.tight_layout()
save_plot("mappa_concettuale_calcestruzzo_engineered.png", img_dir=img_dir)