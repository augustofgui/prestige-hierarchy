import networkx as nx
import numpy as np
import pandas as pd
import os
os.environ['MPLCONFIGDIR'] = "/scisci/prestige-hierarchy"
import matplotlib.pyplot as plt
from SpringRank import SpringRank
import tools as tl
import random
from tqdm import tqdm
from scipy.stats import ttest_ind

import seaborn as sns
sns.set_theme(style="white")

def generate_random_network(din, dout):
    rewired_G = nx.directed_configuration_model(din, dout, seed=42)

    return rewired_G

def calculate_rank(G):
    alpha=0.
    l0=1.
    l1=1.

    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_matrix(G, dtype=float,nodelist=nodes)
    rank = SpringRank(A, alpha=alpha, l0=l0, l1=l1)
    rank = tl.shift_rank(rank)
    rank = {nodes[i]:rank[i] for i in range(G.number_of_nodes())}
    return rank

def calculate_hierarchy_strength(G):
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return 0
    rank = calculate_rank(G)
    hierarchy_edges = 0
    total = 0
    for u, v, d in G.edges(data=True):
        total += d['weight']
        if rank[u] < rank[v]:
            hierarchy_edges += d['weight']
    return hierarchy_edges / total

def calculate_p_value_two_sided(observed_strength, random_strengths):
    count = sum(1 for s in random_strengths if s <= observed_strength)
    return count / len(random_strengths)

def plot_monte_carlo_results(original_strength, rewired_strengths, plot):
    sns.histplot(rewired_strengths, bins=30, kde=False, color='blue', alpha=0.7, label='Rewired Networks', ax=plot)
    
    plot.axvline(x=original_strength, color='r', linestyle='-', linewidth=2, 
                label=f'Original Network: {original_strength:.4f}')
    
  
    p_value = calculate_p_value_two_sided(original_strength, rewired_strengths)
    
    plot.xlabel('Hierarchy Strength')
    plot.ylabel('Frequency')
    plot.legend()
    plot.grid(True)
    
    return p_value

def monte_carlo_configuration_model(G, num_iterations=1000):
    din = list(d for _, d in G.in_degree())
    dout = list(d for _, d in G.out_degree())

    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    random.shuffle(weights)
    original_rank = calculate_hierarchy_strength(G)
    null_ranks = []
    for _ in tqdm(range(num_iterations)):
        try:
            G_null = nx.directed_configuration_model(din, dout)
            for i, (_, _, d) in zip(range(0, G_null.number_of_edges()), G_null.edges(data=True)):
                d['weight'] = weights[i]
            random.shuffle(weights)
            mapping = {i: node for i, node in enumerate(G.nodes())}
            G_null = nx.relabel_nodes(G_null, mapping)
            null_rank = calculate_hierarchy_strength(G_null)
            null_ranks.append(null_rank)
        except:
            continue
    
    return original_rank, null_ranks
   

np.random.seed(42)
G = tl.build_graph_from_adjacency("data/BRCS_adjacency.dat")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

original_rank, null_ranks = monte_carlo_configuration_model(G, 10)
p_value = plot_monte_carlo_results(original_rank, null_ranks, ax[0])
print(f"Original hierarchy strength: {original_rank:.4f}")
print(f"Mean rewired hierarchy strength: {np.mean(null_ranks):.4f}")
print(f"p-value: {p_value:.10f}")

plt.tight_layout()
plt.show()
plt.savefig("figures/montecarlo.pdf")

