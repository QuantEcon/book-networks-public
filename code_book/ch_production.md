---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Chapter 2 - Production Code

We begin with some imports

```{code-cell}
import quantecon as qe
import quantecon_book_networks.input_output as qbn_io
import quantecon_book_networks.data as qbn_data
ch2_data = qbn_data.production()
```

```{code-cell}
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as plc
from matplotlib import cm
```

## Multisector Models

### Backward linkages for 15 US sectors in 2019

We start by loading a graph of linkages between 15 US sectors in 2019. Our graph comes as an adjacency matrix and list of the associated industry codes. The A\[i,j\] weight is the sales from industry i to industry j as a fraction of total sales industry j.

```{code-cell}
codes = ch2_data["us_sectors_15"]["codes"]
A = ch2_data["us_sectors_15"]["adjacency_matrix"]
X = ch2_data["us_sectors_15"]["total_industry_sales"]
```

Here we will use the quantecon_book_networks package to convert the adjacency matrix into a networkx graph object.

```{code-cell}
G = qbn_io.adjacency_matrix_to_graph(A, codes, node_weights=X, tol=0.0)
```

Next we calculate our graph’s properties. We use hub-based eigenvector centrality as our centrality measure for this plot.

```{code-cell}
centrality = qbn_io.eigenvector_centrality(A)
edge_weights = qbn_io.edge_weights(G)
```

Now we convert our graph features to plot features.

```{code-cell}
node_pos_dict = nx.circular_layout(G)

node_sizes = X * 0.0005
edge_widths = qbn_io.normalise_weights(edge_weights,10)

node_colors = qbn_io.colorise_weights(centrality,beta=False)
node_to_color = dict(zip(G.nodes,node_colors))
edge_colors = []
for src,_ in G.edges:
    edge_colors.append(node_to_color[src])
```

Finally we produce the plot.

```{code-cell}

fig, ax = plt.subplots(figsize=(8, 10))
plt.axis("off")

nx.draw_networkx_nodes(G, 
                        node_pos_dict, 
                        node_color=node_colors, 
                        node_size=node_sizes, 
                        edgecolors='grey', 
                        linewidths=2, 
                        alpha=0.6, 
                        ax=ax)

nx.draw_networkx_labels(G, 
                        node_pos_dict, 
                        font_size=10, 
                        ax=ax)

nx.draw_networkx_edges(G, 
                        node_pos_dict, 
                        edge_color=edge_colors, 
                        width=edge_widths, 
                        arrows=True, 
                        arrowsize=20, 
                        alpha=0.6,  
                        ax=ax, 
                        arrowstyle='->', 
                        node_size=node_sizes, 
                        connectionstyle='arc3,rad=0.15')

plt.show()
```

### Network for 71 US sectors in 2019

We start by loading a graph of linkages between 75 US sectors in 2019.

```{code-cell}
codes_71 = ch2_data['us_sectors_71']['codes']
A_71 = ch2_data['us_sectors_71']['adjacency_matrix']
X_71 = ch2_data['us_sectors_71']['total_industry_sales']
```

We will again use the quantecon_book_networks package to convert the adjacency matrix into a networkx graph object.

```{code-cell}
G_71 = qbn_io.adjacency_matrix_to_graph(A_71, codes_71, node_weights=X_71, tol=0.01)
```

Next we calculate our graph’s properties. We use hub-based eigenvector centrality as our centrality measure for this plot.

```{code-cell}
centrality_71 = qbn_io.eigenvector_centrality(A_71)
edge_weights_71 = qbn_io.edge_weights(G_71)
```

Now we convert our graph features to plot features.

```{code-cell}
node_pos_dict = nx.shell_layout(G_71)

node_sizes = X_71 * 0.0005
edge_widths = qbn_io.normalise_weights(edge_weights_71,4)

node_colors = qbn_io.colorise_weights(centrality_71,beta=False)
node_to_color = dict(zip(G_71.nodes,node_colors))
edge_colors = []
for src,_ in G_71.edges:
    edge_colors.append(node_to_color[src])
```

Finally we produce the plot.

```{code-cell}
fig, ax = plt.subplots(figsize=(10, 12))
plt.axis("off")

nx.draw_networkx_nodes(G_71, 
                        node_pos_dict, 
                        node_color=node_colors, 
                        node_size=node_sizes, 
                        edgecolors='grey', 
                        linewidths=2, 
                        alpha=0.6, 
                        ax=ax)

nx.draw_networkx_labels(G_71, 
                        node_pos_dict, 
                        font_size=10, 
                        ax=ax)

nx.draw_networkx_edges(G_71, 
                        node_pos_dict, 
                        edge_color=edge_colors, 
                        width=edge_widths, 
                        arrows=True, 
                        arrowsize=20, 
                        alpha=0.6,  
                        ax=ax, 
                        arrowstyle='->', 
                        node_size=node_sizes, 
                        connectionstyle='arc3,rad=0.15')

plt.show()
```


### The Leontief inverse 𝐿 (hot colors are larger values)

We construct the Leontief inverse matrix, from 15 sector adjacency matrix.

```{code-cell}
I = np.identity(len(A))
L = np.linalg.inv(I - A)
```

Now we produce the plot.

```{code-cell}
fig, ax = plt.subplots(figsize=(6.5, 5.5))

ticks = range(len(L))

levels = np.sqrt(np.linspace(0, 0.75, 100))

co = ax.contourf(ticks, 
                    ticks,
                    L,
                    levels,
                    alpha=0.85, cmap=cm.plasma)

ax.set_xlabel('sector $j$', fontsize=12)
ax.set_ylabel('sector $i$', fontsize=12)
ax.set_yticks(ticks)
ax.set_yticklabels(codes)
ax.set_xticks(ticks)
ax.set_xticklabels(codes)

plt.show()
```


### Propagation of demand shocks via backward linkages

```{code-cell}

sim_length = 6
N = len(A)
d = np.random.rand(N) # np.zeros(N)
d[6] = 1  # positive shock to agriculture
x = d
x_vecs = []
for i in range(sim_length):
    x_vecs.append(x)
    x = A @ x

```

```{code-cell}

fig, axes = plt.subplots(3, 2, figsize=(8, 10))
axes = axes.flatten()

for ax, x_vec, i in zip(axes, x_vecs, range(sim_length)):
    ax.set_title(f"round {i}")
    x_vec_cols = cm.plasma(to_zero_one(x_vec))
    plot_graph(A, X, ax, codes,
                  layout_type='spring', # alternative layouts: spring, circular, random, spiral
                  layout_seed=342156,
                  node_color_list=x_vec_cols,
                  node_size_multiple=0.00028,
                  edge_size_multiple=0.8)

plt.tight_layout()
plt.show()

```

### Eigenvector centrality of across US industrial sectors

```{code-cell}
ecentral = eigenvector_centrality(A)
ecentral_color_list = cm.plasma(to_zero_one(ecentral))
```

```{code-cell}
fig, ax = plt.subplots()
ax.bar(codes, ecentral, color=ecentral_color_list, alpha=0.6)

ax.set_ylabel("eigenvector centrality", fontsize=12)

plt.show()
```


### Output multipliers across 15 US industrial sectors

```{code-cell}
omult = katz_centrality(A, authority=True)
omult_color_list = cm.plasma(to_zero_one(omult))

```

```{code-cell}
fig, ax = plt.subplots()
ax.bar(codes, omult, color=omult_color_list, alpha=0.6)

ax.set_ylabel("Output multipliers", fontsize=12)

plt.show()
```


### Forward linkages and upstreamness over US industrial sectors

```{code-cell}
upstreamness = katz_centrality(F)
upstreamness_color_list = cm.plasma(to_zero_one(upstreamness))

```

```{code-cell}

fig, ax = plt.subplots(figsize=(8, 10))
plt.axis("off")

plot_graph(F, X, ax, codes, 
              layout_type='spring', # alternative layouts: spring, circular, random, spiral
              layout_seed=5432167,
              tol=0.0,
              node_color_list=upstreamness_color_list) 

plt.show()

```

### Relative upstreamness of US industrial sectors

```{code-cell}
fig, ax = plt.subplots()
ax.bar(codes, upstreamness, color=upstreamness_color_list, alpha=0.6)

ax.set_ylabel("upstreamness", fontsize=12)

plt.show()
```

## General Equilibrium

### GDP growth rates and std. deviations (in parentheses) for 8 countries

**No code in repo**


### Hub-based Katz centrality of across 15 US industrial sectors

```{code-cell}

kcentral = katz_centrality(A)
kcentral_color_list = cm.plasma(to_zero_one(kcentral))
```

```{code-cell}

fig, ax = plt.subplots()
ax.bar(codes, kcentral, color=kcentral_color_list, alpha=0.6)
ax.set_ylabel("Katz hub centrality", fontsize=12)

plt.show()

```