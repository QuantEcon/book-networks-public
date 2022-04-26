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

# quantecon_book_networks

## input_output

### node_total_exports
```{code-cell}
def node_total_exports(G):
    node_exports = []
    for node1 in G.nodes():
        total_export = 0
        for node2 in G[node1]:
            total_export += G[node1][node2]['weight']
        node_exports.append(total_export)
    return node_exports
```

### node_total_imports
```{code-cell}
def node_total_imports(G):
    node_imports = []
    for node1 in G.nodes():
        total_import = 0
        for node2 in G[node1]:
            total_import += G[node2][node1]['weight']
        node_imports.append(total_import)
    return node_imports
```

### edge_weights
```{code-cell}
def edge_weights(G):
    edge_weights = [G[u][v]['weight'] for u,v in G.edges()]
    return edge_weights
```

### normalise_weights
```{code-cell}
def normalise_weights(weights,scalar=1):
    max_value = np.max(weights)
    return [scalar * (weight / max_value) for weight in weights]
```

### to_zero_one
```{code-cell}
def to_zero_one(x):
    "Map vector x to the zero one interval."
    x = np.array(x)
    x_min, x_max = x.min(), x.max()
    return (x - x_min)/(x_max - x_min)
```

### to_zero_one_beta
```{code-cell}
def to_zero_one_beta(x, 
                qrange=[0.25, 0.75], 
                beta_para=[0.5, 0.5]):
    
    """
    Nonlinearly map vector x to the zero one interval with beta distribution.
    https://en.wikipedia.org/wiki/Beta_distribution
    """
    x = np.array(x)
    x_min, x_max = x.min(), x.max()
    if beta_para != None:
        a, b = beta_para
        return beta.cdf((x - x_min) /(x_max - x_min), a, b)
    else:
        q1, q2 = qrange
        return (x - x_min) * (q2 - q1) /(x_max - x_min) + q1
```

### colorise_weights
```{code-cell}
import matplotlib.cm as cm
def colorise_weights(weights,beta=True,color_palette=cm.plasma):
    if beta:
        cp = color_palette(to_zero_one_beta(weights))
    else:
        cp = color_palette(to_zero_one(weights))
    return cp 
```

### spec_rad
```{code-cell}
def spec_rad(M):
    """
    Compute the spectral radius of M.
    """
    return np.max(np.abs(np.linalg.eigvals(M)))
```

### adjacency_matrix_to_graph
```{code-cell}
def adjacency_matrix_to_graph(A, 
               codes,
               tol=0.0):  # clip entries below tol
    """
    Build a networkx graph object given an adjacency matrix
    """
    G = nx.DiGraph()
    N = len(A)

    # Add nodes
    for i, code in enumerate(codes):
        G.add_node(code, name=code)

    # Add the edges
    for i in range(N):
        for j in range(N):
            a = A[i, j]
            if a > tol:
                G.add_edge(codes[i], codes[j], weight=a)

    return G
```

### eigenvector_centrality
```{code-cell}
def eigenvector_centrality(A, k=40, authority=False):
    """
    Computes the dominant eigenvector of A. Assumes A is 
    primitive and uses the power method.  
    """
    A_temp = A.T if authority else A
    n = len(A_temp)
    r = spec_rad(A_temp)
    e = r**(-k) * (np.linalg.matrix_power(A_temp, k) @ np.ones(n))
    return e / np.sum(e)
```

### katz_centrality
```{code-cell}
def katz_centrality(A, b=1, authority=False):
    """
    Computes the Katz centrality of A, defined as the x solving

    x = 1 + b A x    (1 = vector of ones)

    Assumes that A is square.

    If authority=True, then A is replaced by its transpose.
    """
    n = len(A)
    I = np.identity(n)
    C = I - b * A.T if authority else I - b * A
    return np.linalg.solve(C, np.ones(n))
```

### build_unweighted_matrix
```{code-cell}
def build_unweighted_matrix(Z, tol=1e-5):
    """
    return a unweighted adjacency matrix
    """
    return 1*(Z>tol)
```

### erdos_renyi_graph
```{code-cell}
def erdos_renyi_graph(n=100, p=0.5, seed=1234):
    "Returns an Erdős-Rényi random graph."
    
    np.random.seed(seed)
    edges = itertools.combinations(range(n), 2)
    G = nx.Graph()
    
    for e in edges:
        if np.random.rand() < p:
            G.add_edge(*e)
    return G
```

### build_coefficient_matrices
```{code-cell}
def build_coefficient_matrices(Z, X):
    """
    Build coefficient matrices A and F from Z and X via 
    
        A[i, j] = Z[i, j] / X[j] 
        F[i, j] = Z[i, j] / X[i]
    
    """
    A, F = np.empty_like(Z), np.empty_like(Z)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            A[i, j] = Z[i, j] / X[j]
            F[i, j] = Z[i, j] / X[i]

    return A, F
```

## plotting

### plot_graph
```{code-cell}
def plot_graph(A, 
               X,
               ax,
               codes,
               node_color_list=None,
               node_size_multiple=0.0005, 
               edge_size_multiple=14,
               layout_type='circular',
               layout_seed=1234,
               tol=0.03):  # clip entries below tol

    G = nx.DiGraph()
    N = len(A)

    # Add nodes, with weights by sales of the sector
    for i, w in enumerate(X):
        G.add_node(codes[i], weight=w, name=codes[i])

    node_sizes = X * node_size_multiple

    # Position the nodes
    if layout_type == 'circular':
        node_pos_dict = nx.circular_layout(G)
    elif layout_type == 'spring':
        node_pos_dict = nx.spring_layout(G, seed=layout_seed)
    elif layout_type == 'random':
        node_pos_dict = nx.random_layout(G, seed=layout_seed)
    elif layout_type == 'spiral':
        node_pos_dict = nx.spiral_layout(G)

    # Add the edges, along with their colors and widths
    edge_colors = []
    edge_widths = []
    for i in range(N):
        for j in range(N):
            a = A[i, j]
            if a > tol:
                G.add_edge(codes[i], codes[j])
                edge_colors.append(node_color_list[i])
                width = a * edge_size_multiple
                edge_widths.append(width)
    
    # Get rid of self-loops
    G.remove_edges_from(nx.selfloop_edges(G))         

    # Plot the networks
    nx.draw_networkx_nodes(G, 
                           node_pos_dict, 
                           node_color=node_color_list, 
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
```

### plot_matrices
```{code-cell}
def plot_matrices(matrix,
                  codes,
                  ax,
                  font_size=12,
                  alpha=0.6, 
                  colormap=cm.viridis, 
                  color45d=None, 
                  xlabel='sector $j$', 
                  ylabel='sector $i$'):
    
    ticks = range(len(matrix))

    levels = np.sqrt(np.linspace(0, 0.75, 100))
    
    
    if color45d != None:
        co = ax.contourf(ticks, 
                         ticks,
                         matrix,
#                          levels,
                         alpha=alpha, cmap=colormap)
        ax.plot(ticks, ticks, color=color45d)
    else:
        co = ax.contourf(ticks, 
                         ticks,
                         matrix,
                         levels,
                         alpha=alpha, cmap=colormap)

    #plt.colorbar(co)

    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_yticks(ticks)
    ax.set_yticklabels(codes)
    ax.set_xticks(ticks)
    ax.set_xticklabels(codes)
```

### unit_simplex
```{code-cell}
def unit_simplex(angle):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    vtx = [[0, 0, 1],
           [0, 1, 0], 
           [1, 0, 0]]
    
    tri = Poly3DCollection([vtx], color='darkblue', alpha=0.3)
    tri.set_facecolor([0.5, 0.5, 1])
    ax.add_collection3d(tri)

    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), 
           xticks=(1,), yticks=(1,), zticks=(1,))

    ax.set_xticklabels(['$(1, 0, 0)$'], fontsize=16)
    ax.set_yticklabels([f'$(0, 1, 0)$'], fontsize=16)
    ax.set_zticklabels([f'$(0, 0, 1)$'], fontsize=16)

    ax.xaxis.majorTicks[0].set_pad(15)
    ax.yaxis.majorTicks[0].set_pad(15)
    ax.zaxis.majorTicks[0].set_pad(35)

    ax.view_init(30, angle)

    # Move axis to origin
    ax.xaxis._axinfo['juggled'] = (0, 0, 0)
    ax.yaxis._axinfo['juggled'] = (1, 1, 1)
    ax.zaxis._axinfo['juggled'] = (2, 2, 0)
    
    ax.grid(False)
    
    return ax
```