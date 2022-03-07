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

# The quantecon_book_networks package

## Chapter 1

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
def colorise_weights(weights,zero_one_func=to_zero_one_beta,color_palette=cm.plasma):
    return color_palette(zero_one_func(weights))
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