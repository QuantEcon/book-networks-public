import numpy as np
import matplotlib.cm as cm           # import colormap stuff!
import networkx as nx
from scipy.stats import beta
import itertools

def spec_rad(M):
    """
    Compute the spectral radius of M.
    """
    return np.max(np.abs(np.linalg.eigvals(M)))


def build_unweighted_matrix(Z, tol=1e-5):
    """
    return a unweighted adjacency matrix
    """
    return 1*(Z>tol)


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


def to_zero_one(x):
    "Map vector x to the zero one interval."
    x = np.array(x)
    x_min, x_max = x.min(), x.max()
    return (x - x_min)/(x_max - x_min)


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


def adjacency_matrix_to_graph(A, 
               codes,
               node_weights = None,
               tol=0.0):  # clip entries below tol
    """
    Build a networkx graph object given an adjacency matrix
    """
    G = nx.DiGraph()
    N = len(A)

    # Add nodes
    for i, code in enumerate(codes):
        if node_weights is None:
            G.add_node(code, name=code)
        else:
            G.add_node(code, weight=node_weights[i], name=code)

    # Add the edges
    for i in range(N):
        for j in range(N):
            a = A[i, j]
            if a > tol:
                G.add_edge(codes[i], codes[j], weight=a)

    return G

## Nodes
def node_total_exports(G):
    node_exports = []
    for node1 in G.nodes():
        total_export = 0
        for node2 in G[node1]:
            total_export += G[node1][node2]['weight']
        node_exports.append(total_export)
    return node_exports

def node_total_imports(G):
    node_imports = []
    for node1 in G.nodes():
        total_import = 0
        for node2 in G[node1]:
            total_import += G[node2][node1]['weight']
        node_imports.append(total_import)
    return node_imports

## Edges
def edge_weights(G):
    edge_weights = [G[u][v]['weight'] for u,v in G.edges()]
    return edge_weights

## Plotting prep
def normalise_weights(weights,scalar=1):
    max_value = np.max(weights)
    return [scalar * (weight / max_value) for weight in weights]

def colorise_weights(weights,beta=True,color_palette=cm.plasma):
    if beta:
        cp = color_palette(to_zero_one_beta(weights))
    else:
        cp = color_palette(to_zero_one(weights))
    return cp 

## Random graphs
def erdos_renyi_graph(n=100, p=0.5, seed=1234):
    "Returns an Erdős-Rényi random graph."
    
    np.random.seed(seed)
    edges = itertools.combinations(range(n), 2)
    G = nx.Graph()
    
    for e in edges:
        if np.random.rand() < p:
            G.add_edge(*e)
    return G