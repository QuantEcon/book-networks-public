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

# Chapter 1 - Introduction Code

We begin with some imports.

```{code-cell} ipython3

import pandas as pd
import numpy as np
import networkx as nx
import json
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as plc
import matplotlib.patches as mpatches

```

And defining some helper functions.

```{code-cell} ipython3

from scipy.stats import beta

def to_zero_one(x, 
                qrange=[0.25, 0.75], 
                beta_para=[0.5, 0.5]):
    
    """
    Nonlinearly map vector x to the zero one interval with beta distribution.
    https://en.wikipedia.org/wiki/Beta_distribution
    """
    x_min, x_max = x.min(), x.max()
    if beta_para != None:
        a, b = beta_para
        return beta.cdf((x - x_min) /(x_max - x_min), a, b)
    else:
        q1, q2 = qrange
        return (x - x_min) * (q2 - q1) /(x_max - x_min) + q1
```



## Motivation

- Figure 1.1 - International trade in commercial aircraft during 2019.

We begin by loading a cleaned dataset from [Harvard, CID Dataverse](https://dataverse.harvard.edu/dataverse/atlas), complemented by placement data from the Gephi JSON Network File.

```{code-cell} ipython3

DATA_DIR = "../figures_source/commercial-aircraft-sitcr2-7924-yr2019"

DG = nx.read_gexf(f"{DATA_DIR}/sitcr2-7924-aircraft-network-2019.gexf")

f = open(f"{DATA_DIR}/sitcr2-7924-aircraft-network-2019-layout.json", "r")
data = json.loads(f.read())
pos = {}
for nd in data['nodes']:
    pos[nd['id']] = np.array([nd['x'], nd['y']])

```

Next we compute the node and edge attributes.

```{code-cell} ipython3

# Compute total Exports (for Node Size)
node_size = {}
for node1 in DG.nodes():
    total_export = 0
    for node2 in DG[node1]:
        total_export += DG[node1][node2]['weight']
    node_size[node1] = total_export

# Normalise Node Sizes
node_scalar = 10000
max_value = np.max([v for x,v in node_size.items()])
for node, value in node_size.items():
    node_size[node] = value / max_value * node_scalar

# Edge Weights (from Graph 'weight')
node_scalar = 4
edge_weights = [DG[u][v]['weight'] for u,v in DG.edges()]
edge_weights = edge_weights / np.max(edge_weights)

```

Next we compute network attributes.

```{code-cell} ipython3

centrality = nx.out_degree_centrality(DG)
eig_centrality = nx.eigenvector_centrality(DG)

```

Finally we produce the plot.

```{code-cell} ipython3

# Node Colours
node_names = [nd for nd, val in eig_centrality.items()]
node_colours = cm.viridis(to_zero_one(np.array([val for nd, val in eig_centrality.items()])))
node_to_colour = dict(zip(node_names, node_colours))

# Compute Edge Colour based on Source Node
edge_colours = []
for src,_ in DG.edges:
    edge_colours.append(node_to_colour[src])

fig, ax = plt.subplots(figsize=(10, 10))
plt.axis("off")
nx.draw_networkx(
    DG, 
    ax=ax,
    pos=pos, 
    with_labels=True,
    alpha=0.7,
    arrowsize=15,
    connectionstyle="arc3,rad=0.1",
    node_size=[size for nd,size in node_size.items()], 
    node_color=node_colours,
    edge_color=edge_colours,
    width=edge_weights*10,
)

```

## Probability

- Figure 1.3 - The unit simplex in $\mathbb{R}^3$.

```{code-cell} ipython3

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

unit_simplex(50)
plt.show()

```

## Power Laws

- Figure 1.4: Independent draws from Student’s t- and normal distributions

We start with some imports.

```{code-cell} ipython3

from scipy.stats import cauchy
from scipy.stats import t
from scipy.integrate import quad

```

```{code-cell} ipython3

t_dist = t(df=1.5)
n = 1000
np.random.seed(123)

```


```{code-cell} ipython3

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))

for ax in axes:
    ax.set_ylim((-50, 50))
    ax.plot((0, n), (0, 0), 'k-', lw=0.3)


ax = axes[1]
s = 2
data = np.random.randn(n) * s
ax.plot(list(range(n)), data, linestyle='', marker='o', alpha=0.5, ms=4)
ax.vlines(list(range(n)), 0, data, lw=0.2)
ax.set_title(f"$N(0, \sigma)$ with $\sigma = {s}$", fontsize=11)

ax = axes[0]
data = t_dist.rvs(n)

ax.plot(list(range(n)), data, linestyle='', marker='o', alpha=0.5, ms=4)
ax.vlines(list(range(n)), 0, data, 'k', lw=0.2)
ax.set_title(f"Student t draws", fontsize=11)

#plt.subplots_adjust(hspace=0.4)
plt.tight_layout()

plt.show()

```

- Figure 1.5: CCDFs plots for the Pareto and exponential distributions

```{code-cell} ipython3
α = 1.5
def Gp(x):
    return x**(-α)

λ = 1.0
def Ge(x):
    return np.exp(-λ * x)
```

```{code-cell} ipython3

fig, ax = plt.subplots()

x = np.linspace(1, 10, 500)
ax.plot(np.log(x), np.log(Gp(x)), label="Pareto")
ax.plot(np.log(x), np.log(Ge(x)), label="Exponential")

ax.legend(fontsize=12, frameon=False, loc="lower left")
ax.set_xlabel("$\ln x$", fontsize=12)
ax.set_ylabel("$\ln G(x)$", fontsize=12)

```

- Figure 1.6: CCDFs plots for the Pareto and exponential distributions


```{code-cell} ipython3

dfff = pd.read_csv('data/csv_files/forbes-global2000.csv')
dfff = dfff[['Country', 'Sales', 'Profits', 'Assets', 'Market Value']]

```

```{code-cell} ipython3
import statsmodels.api as sm
from interpolation import interp

def empirical_ccdf(data, 
                   ax, 
                   aw=None,   # weights
                   label=None,
                   xlabel=None,
                   add_reg_line=False, 
                   title=None):
    """
    Take data vector and return prob values for plotting.
    Upgraded empirical_ccdf
    """
    y_vals = np.empty_like(data, dtype='float64')
    p_vals = np.empty_like(data, dtype='float64')
    n = len(data)
    if aw is None:
        for i, d in enumerate(data):
            # record fraction of sample above d
            y_vals[i] = np.sum(data >= d) / n
            p_vals[i] = np.sum(data == d) / n
    else:
        fw = np.empty_like(aw, dtype='float64')
        for i, a in enumerate(aw):
            fw[i] = a / np.sum(aw)
        pdf = lambda x: interp(data, fw, x)
        data = np.sort(data)
        j = 0
        for i, d in enumerate(data):
            j += pdf(d)
            y_vals[i] = 1- j

    x, y = np.log(data), np.log(y_vals)
    
    results = sm.OLS(y, sm.add_constant(x)).fit()
    b, a = results.params
    
    kwargs = [('alpha', 0.3)]
    if label:
        kwargs.append(('label', label))
    kwargs = dict(kwargs)

    ax.scatter(x, y, **kwargs)
    if add_reg_line:
        ax.plot(x, x * a + b, 'k-', alpha=0.6, label=f"slope = ${a: 1.2f}$")
    if not xlabel:
        xlabel='log value'
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("log prob.", fontsize=12)
        
    if label:
        ax.legend(loc='lower left', fontsize=12)
        
    if title:
        ax.set_title(title)
        
    return np.log(data), y_vals, p_vals
```

```{code-cell} ipython3

fig, ax = plt.subplots(figsize=(6.4, 3.5))

label="firm size (market value)"

d = dfff.sort_values('Market Value', ascending=False)

empirical_ccdf(np.asarray(d['Market Value'])[0:500], ax, label=label, add_reg_line=True)

plt.show()

```

## Graph Theory

- Figure 1.7: Zeta and Pareto distributions

```{code-cell} ipython3

γ = 2.0
α = γ - 1

```

```{code-cell} ipython3

def z(k, c=2.0):
    return c * k**(-γ)

k_grid = np.arange(1, 10+1)

```

```{code-cell} ipython3

def p(x, c=2.0):
    return c * x**(-γ)

x_grid = np.linspace(1, 10, 200)

```

```{code-cell} ipython3

fig, ax = plt.subplots()
ax.plot(k_grid, z(k_grid), '-o', label='density of Pareto with tail index $\\alpha$')
ax.plot(x_grid, p(x_grid), label='zeta distribution with $\gamma=2$')
ax.legend(fontsize=12)
ax.set_yticks((0, 1, 2))
plt.show()

```

- Figure 1.10: Networkx digraph plot

```{code-cell} ipython3

G_p = nx.DiGraph()

edge_list = [
    ('p', 'p'),
    ('m', 'p'), ('m', 'm'), ('m', 'r'),
    ('r', 'p'), ('r', 'm'), ('r', 'r')
]

for e in edge_list:
    u, v = e
    G_p.add_edge(u, v)

```

```{code-cell} ipython3

fig, ax = plt.subplots()
nx.draw_spring(G_p, ax=ax, node_size=500, with_labels=True, 
                 font_weight='bold', arrows=True, alpha=0.8,
                 connectionstyle='arc3,rad=0.25', arrowsize=20)
plt.show()

```

- Figure 1.13: International private credit flows by country & Figure 1.18: Centrality measures for the credit network

```{code-cell} ipython3

def read_Z(data_file='data/adjacency_matrix_31-12-2019.csv', t=10):
    """
    Build the Z matrix from the use table.
    
    * Z[i, j] = sales from sector i to sector j
    
    """
    
    df1 = pd.read_csv(data_file)
    df1 = df1.set_index("country")

    df2 = df1.replace(np.nan, 0)          # replace nan with 0

    df3 = df2.replace("...", 0)          # replace ... with 0

    countries = list(df3.index)

    Z = np.asarray(df3.values.tolist(), dtype=np.float64)
    Z = np.where(Z < t, 0, Z)
    return Z, countries

```

```{code-cell} ipython3

def compute_sums(D, sumtype="column_sum"):
    n = len(D)
    ds = np.empty(n)
    for i in range(n):
        if sumtype == "column_sum":
            d = 0
            for j in range(n):
                d += float(D[j, i])
            ds[i] = d
        if sumtype == "row_sum":
            d = 0
            for j in range(n):
                d += float(D[i, j])
            ds[i] = d
    return ds

```

```{code-cell} ipython3

Z, countries = read_Z(data_file='data/csv_files/adjacency_matrix_31-12-2019.csv', t=0)
Z_visual, countries = read_Z(data_file='data/csv_files/adjacency_matrix_31-12-2019.csv', t=10)
countries = np.array(countries)
countries = np.where(countries == 'CH', 'SW', countries)
X = compute_sums(Z, sumtype="row_sum")

```

```{code-cell} ipython3
def spec_rad(M):
    """
    Compute the spectral radius of M.
    """
    return np.max(np.abs(np.linalg.eigvals(M)))
```

```{code-cell} ipython3
def build_unweighted_matrix(Z, tol=1e-5):
    """
    return a unweighted adjacency matrix
    """
    D = np.empty_like(Z, dtype=np.int32)
    n = D.shape[0]
    for i in range(n):
        for j in range(n):
            if Z[i, j] < tol:
                D[i, j] = 0
            else:
                D[i, j] = 1
    return np.nan_to_num(D, nan=0)

D = build_unweighted_matrix(Z)

```

```{code-cell} ipython3

indegree = compute_sums(D, sumtype="column_sum")
indegree_color_list = cm.plasma(to_zero_one(indegree, beta_para=[0.5, 0.5]))

df = pd.DataFrame({'codes':countries,
                  'indegree':indegree, 
                  'indegree_color_list': indegree_color_list.tolist()})

df_sorted1 = df[['codes', 'indegree', 'indegree_color_list']].sort_values('indegree')

```

```{code-cell} ipython3
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

```{code-cell} ipython3

ecentral_authority = eigenvector_centrality(Z, authority=True)
ecentral_authority_color_list = cm.plasma(to_zero_one(ecentral_authority, beta_para=[0.5, 0.5]))


df['ecentral_authority'] = ecentral_authority
df['ecentral_authority_color_list'] = ecentral_authority_color_list.tolist()

df_sorted2 = df[['codes', 'ecentral_authority', 'ecentral_authority_color_list']].sort_values('ecentral_authority')
df_sorted2

```


```{code-cell} ipython3

def katz_centrality(A, b=1, authority=False):
    """
    Computes the Katz centrality of A, defined as the x solving

    x = 1 + b A x    (1 = vector of ones)

    Assumes that A is square.

    If authority=True, then A is replaced by its transpose.
    """
    if spec_rad(b * A) < 1: 
        n = len(A)
        I = np.identity(n)
        C = I - b * A.T if authority else I - b * A
        return np.linalg.solve(C, np.ones(n))
    else:
        return "Stability condition violated. Hint: set b < 1 / r(A)"

```

```{code-cell} ipython3
kcentral_authority = katz_centrality(Z, b=1/1_400_000, authority=True)
kcentral_authority_color_list = cm.plasma(to_zero_one(kcentral_authority, beta_para=[0.5, 0.5]))

df['kcentral_authority'] = kcentral_authority
df['kcentral_authority_color_list'] = kcentral_authority_color_list.tolist()
df_sorted3 = df[['codes', 'kcentral_authority', 'kcentral_authority_color_list']].sort_values('kcentral_authority')

```

```{code-cell} ipython3

outdegree = compute_sums(D, sumtype="row_sum")
outdegree_color_list = cm.plasma(to_zero_one(outdegree, beta_para=[0.5, 0.5]))

df['outdegree'] = outdegree
df['outdegree_color_list'] = outdegree_color_list.tolist()
df_sorted4 = df[['codes', 'outdegree', 'outdegree_color_list']].sort_values('outdegree')

```

```{code-cell} ipython3

ecentral_hub = eigenvector_centrality(Z, authority=False)
ecentral_hub_color_list = cm.plasma(to_zero_one(ecentral_hub, beta_para=[0.5, 0.5]))


df['ecentral_hub'] = ecentral_hub
df['ecentral_hub_color_list'] = ecentral_hub_color_list.tolist()

df_sorted5 = df[['codes', 'ecentral_hub', 'ecentral_hub_color_list']].sort_values('ecentral_hub')

```

```{code-cell} ipython3
kcentral_hub = katz_centrality(Z, b=1/1_400_000)
kcentral_hub_color_list = cm.plasma(to_zero_one(kcentral_hub, beta_para=[0.5, 0.5]))

df['kcentral_hub'] = kcentral_hub
df['kcentral_hub_color_list'] = kcentral_hub_color_list.tolist()
df_sorted6 = df[['codes', 'kcentral_hub', 'kcentral_hub_color_list']].sort_values('kcentral_hub')
```

```{code-cell} ipython3
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
    
    nx.draw_networkx_nodes(G, 
                           node_pos_dict, 
                           node_color=node_color_list, 
                           node_size=node_sizes, 
                           edgecolors='grey', 
                           linewidths=2, 
                           alpha=0.4, 
                           ax=ax)

    nx.draw_networkx_labels(G, 
                            node_pos_dict, 
                            font_size=12,
                            font_weight='black',
                            ax=ax)

    nx.draw_networkx_edges(G, 
                           node_pos_dict, 
                           edge_color=edge_colors, 
                           width=edge_widths, 
                           arrows=True, 
                           arrowsize=20, 
                           alpha=0.8,  
                           ax=ax, 
                           arrowstyle='->', 
                           node_size=node_sizes, 
                           connectionstyle='arc3,rad=0.15')
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 10))
plt.axis("off")

plot_graph(Z_visual, to_zero_one(X, beta_para=[0.5, 0.5]), ax, countries,
           layout_type='spring', # alternative layouts: spring, circular, random, spiral
           layout_seed=1234,    # 5432167
           node_size_multiple=3000,
           edge_size_multiple=0.000006,
           tol=0.0,
           node_color_list=ecentral_hub_color_list) 

plt.show()
```

- Figure 1.18: Centrality measures for the credit network

```{code-cell} ipython3

labels = ['outdegree',  'indegree',
          'ecentral_hub', 'ecentral_authority',
          'kcentral_hub', 'kcentral_authority']

ylabels = ['out degree', 'in degree',
           'eigenvector hub','eigenvector authority', 
           'Katz hub', 'Katz authority']

dfs = [df_sorted4, df_sorted1, 
       df_sorted5, df_sorted2, 
       df_sorted6, df_sorted3]
ylims = [(0, 20), (0, 20), 
         None, None,   
         None, None]

```


```{code-cell} ipython3

fig, axes = plt.subplots(3, 2, figsize=(10, 12))

axes = axes.flatten()

for ax, label, df, ylabel, ylim in zip(axes, labels, dfs, ylabels, ylims):
    ax.bar('codes', label, data=df, color=df[label+"_color_list"], alpha=0.6)
    patch = mpatches.Patch(color=None, label=ylabel, visible=False)
    ax.legend(handles=[patch], fontsize=12, loc="upper left", handlelength=0, frameon=False)
    ax.set_xticklabels(df['codes'], fontsize=8)
    if ylim is not None:
        ax.set_ylim(ylim)


plt.show()

```

- Figure 1.20: Degree distribution for international aircraft trade

```{code-cell} ipython3
def plot_degree_dist(G, ax, loglog=True, label=None):
    "Plot the degree distribution of a graph G on axis ax."
    dd = [x for x in nx.degree_histogram(G) if x > 0]
    dd = np.array(dd) / np.sum(dd)  # normalize
    if loglog:
        ax.loglog(dd, '-o', lw=0.5, label=label)
    else:
        ax.plot(dd, '-o', lw=0.5, label=label)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

plot_degree_dist(DG, ax, loglog=False, label='degree distribution')

xg = np.linspace(0.5, 25, 250)
ax.plot(xg, 0.2 * xg**(-1.1), label='power law')
ax.set_xlim(0.9, 22)
ax.set_ylim(0, 0.25)
ax.legend()
plt.show()
```

- Figure 1.21: An instance of an Erdos–Renyi random graph


```{code-cell} ipython3

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

```{code-cell} ipython3

def plot_graph(G, ax):
    
    n = G.number_of_nodes()
    cols = nx.degree_centrality(G)
    x = np.array(list(cols.values()))
    node_color_list = cm.plasma(to_zero_one(x))
    node_pos_dict = nx.spring_layout(G, k=1.1)
    edge_colors = []

    for i in range(n):
        for j in range(n):
            edge_colors.append(node_color_list[i])

    
    nx.draw_networkx_nodes(G, 
                           node_pos_dict, 
                           node_color=node_color_list, 
                           edgecolors='grey', 
                           node_size=100,
                           linewidths=2, 
                           alpha=0.8, 
                           ax=ax)

    nx.draw_networkx_edges(G, 
                           node_pos_dict, 
                           edge_color=edge_colors, 
                           alpha=0.4,  
                           ax=ax)

```

```{code-cell} ipython3

def plot_degree_dist(G, ax, loglog=True):
    "Plot the degree distribution of a graph G on axis ax."
    dd = [x for x in nx.degree_histogram(G) if x > 0]
    dd = np.array(dd) / np.sum(dd)  # normalize
    if loglog:
        ax.loglog(dd, '-o', lw=0.5)
    else:
        ax.plot(dd, '-o', lw=0.5)

```

```{code-cell} ipython3

n, m, p = 100, 5, 0.05

G_ba = nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=123)

G_er = erdos_renyi_graph(n, p, seed=1234)

```

```{code-cell} ipython3

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))


axes[0].set_title("Graph visualization")
plot_graph(G_er, axes[0])
axes[1].set_title("Degree distribution")
plot_degree_dist(G_er, axes[1], loglog=False)

plt.show()

```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))


axes[0].set_title("Graph visualization")
plot_graph(G_ba, axes[0])
axes[1].set_title("Degree distribution")
plot_degree_dist(G_ba, axes[1], loglog=False)

plt.show()
```