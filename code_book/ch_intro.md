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

# Chapter 1 - Introduction (Python Code)

We begin by importing the quantecon package as well as some functions and data that have been packaged for release with this text.

```{code-cell}
import quantecon as qe
import quantecon_book_networks.input_output as qbn_io
import quantecon_book_networks.plotting as qbn_plotting
import quantecon_book_networks.data as qbn_data
ch1_data = qbn_data.introduction()
```

Next we import some common python libraries. 
```{code-cell}
import numpy as np
import pandas as pd
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as plc
import matplotlib.patches as mpatches
import plotly.graph_objects as go
```


## Motivation

### International trade in crude oil 2019

We begin by loading a Networkx directed graph object the represents the international trade in crude oil.

```{code-cell}
DG = ch1_data["crude_oil"]
```

Next we transform the data to prepare it for display as a sankey diagram.

```{code-cell}
nodeid = {}
for ix,nd in enumerate(DG.nodes()):
    nodeid[nd] = ix

# Links
source = []
target = []
value = []
for src,tgt in DG.edges():
    source.append(nodeid[src])
    target.append(nodeid[tgt])
    value.append(DG[src][tgt]['weight'])
```

Finaly we produce our plot.

```{code-cell}
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = list(nodeid.keys()),
      color = "blue"
    ),
    link = dict(
      source = source,
      target = target,
      value = value
  ))])

fig.update_layout(title_text="Crude Oil", font_size=10, width=600, height=800)
fig.show()
```

### International trade in commercial aircraft during 2019.

For this plot we will use a cleaned dataset from [Harvard, CID Dataverse](https://dataverse.harvard.edu/dataverse/atlas).

```{code-cell}
DG = ch1_data['aircraft_network_2019']
pos = ch1_data['aircraft_network_2019_pos']
```

We begin by calculating some features of our graph using the networkx and the quantecon_book_networks packages.

```{code-cell}
centrality = nx.eigenvector_centrality(DG)
node_total_exports = qbn_io.node_total_exports(DG)
edge_weights = qbn_io.edge_weights(DG)
```

Now we convert our graph features to plot features. 

```{code-cell}
node_pos_dict = pos

node_sizes = qbn_io.normalise_weights(node_total_exports,10000)
edge_widths = qbn_io.normalise_weights(edge_weights,10)

node_colors = qbn_io.colorise_weights(list(centrality.values()),color_palette=cm.viridis)
node_to_color = dict(zip(DG.nodes,node_colors))
edge_colors = []
for src,_ in DG.edges:
    edge_colors.append(node_to_color[src])
```

Finally we produce the plot.

```{code-cell}
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')

nx.draw_networkx_nodes(DG, 
                        node_pos_dict, 
                        node_color=node_colors, 
                        node_size=node_sizes, 
                        linewidths=2, 
                        alpha=0.6, 
                        ax=ax)

nx.draw_networkx_labels(DG, 
                        node_pos_dict,  
                        ax=ax)

nx.draw_networkx_edges(DG, 
                        node_pos_dict, 
                        edge_color=edge_colors, 
                        width=edge_widths, 
                        arrows=True, 
                        arrowsize=20,  
                        ax=ax, 
                        arrowstyle='->', 
                        node_size=node_sizes, 
                        connectionstyle='arc3,rad=0.15')

plt.show()
```


## Spectral Theory

### Spectral Radii

Here we provide code for computing the spectral radius of a matrix.

```{code-cell}
def spec_rad(M):
    """
    Compute the spectral radius of M.
    """
    return np.max(np.abs(np.linalg.eigvals(M)))
```

```{code-cell}
M = np.array([[1,2],[2,1]])
spec_rad(M)
```

This function, along with functions for other important calculations from the text, are available in the quantecon_book_networks package. For convenience, source code for these functions can be seen [here](pkg_funcs).

```{code-cell}
qbn_io.spec_rad(M)
```


## Probability

### The unit simplex in $\mathbb{R}^3$.

We begin by defining a function for plotting the unit simplex.

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

We can now produce the plot.

```{code-cell}
unit_simplex(50)
plt.show()
```


## Power Laws

### Independent draws from Student’s t and Normal distributions

We start by generating 1000 samples from a normal distribution and a student's t distribution.

```{code-cell}
from scipy.stats import t
n = 1000
np.random.seed(123)

s = 2
n_data = np.random.randn(n) * s

t_dist = t(df=1.5)
t_data = t_dist.rvs(n)
```

We then plot our samples.

```{code-cell}

fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))

for ax in axes:
    ax.set_ylim((-50, 50))
    ax.plot((0, n), (0, 0), 'k-', lw=0.3)

ax = axes[0]
ax.plot(list(range(n)), t_data, linestyle='', marker='o', alpha=0.5, ms=4)
ax.vlines(list(range(n)), 0, t_data, 'k', lw=0.2)
ax.set_title(f"Student t draws", fontsize=11)

ax = axes[1]
ax.plot(list(range(n)), n_data, linestyle='', marker='o', alpha=0.5, ms=4)
ax.vlines(list(range(n)), 0, n_data, lw=0.2)
ax.set_title(f"$N(0, \sigma)$ with $\sigma = {s}$", fontsize=11)

plt.tight_layout()
plt.show()

```

### CCDF plots for the Pareto and Exponential distributions

First we define our domain and the Pareto and Exponential distributions.
```{code-cell} 
x = np.linspace(1, 10, 500)
```

```{code-cell} 
α = 1.5
def Gp(x):
    return x**(-α)
```

```{code-cell} 
λ = 1.0
def Ge(x):
    return np.exp(-λ * x)
```

We can then produce our plot.

```{code-cell} 
fig, ax = plt.subplots()

ax.plot(np.log(x), np.log(Gp(x)), label="Pareto")
ax.plot(np.log(x), np.log(Ge(x)), label="Exponential")

ax.legend(fontsize=12, frameon=False, loc="lower left")
ax.set_xlabel("$\ln x$", fontsize=12)
ax.set_ylabel("$\ln G(x)$", fontsize=12)

plt.show()
```

### Empirical CCDF plots for largest firms (Forbes)

We start by loading the forbes_global_2000 dataset.
```{code-cell} 
dfff = ch1_data['forbes_global_2000']
```

Next we define an upgraded empirical_ccdf function.
```{code-cell} 
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

Finally we produce our plot.

```{code-cell} 
fig, ax = plt.subplots(figsize=(6.4, 3.5))

label="firm size (market value)"

empirical_ccdf(np.asarray(dfff['Market Value'])[0:500], ax, label=label, add_reg_line=True)

plt.show()
```


## Graph Theory

### Zeta and Pareto distributions

We begin by defining the Zeta and Pareto distributions.
```{code-cell} 
γ = 2.0
α = γ - 1
```

```{code-cell} 
def z(k, c=2.0):
    return c * k**(-γ)

k_grid = np.arange(1, 10+1)
```

```{code-cell} 
def p(x, c=2.0):
    return c * x**(-γ)

x_grid = np.linspace(1, 10, 200)
```

Then we can produce our plot.

```{code-cell} 
fig, ax = plt.subplots()
ax.plot(k_grid, z(k_grid), '-o', label='density of Pareto with tail index $\\alpha$')
ax.plot(x_grid, p(x_grid), label='zeta distribution with $\gamma=2$')
ax.legend(fontsize=12)
ax.set_yticks((0, 1, 2))
plt.show()
```

### Networkx digraph plot

We start by creating a graph object and populating it with edges. 

```{code-cell} 
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

Now we can plot our graph.

```{code-cell} 
fig, ax = plt.subplots()
nx.draw_spring(G_p, ax=ax, node_size=500, with_labels=True, 
                 font_weight='bold', arrows=True, alpha=0.8,
                 connectionstyle='arc3,rad=0.25', arrowsize=20)
plt.show()
```

The DiGraph object has methods that calculate in-degree and out-degree of vertices.

```{code-cell}
G_p.in_degree('p')
```

```{code-cell}
G_p.out_degree('p')
```

Additionally the Networkx package supplies functions for testing communication and strong connectedness, as well as to
compute strongly conneted components.

```{code-cell} 
G = nx.DiGraph()
G.add_edge(1, 1)
G.add_edge(2, 1)
G.add_edge(2, 3)
G.add_edge(3, 2)
list(nx.strongly_connected_components(G))
```

Like Networkx, the QuantEcon Python library 'quantecon' supplies a graph object that implements certain graph-theoretic algorithms. The set of available algorithms is more limited but each one is faster, accelerated by just-in-time compilation. In the case of QuantEcon’s DiGraph object, an instance is created via the adjacency matrix.

```{code-cell} 
A = ((1, 0, 0),
     (1, 1, 1),
     (1, 1, 1))
A = np.array(A) # Convert to NumPy array
G = qe.DiGraph(A)

G.strongly_connected_components
```

### International private credit flows by country

We begin by loading an adjacency matrix of international private credit flows (in the form of a numpy array and a list of country labels).

```{code-cell} 
Z = ch1_data["adjacency_matrix_2019"]["Z"]
Z_visual= ch1_data["adjacency_matrix_2019"]["Z_visual"]
countries = ch1_data["adjacency_matrix_2019"]["countries"]
```

Here we will use the quantecon_book_networks package to convert the adjacency matrix into a networkx graph object. 

```{code-cell} 
G = qbn_io.adjacency_matrix_to_graph(Z_visual, countries, tol=0.03)
```

Next we calculate our graph's properties. We use hub-based eigenvector centrality as our centrality measure for this plot.  

```{code-cell}
centrality = qbn_io.eigenvector_centrality(Z_visual, authority=False)
node_total_exports = qbn_io.node_total_exports(G)
edge_weights = qbn_io.edge_weights(G)
```

Now we convert our graph features to plot features.

```{code-cell}
node_pos_dict = nx.circular_layout(G)

node_sizes = qbn_io.normalise_weights(node_total_exports,3000)
edge_widths = qbn_io.normalise_weights(edge_weights,10)


node_colors = qbn_io.colorise_weights(centrality)
node_to_color = dict(zip(G.nodes,node_colors))
edge_colors = []
for src,_ in G.edges:
    edge_colors.append(node_to_color[src])
```

Finally we produce the plot.

```{code-cell} 
fig, ax = plt.subplots(figsize=(10, 10))
ax.axis('off')

nx.draw_networkx_nodes(G, 
                        node_pos_dict, 
                        node_color=node_colors, 
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

plt.show()
```

### Centrality measures for the credit network

This figure looks at six different centrality measures.

We begin by generating an unweighted version of our matrix to help calculate in-degree and out-degree.  

```{code-cell}
D = qbn_io.build_unweighted_matrix(Z)
```

We now calculate the centrality measures.

```{code-cell}
outdegree = D.sum(axis=1)
ecentral_hub = qbn_io.eigenvector_centrality(Z, authority=False)
kcentral_hub = qbn_io.katz_centrality(Z, b=1/1_400_000)

indegree = D.sum(axis=0)
ecentral_authority = qbn_io.eigenvector_centrality(Z, authority=True)
kcentral_authority = qbn_io.katz_centrality(Z, b=1/1_400_000, authority=True)
```

Here we provide a helper function that returns a dataframe for each measure that is ordered by that measure and contains color information.

```{code-cell}
def centrality_plot_data(countries, centrality_measures):
    df = pd.DataFrame({'code': countries,
                       'centrality':centrality_measures, 
                       'color': qbn_io.colorise_weights(centrality_measures).tolist()
                       })
    return df.sort_values('centrality')
```

We now plot the various centrality measures. 

```{code-cell} 
centrality_measures = [outdegree, indegree, 
                       ecentral_hub, ecentral_authority, 
                       kcentral_hub, kcentral_authority]

ylabels = ['out degree', 'in degree',
           'eigenvector hub','eigenvector authority', 
           'katz hub', 'katz authority']

ylims = [(0, 20), (0, 20), 
         None, None,   
         None, None]


fig, axes = plt.subplots(3, 2, figsize=(10, 12))

axes = axes.flatten()

for i, ax in enumerate(axes):
    df = centrality_plot_data(countries, centrality_measures[i])
      
    ax.bar('code', 'centrality', data=df, color=df["color"], alpha=0.6)
    
    patch = mpatches.Patch(color=None, label=ylabels[i], visible=False)
    ax.legend(handles=[patch], fontsize=12, loc="upper left", handlelength=0, frameon=False)
    
    ax.set_xticklabels(df['code'], fontsize=8)
    if ylims[i] is not None:
        ax.set_ylim(ylims[i])

plt.show()

```

### Computing in and out degree distributions

The in-degree distribution evaluated at 𝑘 is the fraction of nodes in a network that have in-degree 𝑘. The in-degree distribution of a Networkx DiGraph can be calculated using the below.

```{code-cell} 
def in_degree_dist(G):
    n = G.number_of_nodes()
    iG = np.array([G.in_degree(v) for v in G.nodes()])
    d = [np.mean(iG == k) for k in range(n+1)]
    return d
```

The out-degree distribution is defined analogously.

```{code-cell} 
def out_degree_dist(G):
    n = G.number_of_nodes()
    oG = np.array([G.out_degree(v) for v in G.nodes()])
    d = [np.mean(oG == k) for k in range(n+1)]
    return d
```


### Degree distribution for international aircraft trade

Here we illustrate that the commercial aircraft international trade network is approximately scale-free by plotting the degree distribution alongside 𝑓(𝑥) = 𝑐𝑥−𝛾 with 𝑐 = 0.2 and
𝛾 = 1.1. 

In this calculation of the degree distribution, performed by the Networkx function degree_histogram, directions are ignored and the network is treated as an undirected
graph.

```{code-cell} 
def plot_degree_dist(G, ax, loglog=True, label=None):
    "Plot the degree distribution of a graph G on axis ax."
    dd = [x for x in nx.degree_histogram(G) if x > 0]
    dd = np.array(dd) / np.sum(dd)  # normalize
    if loglog:
        ax.loglog(dd, '-o', lw=0.5, label=label)
    else:
        ax.plot(dd, '-o', lw=0.5, label=label)
```

```{code-cell} 
fig, ax = plt.subplots()

plot_degree_dist(DG, ax, loglog=False, label='degree distribution')

xg = np.linspace(0.5, 25, 250)
ax.plot(xg, 0.2 * xg**(-1.1), label='power law')
ax.set_xlim(0.9, 22)
ax.set_ylim(0, 0.25)
ax.legend()
plt.show()
```

### Random graphs

The code to produce the Erdos–Renyi random graph, used below, applies the combinations function from the itertools library. For the call combinations(A, k), the combinations function returns a list of all subsets of 𝐴 of size 𝑘. For example:

```{code-cell} 
import itertools
letters = 'a', 'b', 'c'
list(itertools.combinations(letters, 2))
```

Below we generate random graphs using the Erdos–Renyi and Barabasi-Albert algorithms. Here, for convenience, we will define a function to plot these graphs.

```{code-cell} 
def plot_random_graph(RG,ax):
    node_pos_dict = nx.spring_layout(RG, k=1.1)

    centrality = nx.degree_centrality(RG)
    node_color_list = qbn_io.colorise_weights(list(centrality.values()))

    edge_color_list = []
    for i in range(n):
        for j in range(n):
            edge_color_list.append(node_color_list[i])

    nx.draw_networkx_nodes(RG, 
                           node_pos_dict, 
                           node_color=node_color_list, 
                           edgecolors='grey', 
                           node_size=100,
                           linewidths=2, 
                           alpha=0.8, 
                           ax=ax)

    nx.draw_networkx_edges(RG, 
                           node_pos_dict, 
                           edge_color=edge_colors, 
                           alpha=0.4,  
                           ax=ax)
```

### An instance of an Erdos–Renyi random graph

```{code-cell} 
n = 100
p = 0.05
G_er = qbn_io.erdos_renyi_graph(n, p, seed=1234)
```


```{code-cell} 
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

axes[0].set_title("Graph visualization")
plot_random_graph(G_er,axes[0])

axes[1].set_title("Degree distribution")
plot_degree_dist(G_er, axes[1], loglog=False)

plt.show()
```

### An instance of a preferential attachment random graph

```{code-cell} 
n = 100
m = 5
G_ba = nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=123)
```

```{code-cell} 
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

axes[0].set_title("Graph visualization")
plot_random_graph(G_ba, axes[0])

axes[1].set_title("Degree distribution")
plot_degree_dist(G_ba, axes[1], loglog=False)

plt.show()
```