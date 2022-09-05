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

# Chapter 5 - Nonlinear Interactions (Python Code)

```{code-cell}
---
tags: [hide-output]
---
pip install --upgrade quantecon_book_networks
```

We begin with some imports

```{code-cell}
import quantecon as qe
import quantecon_book_networks
import quantecon_book_networks.input_output as qbn_io
import quantecon_book_networks.plotting as qbn_plt
import quantecon_book_networks.data as qbn_data
export_figures = False
```

```{code-cell}
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
quantecon_book_networks.config("matplotlib")
```

## Financial Networks

### Equity-Cross Holdings

Here we define a class for modelling a financial network where firms are linked by share cross-holdings,
and there are failure costs as described by [Elliott et al. (2014)](https://www.aeaweb.org/articles?id=10.1257/aer.104.10.3115).

```{code-cell}
class FinNet:
    
    def __init__(self, n=100, c=0.72, d=1, θ=0.5, β=1.0, seed=1234):
        
        self.n, self.c, self.d, self.θ, self.β = n, c, d, θ, β
        np.random.seed(seed)
        
        self.e = np.ones(n)
        self.C, self.C_hat = self.generate_primitives()
        self.A = self.C_hat @ np.linalg.inv(np.identity(n) - self.C)
        self.v_bar = self.A @ self.e
        self.t = np.full(n, θ)
        
    def generate_primitives(self):
        
        n, c, d = self.n, self.c, self.d
        B = np.zeros((n, n))
        C = np.zeros_like(B)

        for i in range(n):
            for j in range(n):
                if i != j and np.random.rand() < d/(n-1):
                    B[i,j] = 1
                
        for i in range(n):
            for j in range(n):
                k = np.sum(B[:,j])
                if k > 0:
                    C[i,j] = c * B[i,j] / k
                
        C_hat = np.identity(n) * (1 - c)
    
        return C, C_hat
        
    def T(self, v):
        Tv = self.A @ (self.e - self.β * np.where(v < self.t, 1, 0))
        return Tv
    
    def compute_equilibrium(self):
        i = 0
        v = self.v_bar
        error = 1
        while error > 1e-10:
            print(f"number of failing firms is ", np.sum(v < self.θ))
            new_v = self.T(v)
            error = np.max(np.abs(new_v - v))
            v = new_v
            i = i+1
            
        print(f"Terminated after {i} iterations")
        return v
    
    def map_values_to_colors(self, v, j):
        cols = cm.plasma(qbn_io.to_zero_one(v))
        if j != 0:
            for i in range(len(v)):
                if v[i] < self.t[i]:
                    cols[i] = 0.0
        return cols
```

Now we create a financial network.

```{code-cell}
fn = FinNet(n=100, c=0.72, d=1, θ=0.3, β=1.0)
```

And compute its equilibrium.

```{code-cell}
fn.compute_equilibrium()
```

### Waves of bankruptcies in a financial network

Now we visualise the network after different numbers of iterations. 

For convenience we will first define a function to plot the graphs of the financial network.

```{code-cell}
def plot_fin_graph(G, ax, node_color_list):
    
    n = G.number_of_nodes()

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

Now we will iterate by applying the operator $T$ to the vector of firm values $v$ and produce the plots.

```{code-cell}
G = nx.from_numpy_matrix(np.matrix(fn.C), create_using=nx.DiGraph)
v = fn.v_bar

k = 15
d = 3
fig, axes = plt.subplots(int(k/d), 1, figsize=(10, 12))

for i in range(k):
    if i % d == 0:
        ax = axes[int(i/d)]
        ax.set_title(f"iteration {i}")

        plot_fin_graph(G, ax, fn.map_values_to_colors(v, i))
    v = fn.T(v)
if export_figures:
    plt.savefig("figures/fin_network_sims_1.pdf")
plt.show()

```
