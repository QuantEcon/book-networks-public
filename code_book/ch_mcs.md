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

# Chapter 4 - Markov Chains and Networks (Python Code)

We begin with some imports.

```{code-cell}
import quantecon as qe
import quantecon_book_networks.input_output as qbn_io
import quantecon_book_networks.plotting as qbn_plt
import quantecon_book_networks.data as qbn_data
ch4_data = qbn_data.markov_chains_and_networks()
```

```{code-cell} ipython3
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
```

## Markov Chains as Digraphs

### Contour plot of transition matrix $P_B$

 Benhabib et al. (2015) estimate the following transition matrix for intergenerational social mobility. Here the states are percentiles of the wealth distribution. In particular, the codes 1, 2,… , 8, correspond to the percentiles 0–20%, 20–40%, 40–60%, 60–80%, 80–90%, 90–95%, 95–99%, 99–100% respectively. 

```{code-cell}
P_B = [[0.222, 0.222, 0.215, 0.187, 0.081, 0.038, 0.029, 0.006],
       [0.221, 0.22, 0.215, 0.188, 0.082, 0.039, 0.029, 0.006],
       [0.207, 0.209, 0.21, 0.194, 0.09, 0.046, 0.036, 0.008],
       [0.198, 0.201, 0.207, 0.198, 0.095, 0.052, 0.04, 0.009],
       [0.175, 0.178, 0.197, 0.207, 0.11, 0.067, 0.054, 0.012],
       [0.182, 0.184, 0.2, 0.205, 0.106, 0.062, 0.05, 0.011],
       [0.123, 0.125, 0.166, 0.216, 0.141, 0.114, 0.094, 0.021],
       [0.084, 0.084, 0.142, 0.228, 0.17, 0.143, 0.121, 0.028]]

P_B = np.array(P_B)
codes =  ( '1','2','3','4','5','6','7','8')
```

Here we define a function for producing contour plots of matrices.

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

Finally, we produce the plot.

```{code-cell}
fig, ax = plt.subplots(figsize=(6,6))
plot_matrices(P_B.transpose(), codes, ax, alpha=0.75, 
                 colormap=cm.viridis, color45d='black',
                 xlabel='state at time $t$', ylabel='state at time $t+1$')

plt.show()
```


### Wealth percentile over time

**no code in figures_cource**


### Predicted vs realized cross-country income distributions for 2019

Here we load a pandas dataframe of GDP per capita data for countries compared to the global average.

```{code-cell}
gdppc_df = ch4_data['gdppc_df']
```

Now we assign countries bins as per Quah (1993).

```{code-cell}
q = [0, 0.25, 0.5, 1.0, 2.0, np.inf]
l = [0, 1, 2, 3, 4]

x = pd.cut(gdppc_df.gdppc_r, bins=q, labels=l)
gdppc_df['interval'] = x
gdppc_df['interval'] = gdppc_df['interval'].astype(float)
```

Now we calculate year on year change in bin for each country. 

```{code-cell}
gdppc_df['diff'] = gdppc_df.groupby('country')['interval'].diff(1)
gdppc_df = gdppc_df.reset_index()
gdppc_df['year'] = gdppc_df['year'].astype(float)
```

Here we define a function for calculating the cross-country income distributions for a given date range.

```{code-cell}
def gdp_dist_estimate(df, l, yr=(1960, 2019)):
    Y = np.zeros(len(l))
    for i in l:
        Y[i] = df[
            (df['interval'] == i) & 
            (df['year'] <= yr[1]) & 
            (df['year'] >= yr[0])
            ].count()[0]
    
    return Y / Y.sum()
```

Calculate the true distribution for 1985.

```{code-cell}
ψ_1985 = gdp_dist_estimate(gdppc_df,l,yr=(1985, 1985))
```

Now, we use the transition matrix to update the 1985 distribution 𝑡 = 2019 − 1985 = 34 times to get our predicted 2019 distribution. 

```{code-cell}
P_Q = [[0.97, 0.03, 0.00, 0.00, 0.00],
       [0.05, 0.92, 0.03, 0.00, 0.00],
       [0.00, 0.04, 0.92, 0.04, 0.00],
       [0.00, 0.00, 0.04, 0.94, 0.02],
       [0.00, 0.00, 0.00, 0.01, 0.99]]
P_Q = np.array(P_Q)
ψ_2019_predicted = ψ_1985 @ np.linalg.matrix_power(P_Q, 2019-1985)
```

Now, calculate the true 2019 distribution.
```{code-cell}
ψ_2019 = gdp_dist_estimate(gdppc_df,l,yr=(2019, 2019))
```

Finally we produce the plot. 

```{code-cell}
states = np.arange(1, 6)

fig, ax = plt.subplots()
width = 0.4
ax.plot(states, ψ_2019_predicted, '-o', alpha=0.7, label='predicted')
ax.plot(states, ψ_2019, '-o', alpha=0.7, label='realized')

ax.legend(loc='upper center', fontsize=12)
plt.show()

```

### Distribution dynamics

Here we define a function for plotting the convergence of marginal distributions $ψ$ under a transition matrix $P$ on the unit simplex.

```{code-cell}
def convergence_plot(ψ, P, n=14, angle=50):

    ax = qbn_plt.unit_simplex(angle)

    # Convergence plot
    
    P = np.array(P)

    ψ = ψ        # Initial condition

    x_vals, y_vals, z_vals = [], [], []
    for t in range(n):
        x_vals.append(ψ[0])
        y_vals.append(ψ[1])
        z_vals.append(ψ[2])
        ψ = ψ @ P

    ax.scatter(x_vals, y_vals, z_vals, c='darkred', s=80, alpha=0.7, depthshade=False)

    mc = qe.MarkovChain(P)
    ψ_star = mc.stationary_distributions[0]
    ax.scatter(ψ_star[0], ψ_star[1], ψ_star[2], c='k', s=80)

    return ψ

```

Now we define P.
```{code-cell}
P = (
    (0.9, 0.1, 0.0),
    (0.4, 0.4, 0.2),
    (0.1, 0.1, 0.8)
    )
```

#### A trajectory from $\psi_0 = (0, 0, 1)$

Here we see the sequence of marginals appears to converge. 

```{code-cell}
ψ_0 = (0, 0, 1)
ψ = convergence_plot(ψ_0, P)
plt.show()
```

#### A trajectory from $\psi_0 = (0, 1/2, 1/2)$

Here we see again that the sequence of marginals appears to converge, and the limit appears not to depend on the initial distribution.

```{code-cell}
ψ_0 = (0, 1/2, 1/2)
ψ = convergence_plot(ψ_0, P, n=12)
plt.show()
```


### Distribution projections from $P_B$

**code not in figures_source**




## Asymptotics

### Convergence of the empirical distribution to $\psi^*$