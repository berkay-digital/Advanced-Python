import matplotlib.pyplot as plt
import numpy as np

def create_scatter_plot(ax):
    n_points = 24
    x = np.random.normal(4, 1.5, n_points)
    y = np.random.normal(4, 1.5, n_points)
    sizes = np.random.uniform(50, 200, n_points)
    colors = np.random.rand(n_points, 3)
    
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.7)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(range(1, 8))
    ax.set_yticks(range(1, 8))
    ax.tick_params(axis='both', which='major', labelsize=8)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

create_scatter_plot(ax1)
create_scatter_plot(ax2)

plt.tight_layout()
plt.show()