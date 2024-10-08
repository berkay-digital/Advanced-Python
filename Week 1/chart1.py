import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

plt.figure(figsize=(12, 8))

x = np.linspace(0, 7, 100)
y_top = 4 + x**2
y_bottom = 1 + 2 * x
y_average = (y_top + y_bottom) / 2

plt.fill_between(x, y_bottom, y_top, alpha=0.3, color='skyblue')
plt.plot(x, y_average, color='blue', linewidth=2, label='f(x)')

plt.title("Important chart with fontsize 24", fontsize=24)
plt.xlabel("This is xlabel with math symbol $\\alpha > \\beta \\sum_{i=0}^\\infty x_i$", fontsize=12)
plt.ylabel("This is ylabel", fontsize=12)

plt.xlim(0, 7)
plt.ylim(1, 22)

plt.xticks(range(8))
plt.yticks(range(1, 23, 3))

plt.grid(color='red', linestyle='--', linewidth=0.5)

ax = plt.gca()
ax.set_facecolor('#FFFACD')

for spine in ax.spines.values():
    spine.set_visible(False)

plt.legend(loc='center left', bbox_to_anchor=(0.1, 0.5))

plt.text(5.2, 6, 'boxed oblique text\nin position 5.2, 6', 
         bbox=dict(facecolor='red', alpha=0.5), 
         rotation=15, ha='left', va='bottom')

ax.tick_params(width=5)

plt.tight_layout()
plt.show()
