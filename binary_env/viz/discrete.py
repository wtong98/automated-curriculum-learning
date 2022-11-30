"""
Visualizations for discrete methods

PLOTS TO MAKE
----------------
Benchmarks:
* Const N, vary eps (x3)
* Const N, vary eps, with generative noise model (x3, x3)
* Const eps, vary N (x3)

Illustrative examples:
* Trajectory plots
* Q-value-over-time plots
"""
# <codecell>
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../')
from env import *


# <codecell>
### BENCHMARK: N x EPS
# iters = 5
# Ns = np.arange(3, 20, step=2)
# eps = np.linspace(-5, 3, num=9)

iters = 2
Ns = np.arange(3, 8, step=2)
eps = np.linspace(2, 2, num=4)

for N in Ns:
    for e in eps:
        pass


# %%
