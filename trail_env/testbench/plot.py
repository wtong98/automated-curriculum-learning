"""
Plotting benchmark results
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# <codecell>
df = pd.read_pickle('meander_bench.pkl')