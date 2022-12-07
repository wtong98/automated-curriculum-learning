"""
Plotting benchmark results
"""

# <codecell>
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# <codecell>
df = pd.read_pickle('meander_results.pkl')

def extract_plot_vals(row):
    traj_lens = [len(traj) for traj in row['runs']]

    return pd.Series([
        row['name'],
        traj_lens
    ], index=['name', 'traj_lens'])


plot_df = df.apply(extract_plot_vals, axis=1).explode('traj_lens')

ax = sns.barplot(plot_df, x='name', y='traj_lens', color='C0')
ax.set_ylabel('Iterations')
ax.set_xlabel('')
ax.set_title(f'Meandering Trail Benchmark')

plt.gcf().tight_layout()