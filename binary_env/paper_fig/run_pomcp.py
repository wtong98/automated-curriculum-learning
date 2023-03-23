"""
POMCP-specific benchmarks
"""

import sys
sys.path.append('../')

import pandas as pd

from multiprocessing import Pool
from common import *

n_procs = 16

n_iters = 3
Ns = [3, 5, 10]
eps = np.linspace(-2, 2, num=5)

T = 3
lr = 0.1
    
def run(case):
    print('info: running case', case)
    run_exp(n_iters=n_iters, cases=[case], max_steps=N*100, lr=lr, T=T)

if __name__ == '__main__':
    raw_data = []

    for N in Ns:
        for e in eps:
            cases = [
                Case('POMCP', run_pomcp_with_retry, {'eps': e, 'goal_length': N}, []),
            ]
            raw_data.extend(cases)

    try:
        with Pool(n_procs) as p:
            p.map(run, raw_data)
    except Exception as e:
        print(f'error: {e}')

    df = pd.DataFrame(raw_data)
    df.to_pickle('pomcp.pkl')

