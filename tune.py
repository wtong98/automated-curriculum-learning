"""
Hyperparameter tuning rig for Teacher agent
"""

# <codecell>
import pickle
from multiprocessing import Pool, Manager

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from benchmark import TeacherAgentTest
from env import Teacher, Student, CurriculumEnv

def train_teacher(N=10, T=20, bins=20, p_eps=0.1,
                  teacher_reward=10, teacher_gamma=1, teacher_lr=0.1, student_reward=10,
                  qe_scale=None, anneal_end=None,
                  max_iters=100000, eval_every=5000, eval_iters=10):

    teacher = Teacher(bins=bins, anneal_sched=anneal_end, gamma=teacher_gamma, lr=teacher_lr)

    i = 0
    all_iters_med = []
    all_iters_max = []
    all_iters_min = []
    all_iters_mean = []

    def log(teacher):
        nonlocal i
        i += 1

        if i % eval_every == 0:
            curr_iters = []
            for _ in range(eval_iters):
                test = TeacherAgentTest(teacher, N)
                iters, _ = test.run(Student(), T, max_iters=5000, student_reward=student_reward)
                curr_iters.append(iters)
            
            all_iters_med.append(np.median(curr_iters))
            all_iters_max.append(np.max(curr_iters))
            all_iters_min.append(np.min(curr_iters))
            all_iters_mean.append(np.mean(curr_iters))
            
    env = CurriculumEnv(N, T, 
        p_eps=p_eps, teacher_reward=teacher_reward, student_reward=student_reward, 
        student_qe_dist=qe_scale)
    teacher.learn(env, max_iters=max_iters, use_tqdm=True, post_hook=log)

    return {
        'teacher': teacher,
        'iters_med': np.array(all_iters_med),
        'iters_max': np.array(all_iters_max),
        'iters_min': np.array(all_iters_min),
        'iters_mean': np.array(all_iters_mean)
    }


def _do_run(run_params):
    results = train_teacher(**run_params)
    return tuple(run_params.values()), results

def tune(params_dict, n_runs=10, n_procs=1):
    names = tuple(params_dict.keys())
    all_run_params = []
    seen = set()
    
    for _ in range(n_runs):
        params = {}
        while len(params) == 0 or tuple(params.values()) in seen:
            for param in params_dict:
                params[param] = np.random.choice(params_dict[param])

        all_run_params.append(params)
        seen.add(tuple(params.values()))
    
    with Pool(n_procs) as pool:
        results = pool.map(_do_run, all_run_params)

    record = {k: v for k, v in results}
    return record, names
    

# <codecell>
params_dict = {
    'bins':[3, 5, 10, 15, 20, 30],
    'p_eps': [0.1, 0.05],
    'teacher_gamma': [0.5, 0.8, 0.9, 0.95, 1],
    'teacher_lr': [0.01, 0.05, 0.1, 0.2],
    'qe_scale': [0, 0.25, 0.5, 1, None],
    'anneal_end': [5, 10, 20, None]
}

record, param_names = tune(params_dict, n_runs=1600, n_procs=16)

with open('results.pkl', 'wb') as fp:
    pickle.dump((record, param_names), fp)

# <codecell>
with open('results.pkl', 'rb') as fp:
    record, param_names = pickle.load(fp)

# <codecell>
#### PICK BEST RECORD
def score(result):
    return result['iters_mean'][-1]

best_params = min(record.keys(), key=lambda k: score(record[k]))
results = record[best_params]

# <codecell>
lwr = results['iters_med'] - results['iters_min']
upr = results['iters_max'] - results['iters_med']
plt.errorbar(np.arange(len(results['iters_med'])), results['iters_med'], yerr=(lwr, upr))
plt.savefig('tmp_run_plot.png')

# %%
