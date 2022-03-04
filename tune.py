"""
Hyperparameter tuning rig for Teacher agent
"""

# <codecell>
from multiprocessing import Pool, Manager

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from benchmark import TeacherAgentTest
from env import Teacher, Student, CurriculumEnv

def train_teacher(N=10, T=20, bins=20, p_eps=0.1,
                  teacher_reward=10, teacher_gamma=1, student_reward=10,
                  qe_gen=None, anneal_sched=None,
                  max_iters=100000, eval_every=10000, eval_iters=10):

    teacher = Teacher(bins=bins, anneal_sched=anneal_sched, gamma=teacher_gamma)

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

    # def log(teacher):
    #     nonlocal i
    #     i += 1

    #     if i % eval_every == 0:
    #         eval_env = CurriculumEnv(N, T, 
    #             p_eps=p_eps, 
    #             student_reward=student_reward, teacher_reward=teacher_reward)
            
    #         completions = []
    #         def count_done(teacher, reward):
    #             if reward > 0:
    #                 completions.append(teacher.iter)
            
    #         teacher.learn(eval_env, is_eval=True, done_hook=count_done, max_iters=10000)

    #         pairs = zip([0] + completions, completions)
    #         curr_iters = []
    #         for start, end in pairs:
    #             curr_iters.append(end - start)
            
    #         all_iters_med.append(np.median(curr_iters))
    #         all_iters_max.append(np.max(curr_iters))
    #         all_iters_min.append(np.min(curr_iters))

            
    env = CurriculumEnv(N, T, 
        p_eps=p_eps, teacher_reward=teacher_reward, student_reward=student_reward, 
        student_qe_dist=qe_gen)
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
    'max_iters': [15000, 20000],
    'N': [8, 10]
}

record, param_names = tune(params_dict, n_runs=4, n_procs=4)

# <codecell>

# <codecell>
results = train_teacher()

# <codecell>
lwr = results['iters_med'] - results['iters_min']
upr = results['iters_max'] - results['iters_med']
plt.errorbar(np.arange(len(results['iters_med'])), results['iters_med'], yerr=(lwr, upr))
# plt.savefig('fig/tmp_run_plot.png')

# %%
