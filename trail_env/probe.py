"""
Testing ground for probing a trained agent
"""
# <codecell>
import torch
from matplotlib.animation import FuncAnimation
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from stable_baselines3 import PPO
from tqdm import tqdm

from trail import TrailEnv
from trail_map import *


# <codecell>
global_discrete = True
global_treadmill = True
trail_class = MeanderTrail
# trail_args = {'width': 3, 'length': 69, 'radius': 100, 'diff_rate': 0.04, 'breaks': [(0.5, 0.8)]}
trail_args = {
    'width': 5, 
    'length': 90, 
    'radius': 100, 
    'diff_rate': 0.01, 
    'reward_dist': 3,
    # 'breaks':[(0.6, 0.8)]
    # 'breaks':[(0.5, 0.99)]
}

# Straight "meandering" trail
# trail_args = {'width': 5, 'length': 69, 'radius': 999, 'diff_rate': 0, 'breaks':[(0.5, 0.8)]}


# <codecell>
# RUNNING AGENT ON TRAILS

trail_map = trail_class(**trail_args, heading=0)
env = TrailEnv(trail_map, discrete=global_discrete, treadmill=global_treadmill)
model = PPO.load('trail_model', device='cpu')
# model = PPO("CnnPolicy", env, verbose=1,
#             n_steps=128,
#             batch_size=256,
#             ent_coef=8e-6,
#             gamma=0.98,
#             gae_lambda=0.9,
#             clip_range=0.3,
#             max_grad_norm=1,
#             vf_coef=0.36,
#             n_epochs=16,
#             learning_rate=0.0001,
#             policy_kwargs={
#                 'net_arch': [{'pi': [128, 32], 'vf': [128, 32]}],
#                 'activation_fn': torch.nn.ReLU
#             },
#             tensorboard_log='log',
#             device='cpu'
# )
# model.set_parameters('gen1.zip')

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_done, _ = env.step(action)

    if is_done:
        break

env.map.plot(ax=plt.gca())
plt.plot(*zip(*env.agent.position_history), linewidth=2, color='black')
plt.savefig('out.png')

# print(env.agent.odor_history)

# <codecell>
# MULTI-SAMPLE PLOTTER
# model = PPO.load('trail_model.zip', device='cpu')
# model = PPO.load('trained/epoch_3/possible_caster_feb5.zip', device='cpu')

n_runs = 8
headings = np.linspace(-np.pi, np.pi, num=n_runs)

maps = []
position_hists = []

for heading in tqdm(headings):
    trail_map = trail_class(**trail_args, heading=heading)
    env = TrailEnv(trail_map, discrete=global_discrete, treadmill=global_treadmill)

    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, is_done, _ = env.step(action)

        if is_done:
            break
    
    maps.append(trail_map)
    position_hists.append(env.agent.position_history)

fig, axs = plt.subplots(2, 4, figsize=(16, 8))

for ax, m, position_history in zip(axs.ravel(), maps, position_hists):
    m.plot(ax=ax)
    ax.plot(*zip(*position_history), linewidth=2, color='black')

fig.suptitle('Sample of agent runs')
fig.tight_layout()
plt.savefig('out.png')

# <codecell>
# FUNC ANIMATION

trail_map = trail_class(**trail_args)
env = TrailEnv(trail_map, discrete=global_discrete, treadmill=global_treadmill)
# model = PPO.load('trained/epoch_2/width10_break_jan29.zip', device='cpu')

obs = env.reset()
frames = [env.agent.position_history[:]]
for _ in range(100):
    action, _ = model.predict(obs, deterministic=False)
    obs, _, is_done, _ = env.step(action)
    frames.append(env.agent.position_history[:])

    if is_done:
        print('reach end')
        break

def plot_frame(frame):
    env.map.plot(ax=plt.gca())
    plt.plot(*zip(*frame), linewidth=2, color='black')

ani = FuncAnimation(plt.gcf(), plot_frame, frames=frames)
ani.save('out.gif')

# %%
# DECISION VISUALS
# model = PPO.load('trail_model.zip', device='cpu')
pi = model.policy
all_actions = torch.arange(model.action_space.n)

trail_map = trail_class(**trail_args, heading=np.pi/3)
env = TrailEnv(trail_map, discrete=global_discrete, treadmill=global_treadmill)

@torch.no_grad()
def plot_probs(obs, ax):
    obs_t, _ = pi.obs_to_tensor(obs)
    value, probs, _ = pi.evaluate_actions(obs_t, all_actions)

    ax.bar(['left', 'forward', 'right'], np.exp(probs), alpha=0.7)
    return value

class VisualPolicy(torch.nn.Module):
    def __init__(self, pi):
        super().__init__()
        self.pi = pi

    def forward(self, obs):
        _, probs, _ = self.pi.evaluate_actions(obs, all_actions)
        return probs.unsqueeze(0)


cam = GradCAM(
    model=VisualPolicy(pi),
    target_layers=[pi.features_extractor.cnn[5]]
)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, is_done, _ = env.step(action)
    print(reward)

    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    ax1 = plt.subplot(121)
    ax1.imshow(obs)

    obs_t, _ = pi.obs_to_tensor(obs)
    grayscale_cam = cam(input_tensor=obs_t, target_category=None)
    ax1.imshow(grayscale_cam[0], alpha=0.5, cmap='bone')

    ax2 = plt.subplot(122)
    value = plot_probs(obs, ax2)

    ax1.set_title(f'State value: {value.numpy()[0,0]:.2f}')

    fig.tight_layout()
    plt.show()

    if is_done:
        print('done')
        break

env.map.plot(ax=plt.gca())
plt.plot(*zip(*env.agent.position_history), linewidth=2, color='black')
plt.savefig('out.png')
# %%
