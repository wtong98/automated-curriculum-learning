"""
Various trail maps and adventures:

author: William Tong
"""

# <codecell>
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw


class TrailMap:
    def __init__(self, start=None, end=None):
        self.start = start if type(start) != type(None) else np.array([0, 0])
        self.end = end if type(end) != type(None) else np.array([0, 0])
        self.tol = 3

    def sample(self, x, y):
        """ Returns odor on scale from 0 to 1 """
        raise NotImplementedError('sample not implemented!')

    def plot(self):
        raise NotImplementedError('plot not implemented!')

    def reset(self):
        raise NotImplementedError('reset not implemented!')

    def is_done(self, x, y):
        is_done = np.all(np.isclose(self.end, (x, y), atol=self.tol))
        return bool(is_done)
    
    def is_at_checkpoint(self, x, y):
        return False


class StraightTrail(TrailMap):
    def __init__(self, end=None, narrow_factor=1):
        super().__init__()
        self.end = end if type(end) != type(None) else np.array([10, 15])
        self.narrow_factor = narrow_factor

    def sample(self, x, y):
        eps = 1e-8
        total_dist = np.sqrt((x - self.end[0]) ** 2 + (y - self.end[1]) ** 2)
        perp_dist = np.abs((self.end[0] - self.start[0]) * (self.start[1] - y) - (self.start[0] - x) *
                           (self.end[1] - self.start[1])) / np.sqrt((self.start[0] - self.end[0]) ** 2 + (self.start[1] - self.end[1])**2 + eps)

        max_odor = np.sqrt(np.sum((self.end - self.start) ** 2)) + 1
        odor = max_odor - total_dist
        odor *= 1 / (perp_dist + 1) ** self.narrow_factor

        odor = np.clip(odor, 0, np.inf)
        return odor / max_odor

    def plot(self, ax=None):
        x = np.linspace(-20, 20, 100)
        y = np.linspace(-20, 20, 100)
        xx, yy = np.meshgrid(x, y)

        odors = self.sample(xx, yy)

        if ax:
            return ax.contourf(x, y, odors)
        else:
            plt.contourf(x, y, odors)
            plt.colorbar()

    def reset(self):
        pass


class RandomStraightTrail(StraightTrail):
    def __init__(self, is_eval=False, **kwargs):
        super().__init__(**kwargs)
        self.eval = is_eval
        self.next_choice = 0

        self.end = self._rand_coords()
        self.tol = 4

    def _rand_coords(self):
        # branches = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i, j) != (0, 0)]
        # branches = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0)]
        branches = [(-1, 1), (0, 1), (1, 1)]

        if self.eval:
            idx = self.next_choice
            self.next_choice = (self.next_choice + 1) % len(branches)
        else:
            idx = np.random.choice(len(branches))

        x_fac, y_fac = branches[idx]
        new_end = np.array([x_fac * 15, y_fac * 15])

        # print(new_end)
        return new_end

    def reset(self):
        self.end = self._rand_coords()


class RoundTrail(TrailMap):
    def __init__(self):
        super().__init__()
        self.end = np.array([10, 15])

    def sample(self, x, y):
        # if not isinstance(x, np.ndarray):
        #     x = np.array([x])
        #     y = np.array([y])
        max_odor = 100
        odor = - 0.1 * ((x - self.end[0]) ** 2 + (y - self.end[1]) ** 2) + max_odor
        odor = np.clip(odor, 0, np.inf)
        return odor / max_odor

    def plot(self):
        xx, yy = np.meshgrid(
            np.linspace(-20, 20, 100),
            np.linspace(-20, 20, 100))

        odors = self.sample(xx, yy)
        odors = np.flip(odors, axis=0)
        plt.imshow(odors)

    def reset(self):
        pass


class RandomRoundTrail(RoundTrail):
    def __init__(self):
        super().__init__()
        self.end = self._rand_coords()

    def _rand_coords(self):
        new_end = np.random.randint(10, 16, 2) * np.random.choice([-1, 1], 2)
        return new_end

    def reset(self):
        self.end = self._rand_coords()


class TrainingTrailSet(TrailMap):  # TODO: test
    def __init__(self, trails: List[TrailMap]):
        super().__init__()
        self.trails = trails
        self.curr_trail = self._get_rand_trail()

    def _get_rand_trail(self):
        rand_idx = np.random.randint(len(self.trails))
        return self.trails[rand_idx]

    def sample(self, x, y):
        return self.curr_trail.sample(x, y)

    def plot(self):
        for trail in self.trails:
            trail.plot()

    def reset(self):
        self.curr_trail = self._get_rand_trail()


# TODO: scale length with max_steps
class MeanderTrail(TrailMap):
    def __init__(self, length=50, 
                       width=3, 
                       breaks : List[Tuple[float, float]] = [],   # list of tuple (start_break, end_break) proportions
                       range=(-np.pi / 4, np.pi / 4), 
                       heading=None,
                       reward_dist=10, 
                       res=25, radius=100, diff_rate=0.04, local_len=1, is_eval=False):
        super().__init__(start=None, end=None)
        self.T = length
        self.res = res
        self.xi = radius
        self.k = diff_rate
        self.lamb = local_len

        if heading != None:
            self.range = (heading, heading)
        else:
            self.range = range

        self.reward_dist = reward_dist
        self.width = width
        self.breaks = breaks

        self.x_coords, self.y_coords, self.checkpoints = self._sample_trail()
        self.end = np.array([self.x_coords[-1], self.y_coords[-1]])
        
    
    def _sample_trail(self):
        dt = self.lamb / self.res
        n_samps = int(self.T / dt)
        x, y, theta, K = np.zeros((4, n_samps))
        D = 1 / (self.lamb * self.xi ** 2)
        K[0] = np.sqrt(D * self.lamb) * np.random.randn()

        ckpts = []
        ckpt_len = int(self.reward_dist / dt)

        for i in range(n_samps - 1):
            if self.reward_dist != -1:
                if i % ckpt_len == 0 and i != 0:
                    ckpts.append((x[i], y[i]))

            x[i + 1] = x[i] + dt * np.cos(theta[i])
            y[i + 1] = y[i] + dt * np.sin(theta[i])

            # heading update
            theta[i + 1] = theta[i]  \
                + dt * K[i] \
                + np.sqrt(self.k * dt) * np.random.randn()
            
            # curvature update (Ornstein - Uhlenbeck process)
            K[i + 1] = K[i] \
                - dt * K[i] \
                + np.sqrt(D * dt) * np.random.randn()
        
        total_len = len(x)
        break_mask = np.ones(total_len, dtype=bool)
        for start_prop, end_prop in self.breaks:
            break_mask[int(total_len * start_prop): int(total_len * end_prop)] = False
        
        # rotate to net heading
        net_heading = np.random.uniform(*self.range)
        ang = np.arctan(y[-1] / x[-1])
        if x[-1] < 0:
            ang += np.pi

        rot = (np.pi / 2) - net_heading - ang
        rot_mat = np.array([
            [np.cos(rot), -np.sin(rot)],
            [np.sin(rot), np.cos(rot)]
        ])

        points = np.stack((x, y), axis=0)[:, break_mask]
        net_x, net_y = rot_mat @ points

        if len(ckpts) > 0:
            ckpts = np.array(ckpts) @ rot_mat.T
        
        return net_x, net_y, ckpts

    
    def sample(self, x, y):
        dist2 = (self.x_coords - x) ** 2 + (self.y_coords - y) ** 2
        return np.exp(- np.min(dist2) / self.width)

    
    def plot(self, res=200, ax=None, xmin=-120, xmax=120, ymin=-120, ymax=120):
        x = np.linspace(xmin, xmax, res)
        y = np.linspace(ymin, ymax, res)
        xx, yy = np.meshgrid(x, y)

        odors = np.array([self.sample(x, y) for x, y in zip(xx.ravel(), yy.ravel())])
        odors = odors.reshape(res, res)

        ckpt_x, ckpt_y = self.checkpoints.T if len(self.checkpoints) > 0 else (0, 0)

        if ax:
            ax.plot(self.x_coords, self.y_coords, linewidth=3, color='red', alpha=0.5)
            ax.contourf(x, y, odors)
            ax.scatter(ckpt_x, ckpt_y, color='red')

            circle = plt.Circle((self.x_coords[0], self.y_coords[0]), radius=3, color='blue', zorder=100, alpha=0.7)
            return ax.add_patch(circle)
        else:
            plt.gcf().set_size_inches(8, 8)
            plt.plot(self.x_coords, self.y_coords, linewidth=3, color='red', alpha=0.5)
            plt.contourf(x, y, odors)
            plt.scatter(ckpt_x, ckpt_y, color='red')
            plt.colorbar()

            circle = plt.Circle((self.x_coords[0], self.y_coords[0]), radius=3, color='blue', zorder=100, alpha=0.7)
            plt.gca().add_patch(circle)


    def reset(self):
        self.x_coords, self.y_coords, self.checkpoints = self._sample_trail()
        self.end = np.array([self.x_coords[-1], self.y_coords[-1]])
    

    def is_at_checkpoint(self, x, y):
        for i, ckpt in enumerate(self.checkpoints):
            if np.all(np.isclose((x, y), ckpt, atol=self.tol)):
                # self.checkpoints = np.delete(self.checkpoints, i, axis=0)
                self.checkpoints = self.checkpoints[i+1:]
                return True

        return False
    

    def __str__(self) -> str:
        return f'(len={self.T}, width={self.width})'


class BrokenMeanderTrail(MeanderTrail):
    def __init__(self, exp_breaks=0, 
                       exp_len=0, 
                       max_break_dist=0.9, 
                       trail_length=50, **kwargs):
        
        # self.exp_breaks = exp_breaks
        # forcing breaks
        self.exp_breaks = 1
        self.exp_len = exp_len
        self.max_break_dist = max_break_dist
        self.T = trail_length
        self._reset_breaks()

        super().__init__(length=trail_length, breaks=self.breaks, **kwargs)

    
    def _reset_breaks(self):
        num_breaks = np.random.poisson(self.exp_breaks)
        lens = np.random.exponential(self.exp_len, num_breaks)

        # forcing to break at half-way point
        # starts = np.random.random(num_breaks)
        starts = np.array([0.5])
        # ends = starts + lens / self.T
        # ends = starts + lens[0] / self.T if len(lens) > 0 else []
        ends = np.array([0.6])

        self.breaks = [pair for pair in zip(starts, ends) if pair[1] < self.max_break_dist]

    def reset(self):
        self._reset_breaks()
        super().reset()


# NOTE: wind speed fixed along direction of negative y axis
class PlumeTrail(TrailMap):
    def __init__(self, start=None,
                 start_rate=None,
                 heading=None,
                 range=(-np.pi/4, np.pi/4),
                 diffusivity=1,
                 emission_rate=1,
                 particle_lifetime=150,
                 wind_speed=1,
                 sensor_size=1,
                 length_scale=10,
                 max_steps=200):
        
        self.D = diffusivity * length_scale ** 2
        self.R = emission_rate
        self.tau = particle_lifetime
        self.V = wind_speed * length_scale
        self.a = sensor_size * length_scale

        self.scale = np.sqrt((self.D * self.tau) / (1 + (self.V ** 2 * self.tau) / (4 * self.D)))
        self.base_rate = self.a * self.R

        
        self.range = range
        if heading != None:
            self.range = (heading, heading)
        self.heading = np.random.uniform(*self.range)

        if start_rate:
            self.y_max, self.y_min = np.real(self._compute_y(start_rate))
            start = self._sample_point(start_rate)
            self.start_rate = start_rate
        super().__init__(start, end=np.array((0,0)))

        if max_steps == 'auto':
            assert start_rate != None
            self.max_steps = np.round(3 * - self.y_min)
        else:
            self.max_steps = max_steps
    

    # TODO: sample uniformly across whole level set
    def _sample_point(self, rate):
        y = np.random.uniform(self.y_min, self.y_max)
        dist = np.real(self._compute_dist(rate, y))
        x = np.sqrt(dist ** 2 - y ** 2)

        ang = -self.heading
        rot = np.array([[np.cos(ang), -np.sin(ang)],
                        [np.sin(ang),  np.cos(ang)]])

        if np.random.uniform() > 0.5:
            return rot @ np.array((x, y))
        else:
            return rot @ np.array((-x, y))

    def _compute_dist(self, rate, y):
        arg = self.base_rate / (self.scale * rate) * np.exp(-(y * self.V) / (2 * self.D))
        val = self.scale * lambertw(arg)
        return val
    
    def _compute_y(self, rate):
        fac_plus = (1 / self.scale) + self.V / (2 * self.D)
        fac_minus = (1 / self.scale) - self.V / (2 * self.D)

        y_plus = lambertw(fac_plus * self.base_rate / rate) / fac_plus
        y_minus = lambertw(fac_minus * self.base_rate / rate) / fac_minus
        return y_plus, -y_minus
    
    def _sample_wind_vec(self):
        return np.array([-np.sin(self.heading), -np.cos(self.heading)]).reshape(-1, 1)

    def sample(self, x, y, return_rate=False):
        dist = np.linalg.norm(self.end - [x, y]) 
        dist_factor = np.exp(- dist / self.scale)

        wind_dist = self._sample_wind_vec().T @ (self.end - np.array((x, y)))
        wind_factor = np.exp(-wind_dist * self.V / (2 * self.D))
        rate = (self.base_rate / dist) * wind_factor * dist_factor

        # print('DIST', dist)
        # print('DIST_FAC', dist_factor)
        # print('WIND', wind_factor)
        # print('RATE', rate)

        if return_rate:
            return rate
        else:
            return np.random.poisson(rate) / self.base_rate  # TODO: or binary?

    def plot(self, ax=None, x_lim=(-30, 30), y_lim=(-50, 10)):
        x = np.linspace(*x_lim, 100)
        y = np.linspace(*y_lim, 100)
        xx, yy = np.meshgrid(x, y)

        odors_contours = np.array([self.sample(px, py, return_rate=True) for px, py in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)
        odors_samples = np.array([self.sample(px, py, return_rate=False) for px, py in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)

        lvls = np.arange(0, 1, 0.05)
        if ax != None:
            ax.plot(*self.start, marker='o', markersize=10, color='blue')
            ax.contourf(x, y, odors_samples, levels=lvls)
            ax.contour(x, y, odors_contours, cmap='Reds', alpha=0.1, levels=lvls)
        else:
            plt.plot(*self.start, marker='o', markersize=10, color='blue')
            plt.contourf(x, y, odors_samples, levels=lvls)
            plt.colorbar()
            plt.contour(x, y, odors_contours, cmap='Reds', alpha=0.1, levels=lvls)

    def reset(self):
        self.heading = np.random.uniform(*self.range)
        if self.start_rate != None:
            self.start = self._sample_point(self.start_rate)
    
    def __str__(self) -> str:
        return f'PlumeTrail(start={self.start}  rate={self.start_rate})'
    
    def __repr__(self) -> str:
        return self.__str__()


if __name__ == '__main__':
    # trail = PlumeTrail(range=(-np.pi, np.pi), heading=None, wind_speed=5, start_rate=0.32, length_scale=20, max_steps='auto')
    trail = MeanderTrail(heading=0, length=200, width=5, reward_dist=-1)
    # trail.reset()

# <codecell>
    trail.plot(ymin=-10, ymax=150, xmin=-30, xmax=60)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('example_trail.svg')
# %%
