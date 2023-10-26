"""
Various trail maps and adventures:

author: William Tong
"""

# <codecell>
from typing import List, Tuple
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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


# TODO: scale length with max_steps <-- IMPORTANT
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

    
    def plot(self, res=200, ax=None, x_lim=None, y_lim=None, auto_aspect=True):
        if x_lim is None:
            x_lim = np.min(self.x_coords) - 20, np.max(self.x_coords) + 20
        if y_lim is None:
            y_lim = np.min(self.y_coords) - 20, np.max(self.y_coords) + 20

        x = np.linspace(*x_lim, res)
        y = np.linspace(*y_lim, res)
        xx, yy = np.meshgrid(x, y)

        odors = np.array([self.sample(x, y) for x, y in zip(xx.ravel(), yy.ravel())])
        odors = odors.reshape(res, res)

        if ax is None:
            ax = plt.gca()

        ax.plot(self.x_coords, self.y_coords, linewidth=1.5, color='red', alpha=0.5)
        ax.contourf(x, y, odors, cmap='Greens', alpha=0.3)

        circle = plt.Circle((self.x_coords[0], self.y_coords[0]), radius=1, color='blue', zorder=100, alpha=0.7)

        if auto_aspect:
            fig = plt.gcf()

            ratio = (y_lim[1] - y_lim[0] + 40) / (x_lim[1] - x_lim[0] + 40)
            height = 6 * ratio

            fig.set_size_inches((6, height))
            fig.tight_layout()
        
        root = self.x_coords[-1] - 1.5, self.y_coords[-1] - 1.5
        rect = plt.Rectangle(root, 3, 3, color='darkmagenta', fill=False)

        # ax.add_patch(circle)
        ax.add_patch(rect)

        
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


class PlumeTrail(TrailMap):
    def __init__(self, start=None,
                 start_rate=None,
                 start_rate_range=None,
                 start_dist=None,
                 heading=None,
                 range=(-np.pi/4, np.pi/4),
                 diffusivity=1,
                 emission_rate=1,
                 particle_lifetime=150,
                 wind_speed=1,
                 sensor_size=1,
                 length_scale=10,
                 distance_factor=3,
                 max_steps=200):
        
        self.D = diffusivity * length_scale ** 2
        self.R = emission_rate
        self.tau = particle_lifetime
        self.V = wind_speed * length_scale
        self.a = sensor_size * length_scale

        self.scale = np.sqrt((self.D * self.tau) / (1 + (self.V ** 2 * self.tau) / (4 * self.D)))
        self.base_rate = self.a * self.R

        self.distance_factor = distance_factor
        
        self.range = range
        if heading != None:
            self.range = (heading, heading)
        self.heading = np.random.uniform(*self.range)

        self.start_rate = start_rate
        self.start_rate_range = start_rate_range
        self.start_dist = start_dist

        if start_rate:
            start = self._sample_point(start_rate)
        elif start_rate_range:
            lo, hi = start_rate_range
            inv_rate = np.random.uniform(1 / hi, 1 / lo)
            start = self._sample_point(1 / inv_rate)
        elif start_dist:
            start = start_dist * np.array((-np.sin(self.heading), -np.cos(self.heading)))
        super().__init__(start, end=np.array((0,0)))

        if max_steps == 'auto':
            dist = np.linalg.norm(self.start - self.end)
            self.max_steps = np.round(3 * dist)
        else:
            self.max_steps = max_steps

    def _sample_point(self, rate):
        y_max, y_min = np.real(self._compute_y(rate))
        uppr = y_max
        if self.distance_factor:
            uppr = y_min * (1 - 1 / self.distance_factor)
        y = np.random.uniform(y_min, uppr)
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
            samp = np.random.poisson(rate) / 50  # TODO: or binary?
            return samp.item()

    def plot(self, ax=None, x_lim=(-30, 30), y_lim=(-60, 10), auto_aspect=True):
        # if x_lim is None:
        #     x_lim = np.min(self.x_coords) - 20, np.max(self.x_coords) + 20
        # if y_lim is None:
        #     y_lim = np.min(self.y_coords) - 20, np.max(self.y_coords) + 20

        x = np.linspace(*x_lim, 100)
        y = np.linspace(*y_lim, 100)
        xx, yy = np.meshgrid(x, y)

        odors_contours = np.array([self.sample(px, py, return_rate=True) for px, py in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)
        odors_samples = np.array([self.sample(px, py, return_rate=False) for px, py in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)

        if auto_aspect:
            fig = plt.gcf()

            ratio = (y_lim[1] - y_lim[0] + 40) / (x_lim[1] - x_lim[0] + 40)
            height = 6 * ratio

            fig.set_size_inches((6, height))
            fig.tight_layout()
        
        lvls = 1 / np.linspace(15, 0.05, 18)
        print('LVLS', lvls)
        if ax is None:
            ax = plt.gca()

        ax.plot(*self.start, marker='o', markersize=5, color='black')
        mbp = ax.contourf(x, y, odors_contours, levels=lvls, cmap='Greens', alpha=0.7, norm=colors.LogNorm(vmin=min(lvls), vmax=max(lvls)))
        # ax.contourf(x, y, odors_contours, cmap='viridis', alpha=1)
        # ax.contour(x, y, odors_contours, cmap='Reds', alpha=0.5, levels=lvls)

        circle = plt.Circle((0, 0), radius=1, color='green', zorder=100, alpha=0.9)
        rect = plt.Rectangle([-1.5, -1.5], 3, 3, color='darkmagenta', fill=False)
        ax.add_patch(circle)
        ax.add_patch(rect)

        # TODO: replot, and trail too
        fmt = lambda x, _: '{:.2f}'.format(x)
        plt.colorbar(mbp, format=FuncFormatter(fmt))

    def reset(self):
        self.heading = np.random.uniform(*self.range)
        if self.start_rate != None:
            self.start = self._sample_point(self.start_rate)
        elif self.start_rate_range:
            lo, hi = self.start_rate_range
            inv_rate = np.random.uniform(1 / hi, 1 / lo)
            self.start = self._sample_point(1 / inv_rate)
        elif self.start_dist != None:
            self.start = self.start_dist * np.array((-np.sin(self.heading), -np.cos(self.heading)))
    
    def __str__(self) -> str:
        return f'PlumeTrail(start={self.start}  rate={self.start_rate})'
    
    def __repr__(self) -> str:
        return self.__str__()


if __name__ == '__main__':
    # TODO: prettify plume trail plotting
    trail = PlumeTrail(heading=0, wind_speed=5, start_rate_range=[0.3, 0.3], length_scale=20, max_steps='auto')
    # trail = MeanderTrail(heading=0, length=200, width=5, reward_dist=-1)
    # trail.reset()

    trail.reset()
    trail.plot()
    plt.xticks([])
    plt.yticks([])

    print(trail.start)
    # plt.savefig('example_trail.svg')

# %%
    print(trail.sample(0, -30, return_rate=True))

    # for _ in range(10):
    #     print(trail.sample(0, -15) * 256)

    # trail.base_rate
