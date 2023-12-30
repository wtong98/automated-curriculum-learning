"""
Various trail maps and adventures:

author: William Tong
"""

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


class MeanderTrail(TrailMap):
    def __init__(self, length=50, 
                       width=3, 
                       breaks : List[Tuple[float, float]] = [],   # list of tuple (start_break, end_break) proportions
                       range=(-np.pi / 4, np.pi / 4), 
                       heading=None,
                       reward_dist=10, 
                       res=25, radius=100, diff_rate=0.04, local_len=1):
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

    
    def plot(self, res=200, ax=None, x_lim=None, y_lim=None, auto_aspect=True, margin=20, width=6, with_colorbar=True):
        if x_lim is None:
            x_lim = np.min(self.x_coords) - margin, np.max(self.x_coords) + margin
        if y_lim is None:
            y_lim = np.min(self.y_coords) - margin, np.max(self.y_coords) + margin

        x = np.linspace(*x_lim, res)
        y = np.linspace(*y_lim, res)
        xx, yy = np.meshgrid(x, y)

        odors = np.array([self.sample(x, y) for x, y in zip(xx.ravel(), yy.ravel())])
        odors = odors.reshape(res, res)

        if ax is None:
            ax = plt.gca()

        ax.plot(self.x_coords, self.y_coords, linewidth=1.5, color='red', alpha=0.5)
        mbp = ax.contourf(x, y, odors, cmap='Greens', alpha=0.3)

        if auto_aspect:
            fig = plt.gcf()

            ratio = (y_lim[1] - y_lim[0] + 40) / (x_lim[1] - x_lim[0] + 40)
            height = width * ratio

            fig.set_size_inches((width, height))
            fig.tight_layout()
        
        root = self.x_coords[-1] - 1.5, self.y_coords[-1] - 1.5
        rect = plt.Rectangle(root, 3, 3, color='darkmagenta', fill=False)

        if with_colorbar:
            fmt = lambda x, _: '{:.2f}'.format(x)
            cb = plt.colorbar(mbp, format=FuncFormatter(fmt), aspect=30)
            cb.ax.tick_params(labelsize=20)
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

        if return_rate:
            return rate
        else:
            samp = np.random.poisson(rate) / 50
            return samp.item()

    def plot(self, ax=None, x_lim=(-30, 30), y_lim=(-60, 10), auto_aspect=True, with_colorbar=True):
        x = np.linspace(*x_lim, 100)
        y = np.linspace(*y_lim, 100)
        xx, yy = np.meshgrid(x, y)

        odors_contours = np.array([self.sample(px, py, return_rate=True) for px, py in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)

        if auto_aspect:
            fig = plt.gcf()

            ratio = (y_lim[1] - y_lim[0] + 40) / (x_lim[1] - x_lim[0] + 40)
            height = 6 * ratio

            fig.set_size_inches((6, height))
            fig.tight_layout()
        
        lvls = 1 / np.linspace(15, 0.05, 18)
        if ax is None:
            ax = plt.gca()

        mbp = ax.contourf(x, y, odors_contours, levels=lvls, cmap='Greens', alpha=0.7, norm=colors.LogNorm(vmin=min(lvls), vmax=max(lvls)))

        circle = plt.Circle((0, 0), radius=1, color='green', zorder=100, alpha=0.9)
        rect = plt.Rectangle([-1.5, -1.5], 3, 3, color='darkmagenta', fill=False)
        ax.add_patch(circle)
        ax.add_patch(rect)

        if with_colorbar:
            fmt = lambda x, _: '{:.2f}'.format(x)
            cb = plt.colorbar(mbp, format=FuncFormatter(fmt))
            cb.ax.tick_params(labelsize=16)

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

