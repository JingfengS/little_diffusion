from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import Axes
from typing import Optional
import torch
import numpy as np
from .core import Sampleable, Density

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def hist2d_samples(samples, ax: Optional[Axes] = None, bins: int = 200, scale: float = 5.0, percentile: int = 99, **kwargs):
    """绘制样本的 2D 直方图"""
    if ax is None: ax = plt.gca()
    samples = samples.detach().cpu().numpy()
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=[[-scale, scale], [-scale, scale]])
    cmax = np.percentile(H, percentile)
    norm = cm.colors.Normalize(vmax=cmax, vmin=0.0)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin='lower', norm=norm, **kwargs)

def hist2d_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    """直接从分布对象绘制直方图"""
    assert sampleable.dim == 2
    samples = sampleable.sample(num_samples)
    hist2d_samples(samples, ax, **kwargs)

def scatter_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    """绘制散点图"""
    if ax is None: ax = plt.gca()
    samples = sampleable.sample(num_samples).detach().cpu()
    ax.scatter(samples[:,0], samples[:,1], **kwargs)

def contour_density(density: Density, bins: int = 100, scale: float = 5.0, ax: Optional[Axes] = None, x_offset: float = 0.0, **kwargs):
    """绘制概率密度等高线"""
    if ax is None: ax = plt.gca()
    x = torch.linspace(-scale + x_offset, scale + x_offset, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    d = density.log_density(xy).reshape(bins, bins)
    ax.contour(X.cpu().numpy(), Y.cpu().numpy(), d.detach().cpu().numpy(), **kwargs)