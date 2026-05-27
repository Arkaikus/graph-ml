"""Visual spot-check utilities for earthquake sliding windows."""

import math
import random
from typing import Any

import matplotlib.gridspec as gridspec
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


_FEATURE_COLS = [
    "magnitude",
    "depth_km",
    "time_days",
    "delta_t_days",
    "delta_mag",
    "delta_lat",
    "delta_lon",
]


def _plot_single_window(
    outer_gs_cell: gridspec.SubplotSpec,
    fig: Figure,
    x_num: torch.Tensor,
    y: torch.Tensor,
    window_idx: int,
    global_sample_idx: int,
) -> None:
    """
    Render one window into a subdivided cell of the outer GridSpec.

    Draws three stacked mini-plots:
    - Magnitude sequence (blue) + target value (red star)
    - Depth (km) sequence (green)
    - Inter-event time deltas (orange bars)

    Parameters
    ----------
    outer_gs_cell : gridspec.SubplotSpec
        A single cell from the outer 4x4 GridSpec.
    fig : plt.Figure
        The parent figure.
    x_num : torch.Tensor
        Shape (seq_len, num_features) — raw (pre-standardization) features.
    y : torch.Tensor
        Shape (1,) — target magnitude for this window.
    window_idx : int
        0-based index within the 4x4 grid (for subplot numbering).
    global_sample_idx : int
        Index of this sample in the original samples list.
    """
    seq_len = x_num.shape[0]
    target_mag = y.item()

    magnitudes = x_num[:, 0].numpy()
    depths = x_num[:, 1].numpy()
    delta_t = x_num[:, 3].numpy()

    inner_gs = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer_gs_cell, hspace=0.05
    )

    ax_mag = fig.add_subplot(inner_gs[0])
    ax_dep = fig.add_subplot(inner_gs[1], sharex=ax_mag)
    ax_dt = fig.add_subplot(inner_gs[2], sharex=ax_mag)

    # -- Magnitude --
    ax_mag.plot(magnitudes, color="steelblue", linewidth=0.9)
    ax_mag.axhline(target_mag, color="red", linewidth=0.7, linestyle="--", alpha=0.6)
    ax_mag.plot(seq_len - 0.5, target_mag, marker="*", color="red", markersize=6)
    ax_mag.set_ylabel("Mag", fontsize=6, labelpad=2)
    ax_mag.tick_params(labelbottom=False, labelsize=5)
    ax_mag.set_title(
        f"#{global_sample_idx}  (len={seq_len}, target={target_mag:.2f})",
        fontsize=6,
        pad=2,
    )

    # -- Depth --
    ax_dep.plot(depths, color="seagreen", linewidth=0.9)
    ax_dep.set_ylabel("Dep", fontsize=6, labelpad=2)
    ax_dep.tick_params(labelbottom=False, labelsize=5)

    # -- Inter-event delta_t --
    xs = range(seq_len)
    ax_dt.bar(xs, delta_t, color="darkorange", width=0.8, alpha=0.8)
    ax_dt.set_ylabel("Δt", fontsize=6, labelpad=2)
    ax_dt.tick_params(labelsize=5)

    for ax in (ax_mag, ax_dep, ax_dt):
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.yaxis.set_tick_params(width=0.5)
        ax.xaxis.set_tick_params(width=0.5)


def plot_window_grid(
    samples: list[dict[str, Any]],
    num_samples: int = 16,
    figsize: tuple[int, int] = (20, 20),
    seed: int = 42,
) -> Figure:
    """
    Create a grid of randomly selected window visualizations for spot-checking.

    The grid is as close to 4x4 as possible. If fewer than ``num_samples``
    windows are available, all are shown in a proportionally smaller grid.
    Each cell contains three stacked mini-plots: magnitude sequence, depth,
    and inter-event time delta.

    Parameters
    ----------
    samples : list[dict[str, Any]]
        Windows from ``make_windows`` with keys ``"gh_ids"``, ``"x_num"``, ``"y"``.
    num_samples : int
        Target number of windows to display (default 16 → 4x4 grid).
    figsize : tuple[int, int]
        Overall figure size in inches.
    seed : int
        RNG seed for reproducible random selection.

    Returns
    -------
    Figure
        An Agg-rendered figure; call ``fig.savefig(path)`` then discard.
        Uses the Agg backend directly so it works regardless of whatever
        global matplotlib backend is active (e.g. mpl_ascii in training).
    """
    rng = random.Random(seed)

    actual_n = min(num_samples, len(samples))
    selected_indices = sorted(rng.sample(range(len(samples)), actual_n))
    selected = [(idx, samples[idx]) for idx in selected_indices]

    # Determine grid dimensions: prefer square-ish layout up to 4 columns
    ncols = min(4, actual_n)
    nrows = math.ceil(actual_n / ncols)

    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)  # attach canvas so savefig works
    fig.suptitle(
        f"Window Spot Check — {actual_n} randomly sampled windows "
        f"(seed={seed})",
        fontsize=10,
        y=0.995,
    )

    outer_gs = gridspec.GridSpec(
        nrows, ncols, figure=fig, hspace=0.45, wspace=0.25
    )

    for grid_idx, (sample_idx, sample) in enumerate(selected):
        row = grid_idx // ncols
        col = grid_idx % ncols
        _plot_single_window(
            outer_gs_cell=outer_gs[row, col],
            fig=fig,
            x_num=sample["x_num"],
            y=sample["y"],
            window_idx=grid_idx,
            global_sample_idx=sample_idx,
        )

    return fig
