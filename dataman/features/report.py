import logging

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = 15, 8

logger = logging.getLogger(__name__)

# disable font_manager spamming the debug log
# logging.getLogger('matplotlib').disabled = True
logging.getLogger('matplotlib.fontmanager').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True


# def plot_waveforms_grid(wv, n_rows=10, n_cols=20, n_overlay=10000):
#     n_overlay = min(n_overlay, wv.shape[2])
#
#     target_wv = wv[:, :, np.linspace(0, wv.shape[2] - 1, n_rows * n_cols, dtype='int64')]
#     max_amplitude = target_wv.max(axis=(0, 1, 2))
#     min_amplitude = target_wv.min(axis=(0, 1, 2))
#
#     fig = plt.figure(figsize=(28, 10))
#
#     # gridspec inside gridspec
#     outer_grid = gridspec.GridSpec(1, 2, wspace=0.0, hspace=0.0, width_ratios=[20, 8])
#     inner_grid = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=outer_grid[0], wspace=0.0, hspace=0.0)
#
#     # waveform plots
#     for nr in range(n_rows):
#         for nc in range(n_cols):
#             n = nc + nr * n_cols
#             ax = plt.Subplot(fig, inner_grid[n])
#             ax.axis('off')
#             ax.plot(target_wv[:, :, n], linewidth=1)
#             ax.set_ylim(min_amplitude, max_amplitude)
#             fig.add_subplot(ax)
#
#     target_wv = wv[:, :, np.linspace(0, wv.shape[2] - 1, n_overlay, dtype='int64')]
#     ch_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[1], wspace=0.0, hspace=0.0)
#     for nr in range(2):
#         for nc in range(2):
#             n = nc + nr * 2
#             ax = plt.Subplot(fig, ch_grid[n])
#             ax.plot(target_wv[:, n, :], linewidth=1, c=f'C{n}', alpha=.02)
#             ax.axis('off')
#             ax.set_ylim(min_amplitude, max_amplitude)
#             fig.add_subplot(ax)
#
#     return fig


def plot_feature(feature, feature_name='', timestamps=None):
    num_features = feature.shape[1]
    if num_features > 16:
        raise ValueError("More than 16 features found, indicating wrong array shape.")

    fig, ax = plt.subplots(2, num_features, figsize=(16, 6))
    yq_l, yq_h = np.quantile(feature[:, 0], .1), np.quantile(feature[:, 0], .9)

    for n, num_fet in enumerate(range(0, num_features)):
        lim_l, lim_h = np.quantile(feature[:, num_fet], .1), np.quantile(feature[:, num_fet], .9)
        if n:
            ax[0, n].scatter(feature[:, 0], feature[:, num_fet], s=.2, alpha=.1)
            ax[0, n].set_title('{fn}:{n} / {fn}:0'.format(fn=feature_name, n=num_fet))
            ax[0, n].set_xlim(yq_l, yq_h)
            ax[0, n].set_ylim(lim_l, lim_h)
        else:
            # TODO: PLOT EVENTS PER SECOND
            ax[0, n].plot(timestamps)

        ax[1, n].scatter(timestamps, feature[:, num_fet], s=.2, alpha=.1)
        ax[1, n].set_ylim(lim_l, lim_h)

    return fig


def plot_raw(arr, fs=3e4):
    amplitude_correction = 0.195
    fig, ax = plt.subplots(figsize=(18, 4))

    # Limit offset and width in seconds of data we are going to plot
    dur = int(.5 * fs)
    ofs = int(0 * fs)
    t = np.linspace(ofs, ofs + dur, dur) / fs

    # Plot, but add a fixed value to each channel to separate them for display
    ax.plot(t, arr[ofs:ofs + dur, :] * amplitude_correction + np.linspace(0, 4 * 200, 4))
    ax.set_xlabel('$Seconds$')
    ax.set_ylabel('$\mu V$')

    return fig
