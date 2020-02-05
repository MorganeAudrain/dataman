import base64
import io
import logging
import urllib

import datashader as ds
import numpy as np
import pandas as pd
from datashader import transfer_functions as tf
from matplotlib import pyplot as plt, image as mpimg, cm

from dataman.detect.report import DS_CMAPS

plt.rcParams['figure.figsize'] = 15, 8

logger = logging.getLogger(__name__)

# disable font_manager spamming the debug log
# logging.getLogger('matplotlib').disabled = True
logging.getLogger('matplotlib.fontmanager').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True


def fig2html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    string = base64.b64encode(buf.read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "{}"/>'.format(uri)
    return html


def ds_shade_to_html(shade):
    img_string = base64.b64encode(shade.to_bytesio(format='png').read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(img_string)
    html = '<img src = "{}"/>'.format(uri)
    return html


def ds_plot_waveforms(shades, how, y_min_uv=-500, y_max_uv=400):
    """Plot of datashader rasterized images of waveforms.
    """
    fig, axes = plt.subplots(1, len(shades), figsize=(16, 5))
    fig.suptitle('All waveforms, density "{}" scaled.'.format(how))
    for n, ax in enumerate(axes):
        # cast rastered shade into png image as workaround to false color application by imshow??
        # TODO: That can't be the best way to go about this!!
        img_arr = mpimg.imread(shades[n].to_bytesio('png'))
        h, w = img_arr.shape[:2]

        ax.imshow(img_arr)
        ax.set_title(f'Channel {n}')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: '{:.0f}'.format(x / w * 32)))
        if not n:
            # replace axis ticks of image with actual amplitude scale in microvolts
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, p: '{:.0f}'.format(((-y / h * (y_max_uv - y_min_uv)) + y_max_uv))))
        else:
            ax.set_yticklabels([])
    plt.tight_layout()
    return fig


def ds_shade_waveforms(waveforms, canvas_width=400, canvas_height=400, color_maps=None, y_min_uv=-500, y_max_uv=400,
                       how='log'):
    n_samples = waveforms.shape[0]
    n_channels = waveforms.shape[1]

    if color_maps is None:
        color_maps = DS_CMAPS

    if n_channels > len(color_maps):
        color_maps *= len(color_maps) // n_channels + 1

    cvs = ds.Canvas(plot_height=canvas_height, plot_width=canvas_width,
                    x_range=(0, waveforms.shape[0] - 1),
                    y_range=(y_min_uv / 0.195, y_max_uv / 0.195))

    shades = []
    for ch in range(n_channels):
        df = pd.DataFrame(waveforms[:, ch, :].T)
        agg = cvs.line(df, x=np.arange(n_samples), y=list(range(n_samples)), agg=ds.count(), axis=1)
        with np.errstate(invalid='ignore'):
            img = tf.shade(agg, how=how, cmap=plt.cm.get_cmap(color_maps[ch]))
        shades.append(img)

    # If we wanted to use all channels together, we have to create a categorical column for the selection
    # df = pd.DataFrame(wv_reshaped)
    # df['channel'] = pd.Categorical(np.tile(np.arange(n_channels), wv_reshaped.shape[0]//n_channels))
    # wv_reshaped = waveforms.reshape(n_samples, -1).T
    # agg = cvs.line(df, x=np.arange(n_samples), y=list(range(32)), agg=ds.count_cat('channel'), axis=1)

    return shades


def ds_shade_feature(fet_data, x_range=(-2, 8), y_range=(-2, 8),
                     canvas_width=300, canvas_height=300, color_map='viridis', how='log'):
    """Agg and shade a single feature view. fet_data needs to be two-columnar pandas DataFrame."""
    cmap = cm.get_cmap(color_map)
    try:
        cvs = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height, x_range=x_range, y_range=y_range)
        agg = cvs.points(fet_data, fet_data.columns[0], fet_data.columns[1], agg=ds.count())
        with np.errstate(invalid='ignore'):
            img = tf.shade(agg, how=how, cmap=cmap)
    except ZeroDivisionError:
        logger.warning(f'Zero Division Error in ds_shade_feature for columns {tuple(fet_data.columns)}. Invalid plot.')
        img = None
    return img


# def ds_shade_feature(fet_data, timestamps, x_range=(-2, 8), y_range=(-2, 8),
#                      canvas_width=400, canvas_height=400, color_map='viridis', how='log'):
#
#     df = pd.DataFrame(fet_data)
#     df.rename(columns={k: str(k) for k in df.columns}, inplace=True)  # because numerical column names don't work well
#     df['time'] = timestamps
#     cmap = cm.get_cmap(color_map)
#
#     cols = df.columns
#     cvs = ds.Canvas(plot_width=canvas_width, plot_height=canvas_height, x_range=x_range, y_range=y_range)
#     images = []
#     for cc in list(combinations(cols[:-2], 2)):
#         agg = cvs.points(df, cc[0], cc[1], agg=ds.count())
#         images.append(tf.shade(agg, how=how, cmap=cmap))
#
#     # show features over time
#     cvs = ds.Canvas(plot_width=300, plot_height=150, y_range=(-2, 8))
#     t_images = []
#     for c in cols[:-2]:
#         agg = cvs.points(df, 't', c, agg=ds.count())
#         t_images.append(tf.shade(agg, how=how, cmap=cmap))
#
#     return images, t_images

def ds_plot_features(shades, how, fet_titles, y_min=-500, y_max=400, plot_width=16, plot_height=3, max_cols=6):
    """Plot of datashader rasterized images of waveforms.
    """
    n_cols = min(len(shades), max_cols)
    n_rows = max(len(shades) // n_cols, 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(plot_width, plot_height + 2 * n_rows))
    for n, ax in enumerate(axes.flat):
        # cast rasterized shade into png image as workaround to false color application by imshow??
        # TODO: That can't be the best way to go about this!!
        if shades[n] is None:
            logger.warning(f'ds_plot_feature received "None" datashader shades[{n}], replacing with zeros array.')
            img_arr = np.zeros((400, 400, 3), dtype='uint8')
        else:
            img_arr = mpimg.imread(shades[n].to_bytesio('png'))
        h, w = img_arr.shape[:2]

        ax.imshow(img_arr)
        ax.set_title(f'{fet_titles[n]}', fontsize=8)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: '{:.0f}'.format(x / w * 32)))
        if not n:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, p: '{:.0f}'.format(((-y / h * (y_max - y_min)) + y_max))))
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='major', labelsize=5)
    plt.tight_layout()
    return fig
