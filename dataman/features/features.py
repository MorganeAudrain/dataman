import argparse
import logging
import time
from itertools import combinations
from pathlib import Path

import h5py
import hdf5storage as h5s
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from dataman.lib.report import fig2html, ds_shade_waveforms, ds_plot_waveforms, ds_shade_feature, ds_plot_features
from dataman.lib.util import run_prb
from scipy.ndimage import gaussian_filter

PRECISION = np.dtype(np.single)
N_SAMPLES = 32
N_CHANNELS = 4
sampling_rate = 3e4

logger = logging.getLogger(__name__)

# disable font_manager spamming the debug log
# logging.getLogger('matplotlib').disabled = True
logging.getLogger('matplotlib.fontmanager').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True

# Available features
# TODO: per feature CLI arguments
# TODO: feature discovery as modules
AVAILABLE_FEATURES = ['energy', 'energy24', 'cpca', 'cpca24', 'chwpca']


def scale_feature(fet):
    nfet = fet - np.mean(fet, axis=0)
    std = np.std(nfet, axis=0)
    if not std.all():
        logging.warning('Zeros in standard deviation of feature normalizaton, setting those to 1.0')
        std[std == 0] = 1
    nfet /= std
    return nfet


def feature_peak(wv, minimum=True, absolute=False):
    """Calculate per-channel peak amplitude."""
    if minimum:
        return wv.max(axis=0).T

    if absolute:
        return wv.abs().max(axis=0).T

    if absolute and minimum:
        raise NotImplementedError('Minimum and absolute do not work together.')


def feature_energy(wv):
    """Calculate l2 (euclidean) norm for vector containing spike waveforms
    """
    return np.sqrt(np.sum(wv ** 2, axis=0)).T


def feature_energy24(wv):
    """Calculate l2 (euclidean) norm for vector containing spike waveforms for the first 24 samples
    """
    return np.sqrt(np.sum(wv[2:22, :, :] ** 2, axis=0)).T


def feature_weighted_energy(wv, dropoff=.1, s_pre=8, s_post=16):
    """Calculate l2 (euclidean) norm for vectors of spike waveform, where
    waveform amplitudes are weighted towards the peak/detection center at dropoff rate"""
    kernel = np.ones(s_pre + s_post)
    raise NotImplemented('This feature has not been implemented yet.')


def feature_position(pos_file, dat_offsets, timestamps, indices, sampling_rate=3e4):
    raise NotImplemented('This feature has not been implemented yet.')
    # pos_f = h5py.File(pos_file, 'r')
    # positions = np.array(pos_f['XY_data']).T
    # av_pos = (np.nanmax(positions) - np.nanmin(positions)) / 2
    #
    # n_frames = len(positions)
    # n_records = [(nb - 1024) / 2070 for nb in n_bytes]
    # assert not any([nr % 1 for nr in n_records])
    # n_samples = [int(nr * 1024) for nr in n_records]
    #
    # starts = [sum(n_samples[:n]) for n in range(len(n_samples))]
    # s_duration = [ns / sampling_rate for ns in n_samples]
    # fps = n_frames / s_duration[1]
    #
    # idx_vid_start = np.nonzero(indices > starts[1])[0].min()
    # idx_vid_end = np.nonzero(indices > starts[2])[0].min() - 1
    #
    # t_pos = np.zeros(len(timestamps))
    # t_vel = np.zeros_like(t_pos)
    # t_acc = np.zeros_like(t_pos)
    #
    # t_pos[np.squeeze(timestamps) < starts[1] / 3] = -1
    # t_pos[np.squeeze(timestamps) > starts[2] / 3] = -1
    #
    # for idx in range(idx_vid_start, idx_vid_end):
    #     vid_index = int((indices[idx] - n_samples[0]) * fps // sampling_rate)
    #
    #     t_pos[idx] = positions[vid_index]
    #
    # # fill nans by interpolation
    #
    # t_vel[1:] = np.diff(t_pos)
    # t_acc[1:] = np.diff(t_vel)
    #
    # return np.vstack([t_pos, t_pos, t_vel, t_acc]).T


def feature_cPCA(wv, n_components=12, incremental=False, batch_size=None):
    """Concatenated PCA. Uses concatenated channels."""
    if incremental:
        raise NotImplementedError("Can't run incremental PCA yet.")

    ers = np.reshape(np.transpose(wv, axes=(1, 0, 2)), (N_SAMPLES * N_CHANNELS, -1))
    pca = PCA(n_components)
    scores = pca.fit_transform(ers.T)
    return scores


def feature_cPCA24(wv, n_components=12, incremental=False, batch_size=None):
    """Concatenated PCA. Uses first 24 samples of concatenated channels."""
    if incremental:
        raise NotImplementedError("Can't run incremental PCA yet.")

    ers = np.reshape(np.transpose(wv[:24, :, :], axes=(1, 0, 2)), (24 * N_CHANNELS, -1))
    pca = PCA(n_components)
    scores = pca.fit_transform(ers.T)
    return scores


def feature_chwPCA(wv, dims=3, energy_normalize=True):
    """ Channel wise (normalized) PCA
    waveforms shape is (nSamples x nChannels x nSpikes)
    """
    pcas = []
    pca_scores = []
    for d in range(4):
        pca = PCA(n_components=dims)
        data = wv[:, d, :].T.astype('float64').copy()
        # data_s = data - np.mean(data, axis=0)  # this messes things up?!
        if energy_normalize:
            l2 = feature_energy(data.T)[:, np.newaxis]

            # With dead channels we end up with zero-energy waveforms sometimes, resulting in division by zero.
            zero_energy_waveforms = (l2 == 0).nonzero()
            if zero_energy_waveforms[0].shape[0]:
                logger.warning(
                    'Found {} instances of zero-energy waveforms in channel {}. Settings those to energy=1.0'.format(
                        zero_energy_waveforms[0].shape[0], d))
                l2[zero_energy_waveforms] = 1.0

            # normaliz
            # e all waveforms by their l2 norm/energy
            data /= l2

        scores = pca.fit_transform(data)
        if np.isnan(scores).any():
            logger.warning('NaN in PCA scores, setting those to 0.0')
            scores.nan_to_num(0)
        pca_scores.append(scores)
        pcas.append(pca)
    pca_scores = np.concatenate(pca_scores, axis=1)
    return pca_scores


def write_features_fet(feature_data, outpath):
    # TODO: Channel validity
    start = time.time()
    fet_data = np.hstack(list(feature_data))
    logger.debug('hstack fet data in {:.2f} s'.format(time.time() - start))
    logging.info("fet_data is {} MB, shape: {}".format(fet_data.nbytes / 1e6, fet_data.shape))

    start = time.time()
    with open(outpath, 'w') as fet_file:
        fet_file.write(f' {fet_data.shape[1]}\n')

        np.savetxt(fet_file, fet_data, fmt='%-16.8f', delimiter=' ')
    logger.debug('Wrote fet file in {:.2f} s'.format(time.time() - start))

    # # faster variant, but only saves ~7 seconds/1M spikes @ 32 features
    # start = time.time()
    # with open(outpath.with_suffix('.test'), 'w') as fet_file:
    #     fet_file.write(f' {fet_data.shape[1]}\n')
    #     fmt = ' '.join(['%g'] * fet_data.shape[1])
    #     fmt = '\n'.join([fmt] * fet_data.shape[0])
    #     fet_string = fmt % tuple(fet_data.ravel())
    #     fet_file.write(fet_string)
    # logger.debug('VARIANT: Wrote fet file in {:.2f} s'.format(time.time() - start))


def write_feature_fd(feature_names, feature_data, timestamps, outpath, tetrode_path, channel_validity=None):
    if type(feature_names) == str:
        feature_names = [feature_names]

    feature_name_ch = ['{}: {}'.format(fn, n + 1) for fn in feature_names for n in range(feature_data.shape[1])]
    feature_std = feature_data.std(axis=0, dtype='double')
    feature_av = feature_data.mean(axis=0, dtype='double')

    if channel_validity is None:
        channel_validity = [1, 1, 1, 1]

    if len(feature_names) > 1:
        raise NotImplementedError('No logic for multiple features yet. External or just .fd?')

    outpath_fname = outpath / '{}_{}.fd'.format(tetrode_path.stem, feature_names[0])

    h5s.savemat(outpath_fname, {'ChannelValidity': np.array(channel_validity, dtype='double'),  # dead channel index
                                'FD_av': feature_av,  # mean
                                'FD_sd': feature_std,  # sd
                                'FeatureData': feature_data,
                                'FeatureIndex': np.arange(1, feature_data.shape[0] + 1, dtype='double'),
                                'FeatureNames': feature_name_ch,
                                'FeaturePar': [],
                                'FeaturesToUse': feature_names,
                                'FeatureTimestamps': timestamps,
                                'TT_file_name': str(tetrode_path.name),
                                }, compress=False, truncate_existing=True, truncate_invalid_matlab=True,
                appendmat=False)

def main(args):
    parser = argparse.ArgumentParser('Generate .fet and .fd files for features from spike waveforms')
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose (debug) output")

    parser.add_argument('target', default='.', help="""Directory with waveform .mat files.""")
    parser.add_argument('-o', '--out_path', help='Output file path Defaults to current working directory')
    parser.add_argument('--sampling-rate', type=float, help='Sampling rate. Default 30000 Hz', default=3e4)
    parser.add_argument('-f', '--force', action='store_true', help='Force overwrite of existing files.')
    parser.add_argument('-a', '--align', help='Alignment method, default: min', default='min')
    parser.add_argument('-F', '--features', nargs='*', help='Features to use. Default: energy', default=['energy'])
    parser.add_argument('--to_fet', nargs='*', help='Features to include in fet file, default: all', default='all')
    parser.add_argument('--ignore-prb', action='store_true',
                        help='Do not load channel validity from dead channels in .prb files')
    parser.add_argument('--no-report', action='store_true', help='Do not generate report file (saves time)')
    cli_args = parser.parse_args(args)

    matpath = Path(cli_args.target).resolve()
    if matpath.is_file():
        matfiles = [matpath]
    else:
        matfiles = sorted(list(map(Path.resolve, matpath.glob('tetrode??.mat'))))

    logger.debug(f'Target files: {[mf.name for mf in matfiles]}')
    logger.info('Found {} waveform files'.format(len(matfiles)))
    logger.debug(f'Requested to fet: {cli_args.to_fet}')

    # TODO:
    # per feature arguments
    sigma = 0.8

    for nt, matfile in tqdm(enumerate(matfiles), total=len(matfiles)):
        outpath = matfile.parent / 'FD'
        if not outpath.exists():
            outpath.mkdir()

        # Load prb file if it exists and set channel validity based on dead channels
        prb_path = matfile.with_suffix('.prb')
        if prb_path.exists():
            prb = run_prb(prb_path)
        else:
            logger.warning(f'No probe file found for {matfile.name} and no channel validity given.')
            prb = None
        if prb is None or 'dead_channels' not in prb:
            channel_validity = [1, 1, 1, 1]
        else:
            channel_validity = [int(ch not in prb['dead_channels']) for ch in prb['channel_groups'][0]['channels']]
        logger.debug('Channel validity: {}'.format(channel_validity) + ('' if all(
            channel_validity) else f', {4 - sum(channel_validity)} dead channel(s)'))

        hf = h5py.File(matfile, 'r')

        waveforms = np.array(hf['spikes'], dtype=PRECISION).reshape([N_SAMPLES, N_CHANNELS, -1])

        timestamps = np.array(hf['index'], dtype='double')
        gauss= gaussian_filter(waveforms,sigma)
        # indices = timestamps * sampling_rate / 1e4

        features = {}
        # Allow to calculate all available features
        if len(cli_args.features) == 1 and cli_args.features[0].lower() == 'all':
            cli_args.features = AVAILABLE_FEATURES

        for fet_name in map(str.lower, cli_args.features):
            if fet_name == 'energy':
                logger.debug(f'Calculating {fet_name} feature')
                features['energy'] = scale_feature(feature_energy(waveforms))

            elif fet_name == 'energy24':
                logger.debug(f'Calculating {fet_name} feature')
                features['energy24'] = scale_feature(feature_energy24(waveforms))

            elif fet_name == 'peak':
                logger.debug(f'Calculating {fet_name} feature')
                features['peak'] = feature_peak(waveforms)

            elif fet_name == 'cpca':
                logging.debug(f'Calculating {fet_name} feature')
                cpca = scale_feature(feature_cPCA(waveforms))
                logger.debug('cPCA shape {}'.format(cpca.shape))
                features['cPCA'] = cpca

            elif fet_name == 'cpca24':
                logging.debug(f'Calculating {fet_name} feature')
                cpca24 = scale_feature(feature_cPCA24(waveforms))
                logger.debug('cPCA24 shape {}'.format(cpca24.shape))
                features['cPCA24'] = cpca24

            elif fet_name == 'chwpca':
                logging.debug(f'Calculating {fet_name} feature')
                chwpca = scale_feature(feature_chwPCA(waveforms))
                logger.debug('chwPCA shape {}'.format(chwpca.shape))
                features['chwPCA'] = chwpca

            else:
                raise NotImplementedError("Unknonw feature: {}".format(fet_name))

        # TODO:
        # fet_cpca_4 = fet_cpca[:, :4]

        # # Position feature
        # n_bytes = [250154314, 101099824, 237970294]
        # fet_pos = feature_position(matpath / 'XY_data.mat', dat_offsets=n_bytes, timestamps=timestamps,
        #                            indices=indices)

        # Generate .fet file used for clustering
        # TODO: Best move this out into the cluster module?
        if 'none' in map(str.lower, cli_args.to_fet):
            logger.warning('Skipping fet file generation')
        else:
            fet_file_path = outpath / matfile.with_suffix('.fet.0').name

            if len(cli_args.to_fet) == 1 and cli_args.to_fet[0].lower() == 'all':
                logger.debug('Writing all features to fet file.')
                included_features = list(map(str.lower, features.keys()))
            else:
                included_features = [fn for fn in map(str.lower, features.keys()) if
                                     fn in list(map(str.lower, cli_args.to_fet))]

            logger.info(f'Writing features {list(included_features)} to .fet')
            fet_data = [fd for fn, fd in features.items() if fn.lower() in included_features]

            logger.debug(f'Writing .fet file {fet_file_path}')
            write_features_fet(feature_data=fet_data, outpath=fet_file_path)

        # Write .fd file for each feature
        for fet_name, fet_data in features.items():
            logger.debug(f'Writing feature {fet_name}.fd file')
            write_feature_fd(feature_names=fet_name, feature_data=fet_data,
                             timestamps=timestamps, outpath=outpath, tetrode_path=matfile,
                             channel_validity=channel_validity)

        logger.debug('Generating waveform graphic')
        with open(matfile.with_suffix('.html'), 'w') as frf:
            frf.write('<head></head><body><h1>{}</h1>'.format(matfile.name))

            frf.write('<h2>Waveforms (n={})</h2>'.format(waveforms.shape[2]))
            density_agg = 'log'

            with np.errstate(invalid='ignore'):  # ignore some matplotlib colormap usage errors
                images = ds_shade_waveforms(waveforms, how=density_agg)
            fig = ds_plot_waveforms(images, density_agg)
            frf.write(fig2html(fig) + '</br>')
            del fig

            for fet_name, fet_data in features.items():
                frf.write('<h3>Feature: {}</h3>\n'.format(fet_name))

                df_fet = pd.DataFrame(fet_data)

                # numerical column names are an issue with datashader, stringify 'em
                df_fet.rename(columns={k: str(k) for k in df_fet.columns}, inplace=True)
                df_fet['time'] = timestamps

                fet_columns = df_fet.columns[:-1]

                # Features vs. features
                images = []
                titles = []
                for cc in list(combinations(fet_columns, 2)):
                    fet_title = f'{fet_name}:{cc[1]} vs {fet_name}:{cc[0]}'
                    logger.debug(f'plotting feature {fet_title}')

                    # Calculate display limits, try to exclude outliers
                    # TODO: correct axis labeling
                    perc_lower = 0.05
                    perc_upper = 99.9
                    x_range = (np.percentile(df_fet[cc[0]], perc_lower), np.percentile(df_fet[cc[0]], perc_upper))
                    y_range = (np.percentile(df_fet[cc[1]], perc_lower), np.percentile(df_fet[cc[1]], perc_upper))
                    with np.errstate(invalid='ignore'):
                        shade = ds_shade_feature(df_fet[[cc[0], cc[1]]], x_range=x_range, y_range=y_range,
                                                 color_map='inferno')
                    images.append(shade)
                    titles.append(fet_title)

                fet_fig = ds_plot_features(images, how='log', fet_titles=titles)
                frf.write(fig2html(fet_fig) + '</br>\n')
                del fet_fig

                # Features over time
                t_images = []
                t_titles = []
                x_range = (0, df_fet['time'].max())

                # Calculate display limits, try to exclude outliers
                # TODO: correct axis labeling
                perc_lower = 0.1
                perc_upper = 99.9
                y_range = (np.percentile(df_fet[cc[1]], perc_lower), np.percentile(df_fet[cc[1]], perc_upper))

                for cc in fet_columns:
                    t_title = f'{fet_name}:{cc} vs. time'
                    logger.debug(f'plotting {t_title}')
                    with np.errstate(invalid='ignore'):
                        shade = ds_shade_feature(df_fet[['time', cc]], x_range=x_range, y_range=y_range,
                                                 color_map='viridis')
                    t_images.append(shade)
                    t_titles.append(t_title)

                t_fig = ds_plot_features(t_images, how='log', fet_titles=t_titles)
                frf.write(fig2html(t_fig) + '</br>\n')
                del t_fig

                frf.write('</hr>\n')

                # np.save('{}.npy'.format(fet_name), fet_data)


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
