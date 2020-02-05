# generate .fet file from .fd files
#           OR
# if those features are missing, generate them with dm features!
# - generate/load features
# - run klustakwik
# - read clusters, build initial clustering report
import argparse
import logging
import shutil
from pathlib import Path
from itertools import combinations

import pkg_resources
import yaml
import subprocess

import platform

import pandas as pd
import datashader as ds  # slow to import, takes ~4 s
from datashader import transfer_functions as ds_tf
import numpy as np
import matplotlib.pyplot as plt
import hdf5storage as h5s

from dataman.lib.report import fig2html, ds_plot_features

logger = logging.getLogger(__name__)


def load_yaml(yaml_path):
    """Load content of a .yaml file as dictionary
    """
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError('Invalid YAML path {}'.format(yaml_path))

    with open(yaml_path, 'r') as yf:
        return yaml.load(yf, Loader=yaml.SafeLoader) or {}


def main(args):
    parser = argparse.ArgumentParser('Clustering with KlustaKwik')
    parser.add_argument('target', help='Target path, either path containing tetrode files, or single tetrodeXX.mat')
    parser.add_argument('--KK', help='Path to KlustaKwik executable')
    parser.add_argument('--features', nargs='*', help='list of features to use for clustering')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--cluster', action='store_true', help='Directly run ')
    parser.add_argument('--force', help='Overwrite existing files.', action='store_true')
    parser.add_argument('--skip', help='Skip if clu file exists already', action='store_true')
    parser.add_argument('--no_spread', help='Shade report plots without static spread', action='store_true')
    cli_args = parser.parse_args(args)

    # Load default configuration yaml file
    default_cfg_path = Path(pkg_resources.resource_filename(__name__, '../resources/cluster_defaults.yml')).resolve()
    if not default_cfg_path.exists():
        logging.error('Could not find default config file.')
        raise FileNotFoundError
    logger.debug('Loading default configuration')
    cfg = load_yaml(default_cfg_path)

    # Load local config file if it exists
    local_cfg_path = Path(
        pkg_resources.resource_filename(__name__, '../resources/cluster_defaults_local.yml')).resolve()
    if local_cfg_path.exists():
        logger.debug('Loading and updating with local configuration')
        local_cfg = load_yaml(local_cfg_path)
        cfg.update(local_cfg)

    # Load custom config path
    custom_cfg_path = Path(cli_args.config).resolve() if cli_args.config else None
    if custom_cfg_path:
        if custom_cfg_path.exists():
            logger.debug('Loading and updating with custom configuration')
            cfg.update(load_yaml(custom_cfg_path))
        else:
            raise FileNotFoundError(f"Could not load configuration file {custom_cfg_path}")

    # Load parameters from command line
    logger.debug('Parsing and updating configuration with CLI arguments')
    cfg.update(vars(cli_args))
    logger.debug(cfg)

    # try to find Klustakwik executable if necessary...
    if cli_args.KK is None:
        cli_args.KK = shutil.which('KlustaKwik') or shutil.which('klustakwik') or shutil.which('Klustakwik')
    if cli_args.KK is None and cli_args.cluster:
        raise FileNotFoundError('Could not find the KlustaKwik executable on the path, and none given.')

    # # Building KlustaKwik Command
    # # 1) Find KlustaKwik executable
    # mclust_path = Path('C:/Users/reichler/src/MClustPipeline/MClust/KlustaKwik')
    # pf_system = platform.system()
    # logger.debug(f'Platform: {pf_system}')
    # if pf_system == 'Linux':
    #     kk_executable = mclust_path / cfg['KLUSTAKWIK_PATH_LINUX']
    # elif pf_system == 'Windows':
    #     kk_executable = mclust_path / cfg['KLUSTAKWIK_PATH_WINDOWS']
    # else:
    #     raise NotImplemented(f'No KlustaKwik executable defined for platform {pf_system}')
    # logger.debug(kk_executable)

    kk_executable = cli_args.KK

    # 2) Find target file stem
    working_dir = Path(cli_args.target).resolve()
    if working_dir.is_file() and working_dir.exists():
        tetrode_files = [working_dir.name]
        working_dir = working_dir.parent
        logger.debug(f'Using single file mode with {str(tetrode_files[0])}')
    else:
        tetrode_files = sorted([tf.name for tf in working_dir.glob(cfg['TARGET_FILE_GLOB'])])

    logger.debug(f'Working dir: {working_dir}')

    # No parallel/serial execution supported right now
    if len(tetrode_files) > 1:
        raise NotImplemented('Currently only one target file per call supported!')
    logger.debug(f'Target found: {tetrode_files}')

    tetrode_file_stem = str(tetrode_files[0]).split(".")[0]
    tetrode_file_elecno = tetrode_files[0].split(".")[-1]

    # 3) Check if output file already exists
    clu_file = (working_dir / tetrode_file_stem).with_suffix(f'.clu.{tetrode_file_elecno}')
    if clu_file.exists() and not (cli_args.force or cli_args.skip):
        raise FileExistsError('Clu file already exists. Use --force to overwrite.')

    # 4) combine executable and arguments
    kk_cmd = f'{kk_executable} {tetrode_file_stem} -ElecNo {tetrode_file_elecno}'
    kk_cmd_list = kk_cmd.split(' ')
    logger.debug(f'KK COMMAND: {kk_cmd}')
    logger.debug(f'KK COMMAND LIST: {kk_cmd_list}')

    # Call KlustaKwik and gather output
    # TODO: Use communicate to interact with KK, i.e. write to log and monitor progress
    #       see https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
    logger.info('Starting KlustaKwik process')

    if cfg['PRINT_KK_OUTPUT']:
        stdout = subprocess.STDOUT
    else:
        stdout = subprocess.PIPE

    # EXECUTE KLUSTAKWIK
    if not clu_file.exists() or cli_args.force:
        kk_call = subprocess.run(kk_cmd_list, stderr=subprocess.PIPE, stdout=stdout)
        kk_error = kk_call.returncode

        logger.debug('Writing KlustaKwik log file')
        with open(clu_file.with_suffix('.log'), 'w') as log_file:
            log_file.write(kk_call.stderr.decode('ascii'))

        # Check call return code and output
        if kk_error:
            logging.error(f'KlustaKwik error code: {kk_error}')
            exit(kk_error)
        else:
            logging.debug(f'KlustaKwik successful: {kk_error}')

    # Load clu file
    logger.debug(f'Loading {clu_file}')
    clu_df = pd.read_csv(clu_file, dtype='category', names=['cluster_id'], skiprows=1)
    cluster_labels = clu_df['cluster_id'].cat.categories
    num_clusters = len(cluster_labels)
    logger.info(f'{len(clu_df)} spikes in {num_clusters} clusters')

    # Find all feature .fd files
    feature_files = list(working_dir.glob(tetrode_file_stem + '_*.fd'))
    ff_sizes = [ff.stat().st_mtime for ff in feature_files]
    feature_files = [f for t, f in sorted(zip(ff_sizes, feature_files))]
    if not len(feature_files):
        raise FileNotFoundError(f'No Feature Files found in {working_dir}')

    # TODO: Stupid, the feature names are in the .fd file already
    feature_names = [str(ff.name).split(tetrode_file_stem + '_')[1].split('.')[0] for ff in feature_files]
    logger.info(f'Loading features: {feature_names}')

    color_keys = cfg['CLUSTER_COLORS']
    with open(clu_file.with_suffix('.html'), 'w') as crf:
        crf.write('<head></head><body><h1>{}</h1>'.format(clu_file.name))
        for fd_file, fet_name in zip(feature_files, feature_names):
            crf.write('<h3>Feature: {}</h3>\n'.format(fet_name))
            logger.info(f'Generating images for feature {fet_name}')
            if not fd_file.exists():
                continue

            logger.debug(f'Loading {fd_file}')
            mat_fet = h5s.loadmat(str(fd_file), appendmat=False)

            fd_df = pd.DataFrame(mat_fet['FeatureData'])
            fd_df.rename(columns={c: str(c) for c in fd_df.columns}, inplace=True)
            if not len(clu_df) == len(fd_df):
                raise ValueError(f'Number of cluster labels ({num_clusters}) does not match number of spikes'
                                 f'in {fd_file} ({len(fd_df)})')

            fd_df['clu_id'] = clu_df.cluster_id.astype('category')
            logger.debug(f'Feature {fet_name} loaded with {len(fd_df)} spikes, {fd_df.shape[1] - 1} dimensions ')

            images = []
            titles = []
            for cc in combinations(map(str, range(len(fd_df.columns) - 1)), r=2):
                fet_title = f'{fet_name}:{cc[1]} vs {fet_name}:{cc[0]}'
                x_range = (np.percentile(fd_df[cc[0]], 0.01), np.percentile(fd_df[cc[0]], 99.9))
                y_range = (np.percentile(fd_df[cc[1]], 0.01), np.percentile(fd_df[cc[1]], 99.9))

                logger.debug(f'shading {len(fd_df)} points in {fd_df.shape[1] - 1} dimensions')
                canvas = ds.Canvas(plot_width=300, plot_height=300, x_range=x_range, y_range=y_range)
                agg = canvas.points(fd_df, x=cc[0], y=cc[1], agg=ds.count_cat('clu_id'))
                with np.errstate(invalid='ignore'):
                    img = ds_tf.shade(agg, how='log', color_key=color_keys)
                    img = img if cli_args.no_spread else ds_tf.spread(img, px=1)
                    images.append(img)
                titles.append(fet_title)

            logger.debug(f'Creating plot for {fet_name}')
            fet_fig = ds_plot_features(images, how='log', fet_titles=titles)
            crf.write(fig2html(fet_fig) + '</br>\n')
            plt.close(fet_fig)


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
