import argparse
import logging
import os
from datetime import datetime as dt
from pathlib import Path

import h5py as h5
import hdf5storage as h5s
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from scipy import signal
from tqdm import tqdm

import dataman.lib.report
from dataman.detect import report
from dataman.lib.util import butter_bandpass

logger = logging.getLogger(__name__)

# disable font_manager spamming the debug log
# logging.getLogger('matplotlib').disabled = True
logging.getLogger('matplotlib.fontmanager').disabled = True

MINIMUM_NOISE_THRESHOLD = 5


def get_batches(length, batch_size):
    """Given length of e.g. an array and a batch size, return size of batches
    """
    batches = [bc * batch_size for bc in range(length // batch_size)]
    if length - (length // batch_size) * batch_size:
        batches.append(length - length % batch_size)
    return batches


def estimate_noise(arr, lc=300, hc=6000, num_channels=4, fs=3e4, microvolt_factor=0.195, ne_bin_s=1):
    """Calulate MAD (mean absolute deviation) of high pass filtered array.
    Returns list of bin-sized estimates in uV.
    """
    ne_bin_size = int(ne_bin_s * fs)  # noise estimation bin size

    # Filter
    b, a = butter_bandpass(lc, hc, fs)
    batches = get_batches(arr.shape[0], ne_bin_size)
    ne = np.zeros((len(batches), num_channels))
    nfac = 1 / 0.6745

    # Calculate MAD (mean absolute deviation) over chunks
    for n, batch in enumerate(tqdm(batches, leave=False, desc='1) estimating')):
        batch_size = min(ne_bin_size, arr.shape[0] - batch)
        filtered = signal.filtfilt(b, a, arr[batch:batch + batch_size, :].astype(np.double), axis=0) * microvolt_factor
        for ch in range(num_channels):
            ne[n, ch] = np.median(abs(filtered[:, ch]) * nfac)
    return ne


def wv_com(arr):
    """Center of mass of abs(wv)."""
    cmass = np.cumsum(np.abs(arr))
    mass = cmass.max()
    return np.argmin(np.abs(cmass - mass / 2))


def wv_alignment(w, method='centroid', kernel_width=11, interp_f=None):
    """Given approximate timestamp of a spike, find the accurate peak position of the interpolated signal
    given an alignment method. Available methods are 'min' for minimum and 'centroid' for centroid filter.
    """
    # channel with highest amplitude
    ch = np.min(w, axis=0).argmin()
    y = w[:, ch]

    # Minimum alignment filter, i.e. peak negative amplitude
    if method == 'min':
        if not interp_f:
            m = np.argmin(y)
            return m, ch, None
        else:
            raise NotImplementedError

    # Centroid alignment method
    # Convolve the waveform with a linear filter and find zero crossing
    elif method == 'centroid':
        # TODO: Use channel energy for weighted decisions?
        centroid_kernel = np.arange(-kernel_width // 2 + 1, kernel_width // 2 + 1)

        # center of mass
        com = wv_com(y)

        # centroid filter
        cvlv = np.convolve(y, centroid_kernel, mode='same')

        # zero crossing after centroid filter
        zxr = np.diff(np.signbit(cvlv)).nonzero()[0]

        # crossing closest to center of mass
        closest_xr = zxr[np.argmin(np.abs(zxr - com))]

        return closest_xr, ch, cvlv

    else:
        raise NotImplementedError


def detect_spikes(arr, min_thresholds, max_sd=18, fs=3e4, chunk_size_s=60, chunk_overlap_s=0.05, lc=300, hc=6000,
                  s_pre=10, s_post=22, reject_overlap=16, align='min'):
    """Given wideband signal, find peaks (minima) in the high-pass filtered signal. Returns a list of
    curated timestamps to reject duplicates and overlapping spikes.
    """
    # TODO: Interpolation
    # TODO: Maximum artifact rejection
    # TODO: Return rejected timestamps

    chunk_size = int(chunk_size_s * fs)  # chunk size for detection

    microvolt_factor = 0.195
    use_thr = min_thresholds / microvolt_factor

    timestamps = []
    crs = []

    max_thr = use_thr * max_sd
    if max_thr is not None or max_thr != 0:
        logger.warning('Maximum rejection for spike detection not implemented.')
    # # waveform_chunks = []
    # rejections = 0

    b, a = butter_bandpass(lc, hc, fs)

    # samples to cut around detection (threshold crossing)
    bc_samples = np.arange(-s_pre, s_post).reshape([1, -1])

    # Chunks will have partial overlap.
    # 0
    # |o|--- chunk 1 ---|x|
    #                 |o|--- chunk 2 ---|x|
    #                                |o|--- chunk 3 ---|
    #                                                end
    # Spikes with peak in the |x| region will be ignored
    chunk_overlap = int(chunk_overlap_s * fs)  # 50 ms chunk boundary overlap

    # Gather chunk start and ends
    end = arr.shape[0]
    chunk_starts = [cs * chunk_size for cs in range(ceil(end / chunk_size))]

    for n_chunk, start in enumerate(tqdm(chunk_starts, leave=False, desc='2) detecting')):
        # limits of core batch chunk
        b_start = start
        b_end = min(start + chunk_size, end)

        # limits of chunk with flanking overlaps
        o_start = max(0, b_start - chunk_overlap)
        o_end = min(b_end + chunk_overlap, end)

        # Bandpass filter raw signal
        filtered = signal.filtfilt(b, a, arr[o_start:o_end], axis=0)

        # Merge threshold crossings
        # TODO: Only merge valid channels!
        crossings = np.clip(np.sum(filtered < -use_thr, axis=1), 0, 1).astype(np.int8)
        crossings = signal.medfilt(crossings, 3)

        # exclude crossings with timestamps too close to boundaries
        xr_starts = (np.diff(crossings) > 0).nonzero()[0]
        xr_starts = xr_starts[(xr_starts > s_pre) & (xr_starts < filtered.shape[0] - s_post)]

        # Warning if no spikes were found. That's suspicious given how we calculate the threshold.
        n_spikes = xr_starts.shape[0]
        if not n_spikes:
            logger.warning(f'No spikes in chunk {n_chunk} @ [{b_start} to {b_end}]')
            continue

        # Extract preliminary waveforms
        # get waveform indices by broadcasting indices of crossings
        bc_spikes = xr_starts.reshape([n_spikes, 1])
        idc = bc_samples + bc_spikes

        try:
            wv = filtered[idc]
        except IndexError:
            logger.error(f'Spikes out of bounds in chunk {n_chunk} @ [{b_start} to {b_end}]')
            break

        # Alignment
        alignments = np.array([wv_alignment(wv[wv_idx, :], method=align)[0] for wv_idx in range(wv.shape[0])])
        wv_starts = xr_starts + alignments + o_start - s_pre

        # first chunk, no overlap
        lim_a = 0 if n_chunk == 0 else b_start
        not_early = wv_starts >= lim_a

        # last chunk, no overlap
        lim_b = end if n_chunk == len(chunk_starts) else b_end

        not_late = wv_starts < lim_b - 32

        timestamps.append(wv_starts[not_early * not_late])
        crs.append(xr_starts[not_early * not_late] + o_start)

    timestamps = np.sort(np.concatenate(timestamps))

    ts_diff = np.diff(timestamps) > reject_overlap

    too_close = timestamps.shape[0] - np.sum(ts_diff)
    logger.warning('{} ({:.1f}%) spikes rejected due to >{} sample overlap'.format(
        too_close, too_close / timestamps.shape[0] * 100, reject_overlap))

    valid_timestamps = timestamps[1:][ts_diff]
    valid_timestamps = np.insert(valid_timestamps, 0, timestamps[0])

    return valid_timestamps


def extract_waveforms(timestamps, arr, outpath, s_pre=10, s_post=22, lc=300, hc=6000, chunk_size_s=60,
                      chunk_overlap_s=0.05, fs=3e4):
    """Extracts waveforms from raw signal around s_pre->s_post samples of spike trough. Waveforms and timestamps
    are stored directly in .mat files.
    """
    assert max(timestamps) + s_post < arr.shape[0]
    assert min(timestamps) - s_pre >= 0

    if s_pre + s_post != 32:
        logger.warning(f'Number of waveforms samples {s_pre}+{s_post} != 32 as expected by MClust!')

    chunk_size = int(chunk_size_s * fs)
    n_samples = s_pre + s_post
    n_channels = arr.shape[1]

    b, a = butter_bandpass(lc, hc, fs)

    # samples to cut around detection (threshold crossing)
    bc_samples = np.arange(-s_pre, s_post).reshape((1, n_samples))

    end = arr.shape[0]
    chunk_starts = [cs * chunk_size for cs in range(ceil(end / chunk_size))]
    chunk_overlap = int(chunk_overlap_s * fs)

    # prepare the mat file
    if os.path.exists(outpath):
        raise FileExistsError('Mat file already exists. Exiting.')

    # TODO: Save additional metadata alongside waveforms, e.g. thresholds, version, original paths
    h5s.savemat(str(outpath), {'n': len(timestamps),
                               'index': np.double((timestamps - s_pre) / 3),  # convert to MClust time domain
                               'readme': 'Written by dataman.',
                               # 'original_path': str(path)
                               }, compress=False)

    n_samples_concat = n_samples * n_channels
    with h5.File(str(outpath), 'a') as hf:
        hf.create_dataset('spikes', (n_samples_concat, len(timestamps)), maxshape=(n_samples_concat, None),
                          dtype='int16')
        for n_chunk, start in enumerate(tqdm(chunk_starts, leave=False, desc='3) extracting')):
            # limits of core batch chunk
            b_start = start
            b_end = min(start + chunk_size, end)

            # limits of chunk with flanking overlaps
            o_start = max(0, b_start - chunk_overlap)
            o_end = min(b_end + chunk_overlap, end)

            # Relevant timestamps
            ts_idc = np.where((timestamps >= b_start) & (timestamps < b_end))[0]
            peaks = timestamps[ts_idc].reshape(-1, 1) - o_start

            if not len(peaks):
                continue

            # Bandpass filter raw signal
            filtered = signal.filtfilt(b, a, arr[o_start:o_end], axis=0)

            # Extract waveforms
            idc = bc_samples + peaks

            try:
                hf['spikes'][:, min(ts_idc):max(ts_idc) + 1] = filtered[idc].reshape(-1, n_samples_concat).T
            except IndexError:
                logger.error('Spikes out of bounds!')
                break
        waveforms = np.array(hf['spikes'], dtype='int16').reshape([s_pre + s_post, n_channels, -1])
    return waveforms


def main(args):
    parser = argparse.ArgumentParser('Detect spikes in .dat files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose (debug) output")

    parser.add_argument('target', default='.',
                        help="""Directory with tetrode files.""")
    parser.add_argument('-o', '--out_path', help='Output file path Defaults to current working directory')
    parser.add_argument('--sampling-rate', type=float, help='Sampling rate. Default 30000 Hz', default=3e4)
    parser.add_argument('--noise_percentile', type=int, help='Noise percentile. Default: 5', default=5)
    parser.add_argument('--threshold', type=float, help='Threshold. Default: 4.5', default=4.5)
    parser.add_argument('-t', '--tetrodes', nargs='*', help='0-index list of tetrodes to look at. Default: all.')
    parser.add_argument('-f', '--force', action='store_true', help='Force overwrite of existing files.')
    parser.add_argument('-a', '--align', help='Alignment method, default: min', default='min')
    parser.add_argument('--start', type=float, help='Segment start in seconds', default=0)
    parser.add_argument('--end', type=float, help='Segment end in seconds')

    cli_args = parser.parse_args(args)
    logger.debug('Arguments: {}'.format(cli_args))

    stddev_factor = cli_args.threshold
    logger.debug('Threshold factor  : {}'.format(stddev_factor))

    fs = cli_args.sampling_rate
    logger.debug('Sampling rate     : {}'.format(fs))

    noise_percentile = cli_args.noise_percentile
    logger.debug('Noise percentile : {}'.format(noise_percentile))

    alignment_method = cli_args.align
    logger.debug('Alignment method : {}'.format(alignment_method))

    target = Path(cli_args.target)
    if target.is_file() and target.exists():
        tetrode_files = [target]
        target = target.parent
        logger.debug('Using single file mode with {}'.format(target))
    else:
        tetrode_files = sorted(target.glob('tetrode*.dat'))

    start = int(cli_args.start * fs) if cli_args.start is not None else 0
    end = int(cli_args.end * fs) if cli_args.end is not None else -1

    now = dt.today().strftime('%Y%m%d_%H%M%S')
    report_path = target / f'dataman_detect_report_{now}.html'

    for tt, tetrode_file in tqdm(enumerate(list(tetrode_files)), desc='Progress', total=len(tetrode_files), unit='TT'):
        report_string = ''

        # General report on shape, lengths etc.
        tqdm.write(f'-> Starting spike detection for {tetrode_file.name}')
        if not tetrode_file.exists():
            logger.info(f"{tetrode_file} not found. Skipping.")
            continue

        matpath = tetrode_file.with_suffix('.mat')
        if matpath.exists() and not cli_args.force:
            logger.error(f'{matpath} already exists. Delete or use --force to overwrite.')
            exit(1)
        elif matpath.exists() and cli_args.force:
            logger.warning(f'{matpath} already exists, deleting it.')
            os.remove(matpath)

        raw_memmap = np.memmap(tetrode_file, dtype='int16')
        logger.debug(f'loading {start}:{end} from memmap {raw_memmap}')
        wb = raw_memmap.reshape((-1, 4))[start:end]
        del raw_memmap

        logger.debug('Creating waveform figure...')
        report_string += '<h1>Recording</h1>\n'
        report_string += 'Length: {:.2f} MSamples, {:.2f} minutes'.format(wb.shape[0] / 1e6, wb.shape[0] / fs / 60)

        report_string += f'<h1>{tetrode_file.name}</h1>'
        report_string += str(tetrode_file) + '<br>'

        fig = report.plot_raw(wb)
        report_string += report.fig2html(fig) + '<br>'
        plt.close(fig)
        del fig

        logger.debug('Creating noise estimation figure...')
        # Noise estimation for threshold calculation
        noise = estimate_noise(wb)

        # Calculate threshold based on all segments with a minimum amount of noise
        # to not incorporate zeroed out segments
        ne_nz = noise.sum(axis=1) > MINIMUM_NOISE_THRESHOLD
        non_zero_ne = noise[ne_nz, :]
        noise_perc = np.percentile(non_zero_ne, noise_percentile, axis=0)

        ne_min = np.min(noise, axis=0)
        ne_max = np.max(noise, axis=0)
        ne_std = np.std(noise, axis=0)

        # Report noise amplitudes
        report_string += '<h2>Noise estimation</h2>'
        fig = report.plot_noise(noise, thresholds=noise_perc, tetrode=tetrode_file.name)
        report_string += report.fig2html(fig) + '<br>'
        plt.close(fig)
        del fig

        thr = noise_perc * stddev_factor
        for ch in range(4):
            info_line = '<b>Channel {}:</b>, Threshold: {:.1f}, 5th percentile: {:.1f} ' \
                        'uV (min: {:.1f}, max: {:.1f}, std: {:.1f})'
            report_string += info_line.format(ch, thr[ch], noise_perc[ch], ne_min[ch], ne_max[ch], ne_std[ch]) + '<br>'

        # Spike Timestamp Detection ##################################################
        report_string += '<h2>Spike Detection</h2>'

        timestamps = detect_spikes(wb, thr, align=cli_args.align, fs=fs)

        sps = len(timestamps) / (wb.shape[0] / fs)
        report_string += '<b>{} spikes</b> ({:.1f} sps) </br>'.format(len(timestamps), sps)
        logger.info(f'{tetrode_file.name}: {len(timestamps)} spikes, {sps:.1f} sps')

        # Spike Waveform Extraction ##################################################
        waveforms = extract_waveforms(timestamps, wb, outpath=matpath, s_pre=10, s_post=22, fs=fs)

        # Create waveform plots
        logger.debug('Creating waveform plots')
        density_agg = 'log'
        images = dataman.lib.report.ds_shade_waveforms(waveforms, how=density_agg)
        fig = dataman.lib.report.ds_plot_waveforms(images, density_agg)
        report_string += report.fig2html(fig) + '</br>'
        del fig

        # Tetrode Done!
        report_string += '</hr>'

        with open(report_path, 'a') as rf:
            rf.write(report_string)

    logger.info('Done!')
