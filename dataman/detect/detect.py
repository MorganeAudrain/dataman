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
logging.getLogger('matplotlib').disabled = True
logging.getLogger('matplotlib.fontmanager').disabled = True


def get_batches(length, batch_size):
    batches = [bc * batch_size for bc in range(length // batch_size)]
    if length - (length // batch_size) * batch_size:
        batches.append(length - length % batch_size)
    return batches



def estimate_noise(arr, lc=300, hc=6000, num_channels=4, fs=3e4, uV_factor=0.195):
    ne_binsize = int(fs)  # noise estimation binsize

    # Filter
    b, a = butter_bandpass(lc, hc, fs)
    batches = get_batches(arr.shape[0], ne_binsize)
    ne = np.zeros((len(batches), num_channels))
    nfac = 1 / 0.6745

    for n, batch in enumerate(tqdm(batches, leave=False, desc='1) estimating')):
        n_samples = min(ne_binsize, arr.shape[0] - batch)
        filtered = signal.filtfilt(b, a, arr[batch:batch + n_samples, :].astype(np.double), axis=0) * uV_factor
        for ch in range(num_channels):
            ne[n, ch] = np.median(abs(filtered[:, ch]) * nfac)
    return ne


def wv_com(arr):
    """Center of mass of abs(wv)."""
    cmass = np.cumsum(np.abs(arr))
    mass = cmass.max()
    return np.argmin(np.abs(cmass - mass / 2))


def wv_alignment(w, method='centroid', kernel_width=11, interp_f=None):
    # channel with highest amplitude
    ch = np.min(w, axis=0).argmin()
    y = w[:, ch]

    # Minimum alignment filter
    #
    # Peak negative amplitude
    if method == 'min':
        if not interp_f:
            m = np.argmin(y)
            return m, ch, None
        else:
            raise NotImplementedError

    # Centroid alignment method
    #
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


# def detect_spikes(arr, thresholds, fs=3e4, uV_factor=0.195, chunk_size_s=60, lc=300, hc=6000):
#     chunk_size = int(chunk_size_s * fs)  # chunk size for detection
#
#     use_thr = thresholds
#     max_thr = use_thr * 8
#
#     waveform_chunks = []
#     timestamp_chunks = []
#
#     spikes_per_chunk = []
#     rejections = 0
#
#     INTERP_F = 4
#     b, a = butter_bandpass(lc, hc, fs)
#
#     # add padding samples for later interpolation
#     s_pad = 0
#     n_samples = 32 + 2 * s_pad
#
#     end = arr.shape[0]
#
#     batches = [b * chunk_size for b in range(end // chunk_size)]
#     if arr.shape[0] - (arr.shape[0] // chunk_size) * chunk_size:
#         batches.append(arr.shape[0] - arr.shape[0] % chunk_size)
#
#     for batch in tqdm(batches, leave=False, desc='extraction'):
#         start = batch
#         end = start + min(chunk_size, arr.shape[0] - start)
#
#         # Filtering and Detection
#         filtered = signal.filtfilt(b, a, arr[start:end, :].astype(np.double), axis=0) * uV_factor
#
#         # Merge threshold crossings
#         # TODO: Only merge valid channels!
#         crossings = np.sum(filtered < -use_thr, axis=1)
#         crossings = np.clip(crossings, 0, 1)
#
#         # make sure crossings in the first 30 samples are ignored to avoid wrong start
#         crossings[:30] = 0
#
#         # Look for threshold crossing "onset"
#         deltas = np.diff(crossings)
#         xr_starts = (deltas > 0).nonzero()[0]
#         xr_ends = (deltas < 0).nonzero()[0]
#
#         assert len(xr_starts) == len(xr_ends)
#         spike_idx = (np.array([xr_starts, xr_ends]).T.mean(axis=1)).astype(np.int64)
#
#         # Check if last spike is too close to boundary
#         while len(spike_idx) and spike_idx[-1] + 22 >= chunk_size:
#             spike_idx = spike_idx[:-1]
#             rejections += 1
#
#         spikes_per_chunk.append(len(spike_idx))
#
#         if not len(spike_idx):
#             continue
#
#         # Extract waveforms
#         n_spikes = spike_idx.shape[0]
#         bc_samples = np.arange(-10 - s_pad, 22 + s_pad).reshape([1, n_samples])
#         bc_spikes = spike_idx.reshape([n_spikes, 1])
#
#         idc = bc_samples + bc_spikes
#
#         try:
#             wv = filtered[idc]
#         except IndexError:
#             print(spike_idx[-1] + 22)
#             break
#
#         # Temporary arrays
#         spk_ts = np.zeros(n_spikes, dtype=np.int64)
#         min_idx = np.zeros(n_spikes, dtype=np.int16)
#
#         x = np.linspace(0, n_samples, n_samples)
#         wv_f_interp = interpolate.interp1d(x, wv, axis=1, kind='cubic')
#
#         x_interp = np.linspace(0, n_samples, n_samples * INTERP_F)
#         wv_interp = wv_f_interp(x_interp)
#
#         reject = np.zeros(len(spk_ts))
#         for n in range(len(spk_ts)):  # tnrange(n_spikes)
#             # interpolated waveform
#             w = wv_interp[n]
#
#             # interpolated minimum
#             mins = np.min(w, axis=0)
#
#             # index of minimum of channel with largest amplitude valley
#             ch_sel = np.argmin(mins)
#             idx = np.argmin(w[:, ch_sel])
#             min_idx[n] = idx
#
#             if np.any(mins < -max_thr):
#                 reject[n] = True
#                 rejections += 1
#
#         spk_ts = spike_idx - 10 + np.round(min_idx / INTERP_F).astype(np.int64)
#         spk_ts = spk_ts[np.logical_not(reject)]
#
#         # Check again if last spike is too close to boundary
#         while len(spk_ts) and spk_ts[-1] + 22 >= chunk_size:
#             spk_ts = spk_ts[:-1]
#             rejections += 1
#
#         bc_samples = np.arange(-10 - s_pad, 22 + s_pad).reshape([1, n_samples])
#         bc_spikes = spk_ts.reshape([spk_ts.shape[0], 1])
#
#         idc = bc_samples + bc_spikes
#
#         spk_wv = filtered[idc]
#
#         waveform_chunks.append(spk_wv)
#         timestamp_chunks.append(spk_ts + int(start))
#
#     waveforms = np.vstack(waveform_chunks)
#     timestamps = np.concatenate(timestamp_chunks) / fs * 10000  # convert to sampling rate and MClust time format
#
#     # Reorder timestamps and waveforms
#     ts_order = np.argsort(timestamps)
#     timestamps.sort()
#     waveforms = waveforms[ts_order]
#
#     return waveforms, timestamps


def detect_spikes(arr, thresholds, fs=3e4, chunk_size_s=60, lc=300, hc=6000, s_pre=10, s_post=22,
                  reject_overlap=16, align='centroid'):
    # TODO: Interpolation
    # TODO: Maximum artifact rejection
    # TODO: Return rejected timestamps

    chunk_size = int(chunk_size_s * fs)  # chunk size for detection

    uV_factor = 0.195
    use_thr = thresholds / uV_factor
    max_thr = use_thr * 8

    # waveform_chunks = []
    timestamps = []
    crs = []

    rejections = 0

    b, a = butter_bandpass(lc, hc, fs)

    # samples to cut around detection (threshold crossing)
    n_samples = s_pre + s_post
    bc_samples = np.arange(-s_pre, s_post).reshape([1, -1])

    # Chunks will have partial overlap.
    # 0
    # |o|--- chunk 1 ---|x|
    #                 |o|--- chunk 2 ---|x|
    #                                |o|--- chunk 3 ---|
    #                                                end
    # Spikes with peak in the |x| region will be ignored
    chunk_overlap = int(0.05 * fs)  # 50 ms chunk boundary overlap

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
            logger.warning('No spikes in chunk {} @ [{} to {}]'.format(n_chunk, b_start, b_end))
            continue

        # Extract preliminary waveforms
        # get waveform indices by broadcasting indices of crossings
        bc_spikes = xr_starts.reshape([n_spikes, 1])
        idc = bc_samples + bc_spikes

        try:
            wv = filtered[idc]
        except IndexError:
            logger.error('Spikes out of bounds in chunk {} @ [{} to {}]'.format(n_chunk, b_start, b_end))
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

    tdif = np.diff(timestamps) > reject_overlap

    too_close = timestamps.shape[0] - np.sum(tdif)
    logger.warning('{} ({:.1f}%) spikes rejected due to >{} sample overlap'.format(
        too_close, too_close / timestamps.shape[0] * 100, reject_overlap))

    valids = timestamps[1:][tdif]
    valids = np.insert(valids, 0, timestamps[0])

    return valids


def extract_waveforms(timestamps, arr, outpath, s_pre=8, s_post=24, lc=300, hc=6000, chunk_size_s=60, fs=3e4):
    assert max(timestamps) + s_post < arr.shape[0]
    assert min(timestamps) - s_pre >= 0

    chunk_size = int(chunk_size_s * fs)

    n_waveforms = timestamps.shape[0]
    n_samples = s_pre + s_post
    n_channels = arr.shape[1]


    b, a = butter_bandpass(lc, hc, fs)


    # samples to cut around detection (threshold crossing)
    n_samples = s_pre + s_post
    bc_samples = np.arange(-s_pre, s_post).reshape((1, n_samples))

    end = arr.shape[0]
    chunk_starts = [cs * chunk_size for cs in range(ceil(end / chunk_size))]
    chunk_overlap = int(0.05 * fs)

    # prepare the mat file
    if os.path.exists(outpath):
        raise FileExistsError('Mat file already exists. Exiting.')

    h5s.savemat(str(outpath), {'n': len(timestamps),
                               'index': np.double((timestamps - s_pre) / 3),
                               'readme': 'Written by dataman.',
                               # 'original_path': str(path)
                               }, compress=False)

    with h5.File(str(outpath), 'a') as hf:
        hf.create_dataset('spikes', (128, len(timestamps)), maxshape=(128, None), dtype='int16')
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
                hf['spikes'][:, min(ts_idc):max(ts_idc) + 1] = filtered[idc].reshape(-1, 128).T
            except IndexError:
                logger.error('Spikes out of bounds!')
                break
        waveforms = np.array(hf['spikes'], dtype='int16').reshape(s_pre+s_post, n_channels, -1)
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
    logger.debug('Thresold factor   : ', stddev_factor)

    fs = cli_args.sampling_rate
    logger.debug('Sampling rate     : ', fs)

    noise_percentile = cli_args.noise_percentile
    logger.debug('Noise percentile : ', noise_percentile)

    alignment_method = cli_args.align
    logger.debug('Alignment method : ', alignment_method)

    target = Path(cli_args.target)
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
        report_string += 'Length: {:.2f} Msamples, {:.2f} minutes'.format(wb.shape[0] / 1e6, wb.shape[0] / fs / 60)

        report_string += f'<h1>{tetrode_file.name}</h1>'
        report_string += str(tetrode_file) + '<br>'

        fig = report.plot_raw(wb)
        report_string += report.fig2html(fig) + '<br>'
        plt.close(fig)
        del fig

        logger.debug('Creating noise estimation figure...')
        # Noise estimation for threshold calculation  ################################
        noise = estimate_noise(wb)

        noise_perc = np.percentile(noise, noise_percentile, axis=0)
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
        waveforms = extract_waveforms(timestamps, wb, outpath=matpath, s_pre=8, s_post=24, fs=fs)

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
