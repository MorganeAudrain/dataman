import numpy as np
import logging
import os
import os.path as op
from dataman.lib import util
from dataman.formats import get_valid_formats
from dataman.formats import open_ephys as oe
import pprint
from contextlib import ExitStack
import time
import tqdm
import argparse
from dataman.lib.constants import LOG_LEVEL_VERBOSE
from pprint import pformat
from pathlib import Path

logger = logging.getLogger(__name__)

# disable font_manager spamming the debug log
# logging.getLogger('matplotlib').disabled = True
logging.getLogger('matplotlib.fontmanager').disabled = True
logging.getLogger('matplotlib.font_manager').disabled = True

WRITE_DATA = True
FORMATS = {fmt.FMT_NAME.lower(): fmt for fmt in get_valid_formats()}

LOG_STR_INPUT = '==> Input: {path}'
LOG_STR_OUTPUT = '<== Output {path}'
LOG_STR_CHAN = 'Channels: {channels}, reference: {reference}, Dead: {dead}, ' \
               'proc_node: {proc_node}, write mode: {file_mode}'
LOG_STR_ITEM = ', Header: channel: {header[channel]}, date: {header[date_created]}'
DEBUG_STR_CHUNK = 'Reading {count} records (left: {left}, max: {num_records})'
DEBUG_STR_REREF = 'Re-referencing by subtracting average of channels {channels}'
DEBUG_STR_ZEROS = 'Zeroing (Flag: {flag}) dead channel {channel}'

MODE_STR = {'a': 'Append', 'w': "Write"}
MODE_STR_PAST = {'a': 'Appended', 'w': "Wrote"}

DEFAULT_FULL_TEMPLATE = '{prefix}--cg({cg_id:02})_ch[{crs}]'
DEFAULT_SHORT_TEMPLATE = '{prefix}--cg{cg_id:02}'


def expand_sessions(lot):
    """Check if item in target list is a .session file. If so, read all lines as
    paths into the target list.
    """
    targets = []
    for target_path in lot:
        if target_path.endswith('.session') or target_path.endswith('.txt'):
            with open(target_path, 'r') as sf:
                session_items = [tp.strip() for tp in sf.readlines() if len(tp.strip())]
                logger.info(f"Expanding session file '{target_path}' with {len(session_items)} targets")
                targets.extend(session_items)

        elif len(target_path):
            targets.append(target_path)
    logger.debug(f'Targets paths post expansion: {targets}')
    return targets


def continuous_to_dat(target_metadata, output_path, channel_group,
                      file_mode='w', chunk_records=1000, duration=0,
                      dead_channel_ids=None, zero_dead_channels=True):
    start_t = time.time()
    # Logging
    file_handler = logging.FileHandler(output_path + '.log')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.log(level=LOG_LEVEL_VERBOSE, msg='Target metadata: {}'.format(pformat(target_metadata, indent=2)))

    # NOTE: Channel numbers zero-based in configuration, but not in file name space. Grml.
    data_channel_ids = channel_group['channels']
    ref_channel_ids = [rid for rid in channel_group['reference']] if "reference" in channel_group else []
    dead_channel_ids = [did for did in dead_channel_ids]
    logger.debug("Zeroing dead channels: {}, dead (OE) channels: {}".format(zero_dead_channels, dead_channel_ids))
    dead_channels_indices = [data_channel_ids.index(dc) for dc in dead_channel_ids if dc in data_channel_ids]

    try:
        logger.debug('Opening output file {} in filemode {}'.format(output_path, file_mode + 'b'))
        with ExitStack() as stack,\
                open(output_path, file_mode + 'b') as out_fid_dat,\
                open(output_path + '.dman', 'a') as dman_offset_file:

            data_duration = 0
            samples_written = 0

            # Loop over all sub-recordings
            for sub_id, subset in target_metadata['SUBSETS'].items():
                logger.debug('Converting sub_id {}'.format(sub_id))
                data_file_paths = [subset['FILES'][cid]['FILEPATH'] for cid in data_channel_ids]
                ref_file_paths = [subset['FILES'][rid]['FILEPATH'] for rid in ref_channel_ids]
                logger.log(level=LOG_LEVEL_VERBOSE, msg=data_file_paths)

                data_files = [stack.enter_context(oe.ContinuousFile(f)) for f in data_file_paths]
                ref_files = [stack.enter_context(oe.ContinuousFile(f)) for f in ref_file_paths]
                for oe_file in data_files:
                    logger.log(level=LOG_LEVEL_VERBOSE, msg="Open data file: {}".format(op.basename(oe_file.path)) +
                                                            LOG_STR_ITEM.format(header=oe_file.header))
                for oe_file in ref_files:
                    logger.log(level=LOG_LEVEL_VERBOSE,
                               msg="Open reference file: {}".format(op.basename(oe_file.path)) +
                                   LOG_STR_ITEM.format(header=oe_file.header))

                n_blocks = subset['JOINT_HEADERS']['n_blocks']
                sampling_rate = subset['JOINT_HEADERS']['sampling_rate']
                # buffer_size = subset['JOINT_HEADERS']['buffer_size']
                block_size = subset['JOINT_HEADERS']['block_size']

                # If duration limited, find max number of records that should be grabbed
                records_left = n_blocks if not duration \
                    else min(n_blocks, int(duration * sampling_rate // block_size))
                if records_left < 1:
                    epsilon = 1 / sampling_rate * block_size * 1000
                    logger.warning(at
                        "Remaining duration limit ({:.0f} ms) less than duration of single block ({:.0f} ms). "
                        "Skipping target.".format(duration * 1000, epsilon))
                    return 0

                # loop over all records, in chunk sizes
                bytes_written = 0
                pbar = tqdm.tqdm(total=records_left * 1024, unit_scale=True, unit='Samples')
                while records_left:
                    count = min(records_left, chunk_records)

                    logger.log(level=LOG_LEVEL_VERBOSE, msg=DEBUG_STR_CHUNK.format(count=count, left=records_left,
                                                                                   num_records=n_blocks))
                    res = np.vstack([f.read_record(count) for f in data_files])

                    # reference channels if needed
                    if len(ref_channel_ids):
                        logger.debug(DEBUG_STR_REREF.format(channels=ref_channel_ids))
                        res -= np.vstack([f.read_record(count) for f in ref_files]).mean(axis=0, dtype=np.int16)

                    # zero dead channels if needed
                    if len(dead_channels_indices) and zero_dead_channels:
                        zeros = np.zeros_like(res[0])
                        for dci in dead_channels_indices:
                            logger.debug(DEBUG_STR_ZEROS.format(flag=zero_dead_channels, channel=data_channel_ids[dci]))
                            res[dci] = zeros

                    res.transpose().tofile(out_fid_dat)

                    records_left -= count
                    pbar.update(count * 1024)
                    samples_written += count * 1024
                    bytes_written += (count * 2048 * len(data_channel_ids))

                pbar.close()

                data_duration += bytes_written / (2 * sampling_rate * len(data_channel_ids))
                elapsed = time.time() - start_t
                speed = bytes_written / elapsed
                logger.debug('{appended} {channels} channels into "{op:s}"'.format(
                    appended=MODE_STR_PAST[file_mode], channels=len(data_channel_ids),
                    op=os.path.abspath(output_path)))
                logger.info(
                    '{n_channels} channels, {rec} blocks ({dur:s}, {bw:.2f} MB) in {et:.2f} s ({ts:.2f} MB/s)'.format(
                        n_channels=len(data_channel_ids), rec=n_blocks - records_left, dur=util.fmt_time(data_duration),
                        bw=bytes_written / 1e6, et=elapsed, ts=speed / 1e6))
                # returning duration of data written, epsilon=1 sample, allows external loop to make proper judgement if
                # going to next target makes sense via comparison. E.g. if time less than one sample short of
                # duration limit.
                logger.removeHandler(file_handler)
                file_handler.close()

            # Writing segment position data
            dman_offset_file.write('{}, {}\n'.format(target_metadata['TARGET'], samples_written))
            print('written!')
            return data_duration

    except IOError as e:
        logger.exception('Operation failed: {error}'.format(error=e.strerror))


def main(args):
    parser = argparse.ArgumentParser('Convert file formats/layouts. Default result is int16 .dat file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose (debug) output")

    # Input/output
    parser.add_argument('target', nargs='*', default='.',
                        help="""Path/list of paths to directories containing raw .continuous data OR path
                                to .session definition file. Listing multiple files will result in data sets
                                being concatenated in listed order.""")
    parser.add_argument('-o', '--out_path', help='Output file path Defaults to current working directory')
    parser.add_argument('-P', '--out_prefix', help='Output file prefix. Default is name of target.')
    parser.add_argument('-T', '--template_fname',
                        help='Output file template. Default: {}'.format(DEFAULT_SHORT_TEMPLATE))

    parser.add_argument('-f', '--format', help='Output format. Default is: {}'.format(list(FORMATS.keys())[2]),
                        choices=FORMATS.keys(), default=list(FORMATS.keys())[2])
    parser.add_argument('--fname_channels', action='store_true', help='Include original channel numbers in file names.')

    # Channel arrangement
    channel_group = parser.add_mutually_exclusive_group()
    channel_group.add_argument('-c', "--channel-count", type=int,
                               help='Number of consecutive channels.')
    channel_group.add_argument('-C', "--channel-list", nargs='*', type=int,
                               help='List of channels in order they are to be merged.')
    channel_group.add_argument('-l', '--layout',
                               help="Path to klusta .probe file.")
    parser.add_argument('-g', '--channel-groups', type=int, nargs="+",
                        help="limit to only a subset of the channel groups")
    parser.add_argument('-S', '--split-groups', action='store_true',
                        help='Split channel groups into separate files.')
    parser.add_argument('-d', '--dead-channels', nargs='*', type=int,
                        help='List of dead channels. If flag set, these will be set to zero.')
    parser.add_argument('-z', '--zero-dead-channels', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Do not write data files (but still create prb/prm')
    parser.add_argument('-p', "--params", help='Path to .params file.')
    parser.add_argument('-D', "--duration", type=int, help='Limit duration of recording (s)')
    parser.add_argument('--remove-trailing-zeros', action='store_true')
    parser.add_argument('--out_fname_template', action='store_true', help='Template for file naming.')

    cli_args = parser.parse_args(args)
    logger.debug('Arguments: {}'.format(cli_args))

    if cli_args.remove_trailing_zeros:
        raise NotImplementedError("Trailing zero removal not implemented (also not a good idea to begin with...)")

    targets = [op.abspath(op.expanduser(t)) for t in expand_sessions(cli_args.target)]
    target_exists = [t for t in targets if op.exists(t)]
    if not all(target_exists):
        raise FileNotFoundError(f'Invalid targets found: {[t for ok, t in zip(target_exists, targets) if not ok]}')

    formats = list(set([util.detect_format(target) for target in targets]))

    # Input file format
    logger.debug('Inputs found: {}'.format(formats))
    format_input = formats[0]
    assert len(formats) == 1
    logger.debug('Using module: {}'.format(format_input.__name__))

    # Output file format
    format_output = FORMATS[cli_args.format.lower()]
    logger.debug('Output module: {}'.format(format_output.__name__))

    # Set up channel layout (channels, references, dead channels) from command line inputs or layout file
    # List of bad channels, will be added to channel group dict
    dead_channels = cli_args.dead_channels if cli_args.dead_channels is not None else []

    # One of channel_count, channel_list, layout_file path from mutex parser group channel_group
    layout = None
    layout_file = None

    if cli_args.layout is not None:
        layout_file = Path(cli_args.layout).resolve()

    # if no .prb file specified, check if there is one matching the name of the session
    else:
        for target in cli_args.target:
            layout_file = Path(target).with_suffix('.prb').resolve()
            if layout_file.exists():
                logger.warning(f'Using probe file {layout_file.name} matching session file name.')
                break

    if cli_args.channel_count is not None:
        channel_groups = {0: {'channels': list(range(cli_args.channel_count)),
                              'dead_channels': dead_channels}}

    elif cli_args.channel_list is not None:
        channel_groups = {0: {'channels': cli_args.channel_list,
                              'dead_channels': dead_channels}}

    elif layout_file:
        logger.debug(f'Reading layout file {layout_file}')
        layout = util.run_prb(layout_file)
        if cli_args.split_groups:
            channel_groups = layout['channel_groups']
            if 'dead_channels' in layout:
                if len(dead_channels) and (layout['dead_channels'] != dead_channels):
                    raise ValueError(
                        'Conflicting bad channel lists: args: {}, layout: {}'.format(layout['dead_channels'],
                                                                                     dead_channels))
                dead_channels = layout['dead_channels']
            if cli_args.channel_groups:
                channel_groups = {i: channel_groups[i] for i in cli_args.channel_groups if i in channel_groups}
        else:
            channels, dead_channels = util.flat_channel_list(layout)
            logger.warning('Not splitting groups! Creating new monotonically increasing channel map.')

            # make a new channel group by merging in the existing ones
            channel_groups = {0: {'channels': channels,
                                  'dead_channels': dead_channels}}
    else:
        logger.debug('No channels given on CLI, will try to get channel number from target later.')
        channel_groups = None

    # Generate configuration from input files found by the format
    # This step already checks for the existence of the data files
    # and can fail prematurely if the wrong naming template is being used
    # This needs more work.
    logger.debug('Getting metadata for all targets')
    targets_metadata_list = [format_input.metadata_from_target(t) for t in targets]

    if channel_groups is None:
        target_channels = list(set([ch for t in targets_metadata_list for ch in t['CHANNELS']]))

        channel_groups = {0: {'channels': target_channels,
                              'dead_channels': dead_channels}}

    # Output file path
    if cli_args.out_path is None:
        out_path = os.getcwd()
        logger.info('Using current working directory "{}" as output path.'.format(out_path))
    else:
        out_path = op.abspath(op.expanduser(cli_args.out_path))

    # Create the output path if necessary
    if len(out_path) and not op.exists(out_path):
        os.mkdir(out_path)
        logger.debug('Creating output path {}'.format(out_path))

    out_fext = format_output.FMT_FEXT
    out_prefix = cli_args.out_prefix if cli_args.out_prefix is not None else op.basename(cli_args.target[0])

    if out_prefix =='':
        out_prefix ='lost_name'
    if '.' in out_prefix:
        out_prefix = out_prefix.split('.')[0]

    logger.debug('Prefix: {}'.format(out_prefix))
    default_template = DEFAULT_FULL_TEMPLATE if cli_args.fname_channels else DEFAULT_SHORT_TEMPLATE
    fname_template = default_template if cli_args.template_fname is None else cli_args.template_fname
    logger.debug('Filename template: {}'.format(fname_template))

    # +++++++++++++++++++++++++++++++++++++++++ MAIN LOOP ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Iterates over all channel groups, calling continuous_to_dat for each to be bundled together
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    total_duration_written = 0
    for cg_id, channel_group in channel_groups.items():
        logger.debug('channel group: {}'.format(channel_group))

        # TODO: Check file name length, shorten if > 256 characters
        # Possible parameters: outfile prefix [outfile], channel group id [cg_id]
        # channel ranges from consecutive channels, for output file naming
        crs = util.fmt_channel_ranges(channel_group['channels'])
        output_basename = fname_template.format(prefix=out_prefix, cg_id=cg_id, crs=crs)
        output_fname = ''.join([output_basename, out_fext])
        output_file_path = op.join(out_path, output_fname)

        with open(output_file_path + '.dataman.offsets', 'w') as dman_offset_file:
            dman_offset_file.write('target_path, num_samples\n')

        duration_written = 0
        # First target, file mode is write, after that, append to output file
        for file_mode, target_metadata in enumerate(targets_metadata_list):
            duration = None if cli_args.duration is None else cli_args.duration - duration_written
            target_path = target_metadata['TARGET']

            logger.debug('Starting conversion for target {}'.format(target_path))

            if not cli_args.dry_run and WRITE_DATA:
                duration_written += continuous_to_dat(
                    target_metadata=target_metadata,
                    output_path=output_file_path,
                    channel_group=channel_group,
                    dead_channel_ids=dead_channels,
                    zero_dead_channels=cli_args.zero_dead_channels,
                    file_mode='a' if file_mode else 'w',
                    duration=duration,
                    chunk_records=5)
            total_duration_written += duration_written

        # create the per-group .prb files
        # FIXME: Dead channels are big mess
        if cli_args.split_groups or (layout is None):
            # One prb file per channel group
            ch_out = channel_group['channels']
            cg_out = {0: {'channels': list(range(len(ch_out)))}}
            dead_channels = sorted([ch_out.index(dc) for dc in dead_channels if dc in ch_out])

        else:
            # Same channel groups, but with flat numbering
            cg_out, dead_channels = util.monotonic_prb(layout)

        with open(op.join(out_path, output_basename + '.prb'), 'w') as prb_out:
            prb_out.write('dead_channels = {}\n'.format(pprint.pformat(dead_channels)))
            prb_out.write('channel_groups = {}'.format(pprint.pformat(cg_out)))

        logger.debug('Done! Total data length written: {}'.format(util.fmt_time(total_duration_written)))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])