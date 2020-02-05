#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cmd
import logging
import os.path as op
import sys

from dataman import __version__ as version
from dataman.lib.constants import LOG_LEVEL_VERBOSE

LOG_LEVEL_DEFAULT = logging.INFO
LOG_LEVEL_DEBUG = logging.DEBUG


class DataMan(cmd.Cmd):
    """Command line tool for quick data documentation."""

    prompt = "dm> "
    intro = "+ DataMan is here to help! +"
    intro = '\n'.join(["=" * len(intro), intro, "=" * len(intro)])

    log = logging.getLogger(__name__)

    def preloop(self):
        self.log.debug("Starting DataMan CLI")
        self.intro_dbg()

    def precmd(self, line):
        self.intro_dbg()
        return line

    def intro_dbg(self):
        self.log.debug('Starting dataman v{} @git [{}]'.format(version.__version__, version.GIT_VERSION))

    def do_ls(self, args_string):
        parser = argparse.ArgumentParser('Recording statistics', )
        parser.add_argument('path', help='Relative or absolute path to directory',
                            default='.', nargs='?')
        parser.add_argument('-d', '--debug', action='store_true',
                            help='Debug mode -- verbose output, no confirmations.')

        cli_args = parser.parse_args(args_string.split(' ') if args_string else '')
        self.log.debug('ls with args: {}'.format(cli_args))
        path = op.abspath(op.expanduser(cli_args.path))
        self.log.debug('Expanded path: {}'.format(path))

        import dataman.dirstats.dirstats as ds
        ds.print_table(ds.gather(path))

    def do_stats(self, args_string):
        self.do_ls(args_string)
        # parser = argparse.ArgumentParser('Recording statistics')
        # parser.add_argument('path', help='Relative or absolute path to directory',
        #                     default='.', nargs='?')
        # parser.add_argument('-d', '--debug', action='store_true',
        #                     help='Debug mode -- verbose output, no confirmations.')
        # cli_args = parser.parse_args(line.split(' ') if line else '')
        # self.log.debug('Stats with args: {}'.format(cli_args))
        # path = op.abspath(op.expanduser(cli_args.path))
        # self.log.debug('Expanded path: {}'.format(path))
        #
        # import dataman.dirstats.dirstats as ds
        # ds.print_table(ds.gather(path))

    @staticmethod
    def do_vis(args_string):
        from dataman.vis import vis
        vis.run(args_string.split(' '))

    @staticmethod
    def do_convert(args_string):
        from dataman.conv import convert
        convert.main(args_string.split(' '))

    @staticmethod
    def do_conv(args_string):
        DataMan.do_convert(args_string)

    @staticmethod
    def do_reference(args_string):
        from dataman.ref import referencing
        referencing.main(args_string.split(' '))

    @staticmethod
    def do_ref(args_string):
        DataMan.do_reference(args_string)

    @staticmethod
    def do_split(args_string):
        from dataman.split import split
        split.main(args_string.split(' '))

    @staticmethod
    def do_detect(args_string):
        from dataman.detect import detect
        detect.main(args_string.split(' '))

    @staticmethod
    def do_features(args_string):
        from dataman.features import features
        features.main(args_string.split(' '))

    @staticmethod
    def do_fet(args_string):
        DataMan.do_features(args_string)

    @staticmethod
    def do_cluster(args_string):
        from dataman.cluster import cluster
        cluster.main(args_string.split(' '))

    @staticmethod
    def do_check(args_string):
        raise NotImplemented('check not implemented')

    @staticmethod
    def do_proc(args_string):
        raise NotImplemented('proc not implemented')

    @staticmethod
    def do_exit(_):
        """Exit"""
        return True

    @staticmethod
    def do_EOF(_):
        """Exit"""
        return True

    def postloop(self):
        print("Done.")


def main():
    # Command line parsing
    # FIXME: Stupid spaces-in-file-names issues... especially after the string conversion to onecmd

    parser = argparse.ArgumentParser(prog="DataMan", add_help=False, usage='''
    dm <command> [<args>]

    Currently implemented commands:
        cli     Interactive CLI
        ls      Basic target statistics
        vis     Simple data visualizer
        convert Convert formats and layouts
        ref     Creating references/reference-subtracting data
        split   Split file into separate files bundling channels
        detect  Detect and extract spikes from the wide-band signal
        feature Calculate features from waveforms used for clustering
        cluster Prepare and run KlustaKwik for generated features
        ''')
    parser.add_argument('command', help='Command to execute', nargs='?', default=None)
    parser.add_argument('-v', '--verbose', action='count',
                        help='Verbosity level -v: Debug, -vv: Verbose mode.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version} @git {git_version}'.format(version=version.__version__,
                                                                               git_version=version.GIT_VERSION))
    parser.add_argument('-h', '--help', action='store_true', help='Show help text.')

    cli_args, cmd_args = parser.parse_known_args()

    if cli_args.command is None and cli_args.help:
        parser.print_help()
        sys.exit(0)

    # we need to re-append arguments that should go down the rabbit hole
    if cli_args.help:
        cmd_args.append('-h')

    logging.addLevelName(LOG_LEVEL_VERBOSE, 'VERBOSE')

    if cli_args.verbose is None:
        log_level = LOG_LEVEL_DEFAULT
    else:
        log_level = LOG_LEVEL_DEBUG if cli_args.verbose == 1 else LOG_LEVEL_VERBOSE

    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.log(level=LOG_LEVEL_VERBOSE, msg='CLI_ARGS: {}'.format(cli_args))
    logger.log(level=LOG_LEVEL_VERBOSE, msg='CMD_ARGS: {}'.format(cmd_args))

    # disable font_manager spamming the debug log
    logging.getLogger('matplotlib').disabled = True
    logging.getLogger('matplotlib.fontmanager').disabled = True

    # start cli
    if cli_args.command in [None, 'cli']:
        logger.debug('Starting CLI via command: {}'.format(cli_args.command))
        try:
            DataMan().cmdloop()
        except KeyboardInterrupt:
            pass

    # some other command was given
    else:
        dm = DataMan()
        dm.precmd('')
        dm.onecmd('{} {}'.format(cli_args.command, ' '.join(cmd_args)))


if __name__ == "__main__":
    main()
