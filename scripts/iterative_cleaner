#!/usr/bin/env python

# Tool to remove RFI from pulsar archives.
# Originally written by Patrick Lazarus. Modified by Lars Kuenkel.
# Adapted for NANOGrav wideband data by Bradley Meyers.

import argparse
import logging
import datetime

from iterative_cleaner.cleaner import clean

# create logger
logger = logging.getLogger('iterative_cleaner')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler('iterative_cleaner_{0}.log'.format(
    datetime.datetime.now(datetime.timezone.utc).strftime("%d%m%Y%H%M%S")))
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Commands for the cleaner')
    parser.add_argument('archive', nargs='+', type=str, help='The chosen archive(s)')
    parser.add_argument('-t', '--template', type=str, default=None,
                        help='Template file (if desired). CURRENTLY NOT IMPLEMENTED PROPERLY. '
                             'If the input is "data", then the template will be constructed iteratively '
                             'from the data after each cleaning step.')
    parser.add_argument('-c', '--chanthresh', type=float, default=5, metavar='channel_threshold',
                        help='The threshold (in number of sigmas) a profile needs to stand out compared '
                             'to others in the same channel for it to be removed.')
    parser.add_argument('-s', '--subintthresh', type=float, default=5, metavar='subint_threshold',
                        help='The threshold (in number of sigmas) a profile needs to stand out compared '
                             'to others in the same subint for it to be removed.')
    parser.add_argument('--bad_chan_frac', type=float, default=1,
                        help='Fraction of subints that needs to be removed in order to remove the whole channel.')
    parser.add_argument('--bad_subint_frac', type=float, default=1,
                        help='Fraction of channels that needs to be removed in order to remove the whole subint.')
    parser.add_argument('-o', '--onpulse', nargs=2, type=float, default=None, metavar=('pulse_start', 'pulse_end'),
                        help='Defines the on-pulse region (if desired)')
    parser.add_argument('-m', '--max_iter', type=int, default=5, metavar='max_iterations',
                        help='Maximum number of iterations.')
    parser.add_argument('--unload_res', action='store_true',
                        help='Creates an archive that contains the pulse free residual.')
    parser.add_argument('--pscrunch', action='store_true', help='Scrunches output archive to total intensity only.')
    parser.add_argument('--memory', action='store_true',
                        help='Do not pscrunch the archive while it is in memory. '
                             'Costs RAM but prevents having to reload the archive.')

    parser.add_argument('--plot_zap', action='store_true',
                        help='Creates a plot that shows which profiles get zapped.')

    args = parser.parse_args()

    # Loop over each of the provided archives
    # This is terribly gross, but I'm trying to make the clean function agnostic...
    arch_list = args.__dict__.pop('archive')
    kwargs = args.__dict__.copy()
    for arch in arch_list:
        output = "{0}_clean.ar".format(arch.rsplit('.', 1)[0])
        clean(arch, output=output, **kwargs)
