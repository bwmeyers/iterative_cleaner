#!/usr/bin/env python

# Tool to remove RFI from pulsar archives.
# Originally written by Patrick Lazarus. Modified by Lars Kuenkel.
# Adapted for NANOGrav wideband data by Bradley Meyers.

import argparse

from cleaner import clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Commands for the cleaner')
    parser.add_argument('archive', nargs='+', type=str, help='The chosen archive(s)')
    parser.add_argument('-t', '--template', type=str, default=None,
                        help="Template file. If the input is 'data', then the template will be constructed iteratively "
                             "from the data after each cleaning step.")
    parser.add_argument('-c', '--chanthresh', type=float, default=5, metavar='channel_threshold',
                        help='The threshold (in number of sigmas) a '
                             'profile needs to stand out compared to '
                             'others in the same channel for it to '
                             'be removed.')
    parser.add_argument('-s', '--subintthresh', type=float, default=5, metavar='subint_threshold',
                        help='The threshold (in number of sigmas) a '
                             'profile needs to stand out compared to '
                             'others in the same subint for it to '
                             'be removed.')
    parser.add_argument('-m', '--max_iter', type=int, default=5, metavar='maximum_iterations',
                        help='Maximum number of iterations.')
    parser.add_argument('-u', '--unload_res', action='store_true',
                        help='Creates an archive that contains the pulse free residual.')
    parser.add_argument('-p', '--pscrunch', action='store_true', help='Pscrunches the output archive.')
    parser.add_argument('-l', '--log', action='store_true', help='Create cleaning log.')
    parser.add_argument('-r', '--pulse_region', nargs=3, type=float, default=[0.0, 1.0, 1.0],
                        metavar=('pulse_start', 'pulse_end', 'scaling_factor'),
                        help="Defines the range of the pulse and a suppression factor.")
    parser.add_argument('--memory', action='store_true',
                        help='Do not pscrunch the archive while it is in memory. '
                             'Costs RAM but prevents having to reload the archive.')
    parser.add_argument('--bad_chan_frac', type=float, default=1,
                        help='Fraction of subints that needs to be removed in order to remove the whole channel.')
    parser.add_argument('--bad_subint_frac', type=float, default=1,
                        help='Fraction of channels that needs to be removed in order to remove the whole subint.')
    parser.add_argument('--plot_zap', action='store_true',
                        help='Creates a plot that shows which profiles get zapped.')

    args = parser.parse_args()

    # Loop over each of the provided archives
    # This is terribly gross, but I'm trying to make the clean function agnostic...
    arch_list = args.__dict__.pop('archive')
    kwargs = args.__dict__.copy()
    for arch in arch_list:
        output = "{0}_clean.ar".format(arch.rsplit('.', 1)[0])

        cleaned_archive = clean(arch, output=output, **kwargs)

        print("Cleaned archive: %s" % output)
