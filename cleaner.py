#!/usr/bin/env python

import datetime
import psrchive

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from utils import (remove_profile_inplace,
                   apply_weights, set_weights_archive,
                   channel_scaler, subint_scaler,
                   find_bad_parts)


def clean(ar, args, arch):
    orig_weights = ar.get_weights()
    if args.memory and not args.pscrunch:
        pass
    else:
        ar.pscrunch()
    patient = ar.clone()
    ar_name = ar.get_filename().split()[-1]
    x = 0
    max_iterations = args.max_iter
    pulse_region = args.pulse_region

    # Create list that is used to end the iteration
    test_weights = []
    test_weights.append(patient.get_weights())
    profile_number = orig_weights.size
    if not args.quiet:
        print("Total number of profiles: %s" % profile_number)
    while x < max_iterations:
        x += 1
        if not args.quiet:
            print("Loop: %s" % x)

        # Prepare the data for template creation
        patient.pscrunch()  # pscrunching again is not necessary if already pscrunched but prevents a bug
        patient.remove_baseline()
        patient.dedisperse()
        patient.fscrunch()
        patient.tscrunch()
        template = patient.get_Profile(0, 0, 0).get_amps() * 10000

        # Reset patient
        patient = ar.clone()
        patient.pscrunch()
        patient.remove_baseline()
        patient.dedisperse()
        remove_profile_inplace(patient, template, pulse_region)

        # re-set DM to 0
        patient.dededisperse()

        if args.unload_res:
            residual = patient.clone()

        # Get data (select first polarization - recall we already P-scrunched)
        data = patient.get_data()[:, 0, :, :]
        data = apply_weights(data, orig_weights)

        # Mask profiles where weight is 0
        mask_2d = np.bitwise_not(np.expand_dims(orig_weights, 2).astype(bool))
        mask_3d = mask_2d.repeat(ar.get_nbin(), axis=2)
        data = np.ma.masked_array(data, mask=mask_3d)

        # RFI-ectomy must be recommended by average of tests
        avg_test_results = comprehensive_stats(data, args, axis=2)

        # Reset patient and set weights in patient
        del patient
        patient = ar.clone()
        set_weights_archive(patient, avg_test_results)

        # Test whether weigths were already used in a previous iteration
        new_weights = patient.get_weights()
        diff_weigths = np.sum(new_weights != test_weights[-1])
        rfi_frac = (new_weights.size - np.count_nonzero(new_weights)) / float(new_weights.size)

        # Print the changes to the previous loop to help in choosing a suitable max_iter
        if not args.quiet:
            print("Differences to previous weights: %s  RFI fraction: %s" % (diff_weigths, rfi_frac))
        for old_weights in test_weights:
            if np.all(new_weights == old_weights):
                if not args.quiet:
                    print("RFI removal stops after %s loops." % x)
                loops = x
                x = 1000000
        test_weights.append(new_weights)

    if x == max_iterations:
        if not args.quiet:
            print("Cleaning was interrupted after the maximum amount of loops (%s)" % max_iterations)
        loops = max_iterations

    # Reload archive if it is not supposed to be pscrunched.
    if not args.pscrunch and not args.memory:
        ar = psrchive.Archive_load(arch)

    # Set weights in archive.
    set_weights_archive(ar, avg_test_results)

    # Test if whole channel or subints should be removed
    if args.bad_chan != 1 or args.bad_subint != 1:
        ar = find_bad_parts(ar, args)

    # Unload residual if needed
    if args.unload_res:
        residual.unload("%s_residual_%s.ar" % (ar_name, loops))

    # Create plot that shows zapped( red) and unzapped( blue) profiles if needed
    if args.print_zap:
        plt.imshow(avg_test_results.T, vmin=0.999, vmax=1.001, aspect='auto',
                   interpolation='nearest', cmap=cm.coolwarm)
        plt.gca().invert_yaxis()
        plt.title("%s cthresh=%s sthresh=%s" % (ar_name, args.chanthresh, args.subintthresh))
        plt.savefig("%s_%s_%s.png" % (ar_name, args.chanthresh,
                                      args.subintthresh), bbox_inches='tight')

    # Create log that contains the used parameters
    if not args.no_log:
        with open("clean.log", "a") as myfile:
            myfile.write("\n %s: Cleaned %s with %s, required loops=%s"
                         % (datetime.datetime.now(), ar_name, args, loops))
    return ar


def comprehensive_stats(data, args):
    """The comprehensive scaled stats that are used for
        the "Surgical Scrub" cleaning strategy.

        Inputs:
            data: A 3-D numpy array.

            args: argparse namepsace object that need to contain the
                following two parameters:

                chanthresh: The threshold (in number of sigmas) a
                    profile needs to stand out compared to others in the
                    same channel for it to be removed.
                    (Default: use value defined in config files)

                subintthresh: The threshold (in number of sigmas) a profile
                    needs to stand out compared to others in the same
                    sub-int for it to be removed.
                    (Default: use value defined in config files)

        Output:
            stats: A 2-D numpy array of stats.
    """
    chanthresh = args.chanthresh
    subintthresh = args.subintthresh

    # NOTE: This is the important part which defines what kind of statistical/measured quantities are used to decide
    # whether some channel is corrupted. We can add/modify/remove any or all of these.
    diagnostic_functions = [
        np.ma.std,
        np.ma.mean,
        np.ma.ptp,
        lambda data, axis: np.max(np.abs(np.fft.rfft(
            data - np.expand_dims(data.mean(axis=axis), axis=axis),
            axis=axis)), axis=axis)
    ]

    # Compute diagnostics, resulting in a single value per diagnostic for each subintegration/channel
    diagnostics = []
    for func in diagnostic_functions:
        diagnostics.append(func(data, axis=2))

    # Now step through data and identify bad profiles
    scaled_diagnostics = []
    for diag in diagnostics:
        chan_scaled = np.abs(channel_scaler(diag)) / chanthresh
        subint_scaled = np.abs(subint_scaler(diag)) / subintthresh
        scaled_diagnostics.append(np.max((chan_scaled, subint_scaled), axis=0))

    test_results = np.median(scaled_diagnostics, axis=0)  # we could be more extreme and take the min/max

    return test_results



