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


def clean(arch, template=None, memory=False, pscrunch=True, max_iter=10, pulse_region=[0, 1, 1], unload_res=True,
          chanthresh=5, subintthresh=5, bad_chan_frac=1, bad_subint_frac=1, plot_zap=True, log=True):
    ar = psrchive.Archive_load(arch)

    mjd = (float(ar.start_time().strtempo()) + float(ar.end_time().strtempo())) / 2.0
    name = ar.get_source()
    cent_freq = ar.get_centre_frequency()

    print("Loaded archive {0}...".format(arch))
    print("Source name: {0}".format(name))
    print("Centre frequency: {0:.3f} MHz".format(cent_freq))
    print("MJD of mid-point: {0}".format(mjd))

    orig_weights = ar.get_weights()
    if memory and not pscrunch:
        pass
    else:
        ar.pscrunch()

    patient = ar.clone()
    ar_name = ar.get_filename().split()[-1]
    x = 0
    max_iterations = max_iter
    pulse_region = pulse_region

    # Create list that is used to end the iteration
    test_weights = []
    test_weights.append(patient.get_weights())
    profile_number = orig_weights.size


    print("Total number of profiles: %s" % profile_number)
    loops = 0
    while x < max_iterations:
        x += 1

        print("Loop: %s" % x)

        # Prepare the data for template creation
        patient.pscrunch()  # pscrunching again is not necessary if already pscrunched but prevents a bug
        patient.remove_baseline()
        patient.dedisperse()
        patient.fscrunch()
        patient.tscrunch()

        if template is None:
            template = patient.get_Profile(0, 0, 0).get_amps() * 10000

        # Reset patient
        patient = ar.clone()
        patient.pscrunch()
        patient.remove_baseline()
        patient.dedisperse()

        remove_profile_inplace(patient, template, pulse_region)

        # re-set DM to 0
        patient.dededisperse()

        if unload_res:
            residual = patient.clone()

        # Get data (select first polarization - recall we already P-scrunched)
        data = patient.get_data()[:, 0, :, :]
        data = apply_weights(data, orig_weights)

        # Mask profiles where weight is 0
        mask_2d = np.bitwise_not(np.expand_dims(orig_weights, 2).astype(bool))
        mask_3d = mask_2d.repeat(ar.get_nbin(), axis=2)
        data = np.ma.masked_array(data, mask=mask_3d)

        # RFI-ectomy must be recommended by average of tests
        avg_test_results = comprehensive_stats(data, cthresh=chanthresh, sthresh=subintthresh)

        # Reset patient and set weights in patient
        del patient
        patient = ar.clone()
        set_weights_archive(patient, avg_test_results)

        # Test whether weigths were already used in a previous iteration
        new_weights = patient.get_weights()
        diff_weigths = np.sum(new_weights != test_weights[-1])
        rfi_frac = (new_weights.size - np.count_nonzero(new_weights)) / float(new_weights.size)

        # Print the changes to the previous loop to help in choosing a suitable max_iter
        print("Differences to previous weights: %s  RFI fraction: %s" % (diff_weigths, rfi_frac))
        for old_weights in test_weights:
            if np.all(new_weights == old_weights):
                print("RFI removal stops after %s loops." % x)
                loops = x
                x = 1000000
        test_weights.append(new_weights)

    if x == max_iterations:
        print("Cleaning was interrupted after the maximum amount of loops (%s)" % max_iterations)
        loops = max_iterations

    # Reload archive if it is not supposed to be pscrunched.
    if not pscrunch and not memory:
        ar = psrchive.Archive_load(arch)

    # Set weights in archive.
    set_weights_archive(ar, avg_test_results)

    # Test if whole channel or subints should be removed
    if bad_chan_frac != 1 or bad_subint_frac != 1:
        ar = find_bad_parts(ar, bad_subint_frac=bad_subint_frac, bad_chan_frac=bad_chan_frac)

    # Unload residual if needed
    if unload_res:
        residual.unload("%s_residual_%s.ar" % (ar_name, loops))

    # Create plot that shows zapped( red) and unzapped( blue) profiles if needed
    if plot_zap:
        plt.imshow(avg_test_results.T, vmin=0.999, vmax=1.001, aspect='auto',
                   interpolation='nearest', cmap=cm.coolwarm)
        plt.gca().invert_yaxis()
        plt.title("%s cthresh=%s sthresh=%s" % (ar_name, chanthresh, subintthresh))
        plt.savefig("%s_%s_%s.png" % (ar_name, chanthresh, subintthresh), bbox_inches='tight')

    # Create log that contains the used parameters
    #TODO: replace this and all output with logging module
    if log:
        with open("clean.log", "w+") as logfile:
            logfile.write(
                """%s: Cleaned %s
                        required loops=%s
                        channel threshold=%f
                        subint threshold=%f
                        bad channel fraction threshold=%f
                        bad subint fraction threshold=%f
                        on-pulse region (start, end, scale_factor)=%s\n\n
                """ % (datetime.datetime.now(), ar_name, loops,
                       chanthresh, subintthresh, bad_chan_frac, bad_subint_frac,
                       pulse_region))

    return ar


def comprehensive_stats(data, cthresh=5, sthresh=5):
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
        chan_scaled = np.abs(channel_scaler(diag)) / cthresh
        subint_scaled = np.abs(subint_scaler(diag)) / sthresh
        scaled_diagnostics.append(np.max((chan_scaled, subint_scaled), axis=0))

    test_results = np.median(scaled_diagnostics, axis=0)  # we could be more extreme and take the min/max

    return test_results



