#!/usr/bin/env python

import datetime
import psrchive

import numpy as np
import matplotlib.pyplot as plt

from utils import (remove_profile_inplace,
                   apply_weights, set_weights_archive,
                   channel_scaler, subint_scaler,
                   find_bad_parts, fft_rotate, get_template_profile_phase)


def clean(archive, template="data", output="cleaned.ar", memory=False, pscrunch=True, max_iter=10,
          pulse_region=[0, 1, 1], unload_res=True, chanthresh=5, subintthresh=5, bad_chan_frac=1, bad_subint_frac=1,
          plot_zap=True, log=True):

    if archive is None:
        print("No archive provided")
        return

    ar = psrchive.Archive_load(str(archive))

    mjd = (float(ar.start_time().strtempo()) + float(ar.end_time().strtempo())) / 2.0
    name = ar.get_source()
    cent_freq = ar.get_centre_frequency()

    print("Loaded archive {0}...".format(str(archive)))
    print("Source name: {0}".format(name))
    print("Centre frequency: {0:.3f} MHz".format(cent_freq))
    print("MJD of mid-point: {0}".format(mjd))

    orig_weights = ar.get_weights()
    if memory and not pscrunch:
        pass
    else:
        ar.pscrunch()

    patient = ar.clone()
    patient_nchan = patient.get_nchan()
    patient_nsub  = patient.get_nsubint()
    ar_name = ar.get_filename().split()[-1]
    x = 0
    max_iterations = max_iter
    pulse_region = pulse_region

    # Create list that is used to end the iteration
    test_weights = []
    test_weights.append(patient.get_weights())
    profile_number = orig_weights.size

    # Try to load template and ensure it will work with the data (i.e. has the same number of channels)
    template_from_file = False
    if template is not None and template != "data":
        template_from_file = True
        # Assuming the template is an archive, too
        temp = psrchive.Archive_load(str(template))
        temp.pscrunch()
        temp.remove_baseline()
        temp.dedisperse()
        temp.tscrunch()

        temp_nchan = len(temp.get_frequencies())
        if temp_nchan != patient_nchan:
            print("Number of channels in template ({0}) doesn't match data ({1})... f-scrunching to 1D!".format(
                temp_nchan, patient_nchan))
            temp.fscrunch()

        # To estimate the phase offset between template and profile, first we should make a 1D template
        print("First pass estimate of phase offset between template and data")
        profile = patient.clone()
        profile.pscrunch()
        profile.remove_baseline()
        profile.dedisperse()
        profile.tscrunch()
        profile.fscrunch()
        profile = profile.get_data()[0, 0, 0, :]

        temp_phs = temp.clone()
        temp_phs.fscrunch()
        temp_phs = np.apply_over_axes(np.sum, temp_phs.get_data(), (0, 1)).squeeze()
        amp_guess = max(profile) - min(profile) - max(temp_phs)
        phase_guess = -np.argmax(profile) + np.argmax(temp_phs)
        amp, offset = get_template_profile_phase(temp_phs, profile, amp_guess=amp_guess, phase_guess=phase_guess)
        print("template phase offset = {0} bins".format(offset))
        temp = np.squeeze(temp.get_data()[0, 0, :, :])
    elif template == "data":
        # Create an initial template from the data if one is not provided
        print("No template file provided, will iteratively create one from data")
        patient.pscrunch()
        patient.remove_baseline()
        patient.dedisperse()
        patient.fscrunch()
        patient.tscrunch()
        temp = patient.get_Profile(0, 0, 0).get_amps() * 10000
        temp_nchan = 1
        offset = 0
        # No need to phase align here since it's created from the data
    else:
        print("No template provided, will not attempt to remove pulse profile from data")
        temp = None
        temp_nchan = 0


    print("Total number of profiles: %s" % profile_number)
    loops = 0
    while x < max_iterations:
        x += 1

        print("Loop: %s" % x)
        # Reset patient
        patient = ar.clone()
        patient.pscrunch()
        patient.remove_baseline()
        patient.dedisperse()

        if temp_nchan > 1 and template_from_file:
            # 2D template, needs to be rotated
            print("rotating 2D template")
            rotated_template = np.apply_along_axis(fft_rotate, 1, temp, args=(offset))
        elif temp_nchan == 1 and template_from_file:
            # 1D template, needs to be rotated
            print("rotating 1D template")
            rotated_template = fft_rotate(temp, offset)
        else:
            # 1D template made from data, no rotation OR no template
            rotated_template = temp

        if rotated_template is not None:
            remove_profile_inplace(patient, rotated_template, pulse_region)

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

        # Test whether weights were already used in a previous iteration
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

        if template is None:
            # We need to recreate the template from the data if it was never defined in the first place
            # This makes sure that each iteration will have a better template than before
            print("Re-creating template from this iteration's cleaned data")
            patient.pscrunch()  # pscrunching again is not necessary if already pscrunched but prevents a bug
            patient.remove_baseline()
            patient.dedisperse()
            patient.fscrunch()
            patient.tscrunch()
            temp = patient.get_Profile(0, 0, 0).get_amps() * 10000
            # No need to phase align since it's created from the data

    if x == max_iterations:
        print("Cleaning was interrupted after the maximum amount of loops (%s)" % max_iterations)
        loops = max_iterations

    # Reload archive if it is not supposed to be pscrunched.
    if not pscrunch and not memory:
        ar = psrchive.Archive_load(str(archive))

    # Set weights in archive.
    set_weights_archive(ar, avg_test_results)

    # Test if whole channel or subints should be removed
    if bad_chan_frac != 1 or bad_subint_frac != 1:
        ar = find_bad_parts(ar, bad_subint_frac=bad_subint_frac, bad_chan_frac=bad_chan_frac)

    # Unload residual if needed
    if unload_res:
        residual.unload("{0}_residual_{1}loops.ar".format(ar_name.rsplit('.', 1)[0], loops))

    # Create diagnostic plot showing weights and fraction channels/subints flagged
    if plot_zap:
        fig, (axW, axF, axS) = plt.subplots(nrows=3, gridspec_kw={'height_ratios': [1, 0.3, 0.3]},
                                            figsize=(8, 14))

        # plot 2D representation of weights
        weights = np.invert(ar.get_weights().T.astype(bool)).astype(float)
        axW.imshow(weights, aspect='auto', interpolation='none', cmap=plt.get_cmap('coolwarm'))
        axW.invert_yaxis()
        axW.set_title("%s cthresh=%s sthresh=%s" % (ar_name, chanthresh, subintthresh))
        axW.set_xlabel("Subintegration index")
        axW.set_ylabel("Channel index")

        frac_flagged_chan = np.sum(weights, axis=1) / float(patient_nsub)
        frac_flagged_sub = np.sum(weights, axis=0) / float(patient_nchan)

        # plot fraction of subints with particular channel flagged
        axF.scatter(np.arange(patient_nchan), frac_flagged_chan, marker="x")
        axF.set_xlabel("Channel index")
        axF.set_ylabel("frac. flagged")

        # plot fraction of channels with particular subint flagged
        axS.scatter(np.arange(patient_nsub), frac_flagged_sub, marker="x")
        axS.set_xlabel("Subintegration index")
        axS.set_ylabel("frac. flagged")

        plt.savefig("{0}_chan{1}_sub{2}.png".format(ar_name.rsplit('.', 1)[0], chanthresh, subintthresh),
                    bbox_inches='tight')
        plt.close(fig)

    # Create log that contains the used parameters
    #TODO: replace this and all output with logging module
    if log:
        with open("clean.log", "w+") as logfile:
            logfile.write(
                """%s:  Cleaned %s
                        required loops=%s
                        channel threshold=%f
                        subint threshold=%f
                        bad channel fraction threshold=%f
                        bad subint fraction threshold=%f
                        on-pulse region (start, end, scale_factor)=%s""" % (datetime.datetime.now(), ar_name, loops,
                                                                            chanthresh, subintthresh, bad_chan_frac,
                                                                            bad_subint_frac, pulse_region))

    ar.unload(str(output))

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
    test_results = np.nan_to_num(test_results, posinf=10, neginf=10, nan=10)  # replace any NaNs or infinities with >1

    return test_results



