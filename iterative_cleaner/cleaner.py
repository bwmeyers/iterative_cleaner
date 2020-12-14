#!/usr/bin/env python

import logging
import psrchive

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from .utils import (remove_profile_inplace, set_weights_archive, channel_scaler, subint_scaler,
                    find_bad_parts, fft_rotate, get_template_profile_phase)

cleaner_log = logging.getLogger("iterative_cleaner.cleaner")


def plot_archive_mask(weights, nchan, nsub, name, out):
    fig, (axW, axF, axS) = plt.subplots(nrows=3, gridspec_kw={'height_ratios': [1, 0.3, 0.3]},
                                        figsize=(8, 14))
    # plot 2D representation of weights
    weights = weights.T.squeeze().astype(bool).astype(float)
    axW.imshow(weights, aspect='auto', interpolation='none', cmap=plt.get_cmap('coolwarm'))
    axW.invert_yaxis()
    axW.set_title("%s original weights" % (name))
    axW.set_xlabel("Subintegration index")
    axW.set_ylabel("Channel index")

    frac_flagged_chan = np.sum(weights, axis=1) / float(nsub)
    frac_flagged_sub = np.sum(weights, axis=0) / float(nchan)

    # plot fraction of subints with particular channel flagged
    axF.scatter(np.arange(nchan), frac_flagged_chan, marker="x")
    axF.set_xlabel("Channel index")
    axF.set_ylabel("frac. flagged")
    axF.set_ylim(-0.05, 1.05)

    # plot fraction of channels with particular subint flagged
    axS.scatter(np.arange(nsub), frac_flagged_sub, marker="x")
    axS.set_xlabel("Subintegration index")
    axS.set_ylabel("frac. flagged")
    axS.set_ylim(-0.05, 1.05)

    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return


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
    cleaner_log.debug("computing diagnostic values for each subintegration and channel")
    diagnostics = []
    for func in diagnostic_functions:
        diagnostics.append(func(data, axis=2))

    # Now step through data and identify bad profiles
    cleaner_log.debug("scaling diagnostic values")
    scaled_diagnostics = []
    for diag in diagnostics:
        chan_scaled = np.abs(channel_scaler(diag)) / cthresh
        subint_scaled = np.abs(subint_scaler(diag)) / sthresh
        scaled_diagnostics.append(np.max((chan_scaled, subint_scaled), axis=0))

    cleaner_log.debug("reducing diagnostics to median values")
    test_results = np.median(scaled_diagnostics, axis=0)  # we could be more extreme and take the min/max
    test_results[~np.isfinite(test_results)] = 10

    return test_results


def clean(archive, template=None, output="cleaned.ar",
          max_iter=10, chanthresh=5, subintthresh=5, bad_chan_frac=1, bad_subint_frac=1, onpulse=None,
          unload_res=True, memory=False, pscrunch=True, plot_zap=True):

    if archive is None:
        cleaner_log.error("No archive provided")
        return

    if onpulse is None:
        onpulse = [None, None]

    ar = psrchive.Archive_load(str(archive))
    ar.convert_state(b'Stokes')
    ar_name = ar.get_filename().split()[-1]
    patient_nchan = ar.get_nchan()
    patient_nsub = ar.get_nsubint()
    patient_nbin = ar.get_nbin()
    mjd = (float(ar.start_time().strtempo()) + float(ar.end_time().strtempo())) / 2.0
    name = ar.get_source()
    cent_freq = ar.get_centre_frequency()

    cleaner_log.info("Loaded archive {0}...".format(str(archive)))
    cleaner_log.debug("source name: {0}".format(name))
    cleaner_log.debug("centre frequency: {0:.3f} MHz".format(cent_freq))
    cleaner_log.debug("MJD of mid-point: {0}".format(mjd))

    orig_weights = ar.get_weights()
    orig_weights_mask = np.bitwise_not(np.expand_dims(orig_weights, 2).astype(bool))

    if plot_zap:
        cleaner_log.debug("plotting initial archive mask")
        orig_out_fig = "{0}_orig_weights.png".format(name.rsplit('.', 1)[0])
        plot_archive_mask(orig_weights_mask, patient_nchan, patient_nsub, ar_name, orig_out_fig)

    if memory and not pscrunch:
        pass
    else:
        ar.pscrunch()

    patient = ar.clone()

    # construct a slice object from the given pulse_region
    pulse_region_slc = slice(*[int(x * patient_nbin) if (x is not None) else None for x in onpulse])

    mask_3d = orig_weights_mask.repeat(patient_nbin, axis=2)
    if pulse_region_slc != slice(None, None):
        cleaner_log.info("On-pulse region provided (bins) = {0}".format(pulse_region_slc))
        mask_3d[..., pulse_region_slc] = True
        cleaner_log.debug("masked on-pulse region")

    x = 0
    max_iterations = max_iter

    # Create list that is used to end the iteration
    test_weights = []
    test_weights.append(patient.get_weights())
    profile_number = orig_weights.size

    # Try to load template and ensure it will work with the data (i.e. has the same number of channels)
    template_from_file = False
    offset = 0
    if template is not None and template != "data":
        cleaner_log.info("Loading template from {0}".format(template))
        template_from_file = True
        # Assuming the template is an archive, too
        temp = psrchive.Archive_load(str(template))
        temp.pscrunch()
        temp.remove_baseline()
        temp.dedisperse()
        temp.tscrunch()

        temp_nchan = len(temp.get_frequencies())
        if temp_nchan == 1:
            cleaner_log.debug("template is 1D")
        elif temp_nchan != patient_nchan and temp_nchan > 1:
            cleaner_log.debug("template is 2D, but...")
            cleaner_log.warning("Number of channels in template ({0}) doesn't match data ({1})... "
                                "f-scrunching!".format(temp_nchan, patient_nchan))
            temp.fscrunch()

        # To estimate the phase offset between template and profile, first we should make a 1D template
        cleaner_log.info("Calculating phase offset between template and data.")
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
        amp_guess = np.abs(max(profile) - max(temp_phs))
        phase_guess = -np.argmax(temp_phs) + np.argmax(profile)
        amp, offset = get_template_profile_phase(temp_phs, profile, amp_guess=amp_guess, phase_guess=phase_guess)
        cleaner_log.debug("template phase offset = {0} bins".format(offset))
        temp = np.squeeze(temp.get_data()[0, 0, :, :])
    elif template == "data":
        # Create an initial template from the data if one is not provided
        # TODO: This doesn't actually make sense, since all that will end up happening is subtraction of a scaled
        #       version of the data, which doesn't even leave you with real residuals...
        #       the template has to be smooth and have constant baseline (for each channel at least)
        cleaner_log.info("No template file provided, will iteratively create one from data.")
        patient.pscrunch()
        patient.remove_baseline()
        patient.dedisperse()
        patient.fscrunch()
        patient.tscrunch()
        temp = savgol_filter(patient.get_Profile(0, 0, 0).get_amps(), 11, 3)
        temp = temp / temp.max()
        temp_nchan = 1
        # No need to phase align here since it's created from the data
    else:
        cleaner_log.info("Will not attempt to remove pulse profile from data prior to RFI mitigation.")
        temp = None
        temp_nchan = 0

    cleaner_log.info("Total number of profiles: {0}".format(profile_number))
    loops = 0
    while x < max_iterations:
        x += 1

        cleaner_log.debug("Loop: {0}".format(x))
        # Reset patient
        patient = ar.clone()
        patient.pscrunch()
        patient.remove_baseline()
        patient.dedisperse()

        if temp_nchan > 1 and template_from_file:
            # 2D template, needs to be rotated
            cleaner_log.debug("rotating 2D template")
            rotated_template = np.apply_along_axis(fft_rotate, 0, temp, offset)
        elif temp_nchan == 1 and template_from_file:
            # 1D template, needs to be rotated
            cleaner_log.debug("rotating 1D template")
            rotated_template = fft_rotate(temp, offset)
        else:
            # 1D template made from data, no rotation OR no template
            rotated_template = temp

        if rotated_template is not None:
            cleaner_log.info("Extracting scaled template from data.")
            remove_profile_inplace(patient, rotated_template)
            cleaner_log.info("... finished")

        # re-set DM to 0, more sensitive to RFI this way
        patient.dededisperse()

        if unload_res:
            residual = patient.clone()

        # Get data (select first polarization - recall we already P-scrunched)
        cleaner_log.debug("Getting archive weights and masking data")
        data = patient.get_data()[:, 0, :, :]
        # data = apply_weights(data, orig_weights)  # No point apply weights and then just masking the array
        data = np.ma.masked_array(data, mask=mask_3d)

        # RFI-ectomy must be recommended by average of tests
        cleaner_log.debug("Computing comprehensive statistics on channel/subint basis")
        avg_test_results = comprehensive_stats(data, cthresh=chanthresh, sthresh=subintthresh)

        # Reset patient and set weights in patient
        del patient
        patient = ar.clone()
        cleaner_log.debug("Zapping data in working copy...")
        set_weights_archive(patient, avg_test_results)

        # Test whether weights were already used in a previous iteration
        new_weights = patient.get_weights()
        diff_weights = np.sum(new_weights != test_weights[-1])
        rfi_frac = (new_weights.size - np.count_nonzero(new_weights)) / float(new_weights.size)

        # Print the changes to the previous loop to help in choosing a suitable max_iter
        cleaner_log.info("Loop {0}, differences to previous weights: {1}".format(x, diff_weights))
        cleaner_log.info("Loop {0}, RFI fraction: {1}".format(x, rfi_frac))
        for old_weights in test_weights:
            if np.all(new_weights == old_weights):
                cleaner_log.debug("RFI removal stops after {0} loops.".format(loops))
                loops = x
                x = 1000000
        test_weights.append(new_weights)

        if template == "data":
            # We need to recreate the template from the data if it was never defined in the first place
            # This makes sure that each iteration will have a better template than before
            cleaner_log.debug("re-creating template from this iteration's cleaned data")
            patient.pscrunch()  # pscrunching again is not necessary if already pscrunched but prevents a bug
            patient.remove_baseline()
            patient.dedisperse()
            patient.fscrunch()
            patient.tscrunch()
            temp = savgol_filter(patient.get_Profile(0, 0, 0).get_amps(), 11, 3)
            temp = temp / temp.max()
            # No need to phase align since it's created from the data

    if x == max_iterations:
        cleaner_log.warning("Cleaning was interrupted after the maximum amount of loops (%s)" % max_iterations)
        loops = max_iterations

    # Reload archive if it is not supposed to be pscrunched.
    if not pscrunch and not memory:
        cleaner_log.debug("Reloading full poln. archive")
        ar = psrchive.Archive_load(str(archive))

    # Set weights in archive
    cleaner_log.info("Updating weights in final archive")
    set_weights_archive(ar, avg_test_results)

    # Test if whole channel or subints should be removed
    if bad_chan_frac != 1 or bad_subint_frac != 1:
        cleaner_log.info("Masking addition data based on bad channel/subint fraction thresholds")
        ar = find_bad_parts(ar, bad_subint_frac=bad_subint_frac, bad_chan_frac=bad_chan_frac)

    # Unload residual if needed
    if unload_res:
        cleaner_log.debug("Unloading residual archive")
        residual.unload("{0}_residual_{1}loops.ar".format(ar_name.rsplit('.', 1)[0], loops))

    # Create diagnostic plot showing weights and fraction channels/subints flagged
    if plot_zap:
        cleaner_log.info("Plotting results of cleaning process")
        out_final_fig = "{0}_chan{1}_sub{2}.png".format(ar_name.rsplit('.', 1)[0], chanthresh, subintthresh)
        plot_archive_mask(ar.get_weights(), patient_nchan, patient_nsub, ar_name, out_final_fig)

    cleaner_log.info("Cleaned archive unloaded as: {0}".format(output))
    ar.unload(str(output))

    return ar




