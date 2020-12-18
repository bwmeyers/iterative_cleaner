#!/usr/bin/env python

import logging
import numpy as np
from scipy.optimize import least_squares


utils_log = logging.getLogger('iterative_cleaner.utils')


def fft_rotate(data, bins):
    """
    Return data rotated by 'bins' places to the left.
    The rotation is done in the Fourier domain using the Shift Theorem.

    NOTE: This can leave artifacts in the rotated template if the number
          of bins rotated is not an integer

        Inputs:
            data: A 1-D numpy array to rotate.
            bins: The (possibly fractional) number of bins to rotate by.
        Outputs:
            rotated: The rotated data.
    """
    freqs = np.arange(data.size // 2 + 1, dtype=np.float)
    phasor = np.exp(complex(0.0, 2.0 * np.pi) * freqs * bins / float(data.size))

    return np.fft.irfft(phasor * np.fft.rfft(data))


def subtract_scaled_rotated_template(params, temp, prof, rot=False):
    amp, off = params
    if rot:
        return prof - amp * fft_rotate(temp, off)
    else:
        return prof - amp * temp


def get_template_profile_phase(template, prof, amp_guess=1.0, phase_guess=0.0):
    """ Given a template and profile, estimate the phase offset between the two"""
    result = least_squares(subtract_scaled_rotated_template, [amp_guess, phase_guess],
                           args=(template, prof), kwargs={'rot': True}, loss='soft_l1')
    if not result.success:
        utils_log.warning("Bad status for least squares fit when rotating template.")
        best_amp = amp_guess
        best_phase = phase_guess
    else:
        utils_log.debug("Template rotation fit terminated successfully.")
        best_amp, best_phase = result.x

    return best_amp, best_phase


def remove_profile1d(prof, template, isub, ichan):
    """Given a specific profile and template, attempt to subtract a scaled version of the template from the data such
    that we are left only with the nominal baseline."""
    result = least_squares(subtract_scaled_rotated_template, [0.1 * prof[np.argmax(template)], 0.0],
                           args=(template, prof), kwargs={'rot': False}, loss='soft_l1')

    if not result.success:
        utils_log.warning(
            "isub={0} ichan={1} Bad status for least squares fit when removing profile.".format(isub, ichan))
        return None, result.x
    else:
        return result.fun, result.x


def remove_profile_inplace(ar, template):
    """Remove the template profile from the individual data profiles."""
    data = ar.get_data()[:, 0, :, :]  # Select first polarization channel
    # archive is P-scrunched, so this is total intensity
    nchan = ar.get_nchan()
    nsub = ar.get_nsubint()

    for isub, ichan in np.ndindex(nsub, nchan):
        if len(template.shape) > 1:
            amps, fit_res = remove_profile1d(data[isub, ichan], template[ichan], isub, ichan)
        else:
            amps, fit_res = remove_profile1d(data[isub, ichan], template, isub, ichan)

        prof = ar.get_Profile(isub, 0, ichan)
        if amps is None:
            utils_log.warning("isub={0} ichan={1} Zero-weighting profile".format(isub, ichan))
            prof.set_weight(0)
        else:
            prof.get_amps()[:] = amps


def apply_weights(data, weights):
    """Apply weights to an array."""
    nsubs, nchans, nbins = data.shape
    for isub in range(nsubs):
        data[isub] = data[isub] * weights[isub, ..., np.newaxis]

    return data


def update_weights(weights, test_results):
    """Update a weights array according to the test results.
    """
    updated_weights = np.copy(weights)
    for (isub, ichan) in np.argwhere(test_results >= 1):
        # utils_log.debug(f"subint {isub}, channel {ichan} failed tests")
        updated_weights[isub, ichan] = 0.0

    return updated_weights


def update_archive_weights(archive, weights):
    """Update the weights in an archive (in place!)"""
    for isub, ichan in np.ndindex(weights.shape):
        integration = archive.get_Integration(isub)
        integration.set_weight(ichan, float(weights[isub, ichan]))


def channel_scaler(array2d):
    """For each channel scale it by the median absolute deviation"""
    scaled = np.empty_like(array2d)
    nchans = array2d.shape[1]

    for ichan in np.arange(nchans):
        with np.errstate(invalid='ignore', divide='ignore'):
            channel = array2d[:, ichan]
            median = np.ma.median(channel)
            channel_rescaled = channel - median
            mad = np.ma.median(np.abs(channel_rescaled))
            scaled[:, ichan] = channel_rescaled / mad

    return scaled


def subint_scaler(array2d):
    """For each sub-int scale it by the median absolute deviation"""
    scaled = np.empty_like(array2d)
    nsubs = array2d.shape[0]

    for isub in np.arange(nsubs):
        with np.errstate(invalid='ignore', divide='ignore'):
            subint = array2d[isub, :]
            median = np.ma.median(subint)
            subint_rescaled = subint - median
            mad = np.ma.median(np.abs(subint_rescaled))
            scaled[isub, :] = subint_rescaled / mad

    return scaled


def find_bad_parts(weights, bad_subint_frac=1, bad_chan_frac=1):
    """Checks whether whole channels or subints should be removed
    """
    updated_weights = np.copy(weights)

    # check for largely corrupted subintegrations
    subint_weights = weights.mean(axis=1)
    n_bad_subints = len(np.argwhere(subint_weights <= 1 - bad_subint_frac))
    utils_log.info(f"Number of removed subints: {n_bad_subints}")
    utils_log.info(
        f"subints flagged: "
        f"{np.where(subint_weights <= 1 - bad_subint_frac)[0]}"
    )
    updated_weights[subint_weights <= 1 - bad_subint_frac, :] = 0

    # check for largely corrupted channels
    channel_weights = weights.mean(axis=0)
    n_bad_channels = len(np.argwhere(channel_weights <= 1 - bad_chan_frac))
    utils_log.info(f"Number of removed channels: {n_bad_channels}")
    utils_log.info(
        f"channels flagged: "
        f"{np.where(channel_weights <= 1 - bad_chan_frac)[0]}"
    )
    updated_weights[:, channel_weights <= 1 - bad_chan_frac] = 0

    return updated_weights

    # for i in range(n_subints):
    #     bad_frac = 1 - np.count_nonzero(weights[i, :]) / float(n_channels)
    #     if bad_frac > bad_subint_frac:
    #         for j in range(n_channels):
    #             integ = archive.get_Integration(int(i))
    #             integ.set_weight(int(j), 0.0)
    #         n_bad_subints += 1
    #
    # for j in range(n_channels):
    #     bad_frac = 1 - np.count_nonzero(weights[:, j]) / float(n_subints)
    #     if bad_frac > bad_chan_frac:
    #         for i in range(n_subints):
    #             integ = archive.get_Integration(int(i))
    #             integ.set_weight(int(j), 0.0)
    #         n_bad_channels += 1

    # if n_bad_channels + n_bad_subints != 0:
    #     utils_log.info("Removed %s bad subintegrations and %s bad channels." % (n_bad_subints, n_bad_channels))
    #
    # return archive
