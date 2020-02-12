#!/usr/bin/env python

import numpy as np
from scipy.optimize import least_squares


def remove_profile1d(prof, isub, ichan, template, pulse_region):
    """Given a specific profile and template, attempt to subtract a scaled version of the template from the data such
    that we are left only with the nominal baseline."""
    err = lambda amp: amp * template - prof
    result = least_squares(err, [1.0], method='lm')
    residuals = np.asarray(err(result.x))

    if pulse_region != [0, 1, 1]:
        p_start = int(pulse_region[1])
        p_end = int(pulse_region[2])
        residuals[p_start:p_end] = residuals[p_start:p_end] * pulse_region[-1]

    if not result.success:
        print("Bad status for least squares fit when removing profile.")
        return (isub, ichan), None
    else:
        return (isub, ichan), residuals


def remove_profile_inplace(ar, template, pulse_region):
    """Remove the template profile from the individual data profiles."""
    data = ar.get_data()[:, 0, :, :]  # Select first polarization channel
    # archive is P-scrunched, so this is
    # total intensity, the only polarization
    # channel
    for isub, ichan in np.ndindex(ar.get_nsubint(), ar.get_nchan()):
        amps = remove_profile1d(data[isub, ichan], isub, ichan, template, pulse_region)[1]
        prof = ar.get_Profile(isub, 0, ichan)
        if amps is None:
            prof.set_weight(0)
        else:
            prof.get_amps()[:] = amps


def apply_weights(data, weights):
    """Apply weights to an array."""
    nsubs, nchans, nbins = data.shape
    for isub in range(nsubs):
        data[isub] = data[isub] * weights[isub, ..., np.newaxis]

    return data


def set_weights_archive(archive, test_results):
    """Apply the weights to an archive according to the test results.
    This changes the archive INPLACE, so ensure that the provided archive is a copy of the original.
    """
    for (isub, ichan) in np.argwhere(test_results >= 1):
        integ = archive.get_Integration(int(isub))
        integ.set_weight(int(ichan), 0.0)


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


def find_bad_parts(archive, bad_subint_frac=1, bad_chan_frac=1):
    """Checks whether whole channels or subints should be removed
    """
    weights = archive.get_weights()
    n_subints = archive.get_nsubint()
    n_channels = archive.get_nchan()
    n_bad_channels = 0
    n_bad_subints = 0

    for i in range(n_subints):
        bad_frac = 1 - np.count_nonzero(weights[i, :]) / float(n_channels)
        if bad_frac > bad_subint_frac:
            for j in range(n_channels):
                integ = archive.get_Integration(int(i))
                integ.set_weight(int(j), 0.0)
            n_bad_subints += 1

    for j in range(n_channels):
        bad_frac = 1 - np.count_nonzero(weights[:, j]) / float(n_subints)
        if bad_frac > bad_chan_frac:
            for i in range(n_subints):
                integ = archive.get_Integration(int(i))
                integ.set_weight(int(j), 0.0)
            n_bad_channels += 1

    if n_bad_channels + n_bad_subints != 0:
        print("Removed %s bad subintegrations and %s bad channels." % (n_bad_subints, n_bad_channels))

    return archive
