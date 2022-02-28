#!/usr/bin/env python
import numpy as np


def mask_params_from_array(maskarr):
    """
    Input: 2D mask array of shape (nint, nchan), where the channels follow rfifind convention (index 0 corresponds to the lowest frequency channel)
    (Masking convention: 1/True means mask, 0/False means don't mask)
    Output: mask_zap_chans, mask_zap_ints, mask_zap_chans_per_int which can be used to write a presto .mask file
    """
    nint, nchan = maskarr.shape

    chans_to_mask = np.arange(nchan)[maskarr.sum(axis=0) == nint].astype(np.int32)
    ints_to_mask = np.arange(nint)[maskarr.sum(axis=1) == nchan].astype(np.int32)

    chans_per_int = []
    for i in range(nint):
        chans_per_int.append(np.where(maskarr[i, :] == 1)[0].astype(np.int32))

    return chans_to_mask, ints_to_mask, chans_per_int


def write_mask_file(filename, maskarr, header):
    """Write a mask numpy array as a rfifind .mask file
    filename: filename to write mask to (will add a .mask extension if not present)
    maskarr: 2D mask array of shape (nint, nchan)
             where the channels follow rfifind convention (index 0 corresponds to the lowest frequency channel)
             and 1/True in the array means mask, 0/False means don't mask
    header: dictionary which must contain keys:
        'time_sig' (float) - from rfifind options, PRESTO default is 10
        'freq_sig' (float) - from rfifind options, PRESTO default is 4
        'MJD' (float) - starting MJD of the observation
        'dtint' (float) - length of one time interval in seconds
        'lofreq' (float) - center frequency of lowest channel
        'df' (float) - width of one frequency channel
        'nchan' (int) - number of frequency channels in the mask
        'nint' (int) - number of time intervals in the mask
        'ptsperint' (int) - number of time samples per interval
    """
    # Massage inputs
    if filename[-5:] != ".mask":
        filename += ".mask"

    header_params = [
        np.array(header["time_sig"], dtype=np.float64),
        np.array(header["freq_sig"], dtype=np.float64),
        np.array(header["MJD"], dtype=np.float64),
        np.array(header["dtint"], dtype=np.float64),
        np.array(header["lofreq"], dtype=np.float64),
        np.array(header["df"], dtype=np.float64),
        np.array(header["nchan"], dtype=np.int32),
        np.array(header["nint"], dtype=np.int32),
        np.array(header["ptsperint"], dtype=np.int32),
    ]

    # Check maskarr shape matches nint and nchan in header
    if header["nint"] != maskarr.shape[0]:
        raise ValueError(
            f"nint in header ({header['nint']}) does not match maskarr shape {maskarr.shape}"
        )
    if header["nchan"] != maskarr.shape[1]:
        raise ValueError(
            f"nchan in header ({header['nchan']}) does not match maskarr shape {maskarr.shape}"
        )

    # Write to file
    with open(filename, "wb") as fout:
        for variable in header_params:
            variable.tofile(fout)

        zap_chans, zap_ints, zap_chans_per_int = mask_params_from_array(maskarr)

        nzap_chan = np.asarray(zap_chans.size, dtype=np.int32)
        nzap_chan.tofile(fout)
        if nzap_chan:
            zap_chans.tofile(fout)

        nzap_int = np.asarray(zap_ints.size, dtype=np.int32)
        nzap_int.tofile(fout)
        if nzap_int:
            zap_ints.tofile(fout)

        nzap_per_int = []
        for an_arr in zap_chans_per_int:
            nzap_per_int.append(an_arr.size)
        if len(nzap_per_int) != header["nint"]:
            raise AttributeError("BUG: nzap_per_int should be of length nint!")
        nzpi = np.asarray(nzap_per_int, dtype=np.int32)
        nzpi.tofile(fout)

        # rfifind.py only calls fromfile if nzap != 0 and nzap != nchan
        for i in range(header["nint"]):
            if nzap_per_int[i]:
                if nzap_per_int[i] != header["nchan"]:
                    zap_chans_per_int[i].tofile(fout)
