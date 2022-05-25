import numpy as np
from presto_without_presto import rfifind, sigproc
from sigproc_utils import get_dtype, get_nbits, write_header, get_fmin_fmax_invert

# from presto import rfifind, sigproc
import copy
import time
import argparse
import sys, os
import logging
from gen_utils import handle_exception

################################################################################
############################ DEFINE FUNCTIONS ETC ##############################

################################################################################
# GENERAL FUNCTIONS:
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]


def zero_or_none(thing):
    """returns True if thing is 0/[]/""/etc or None
    returns False for anything else"""
    if thing is not None:
        return not bool(thing)
    else:
        return True


def not_zero_or_none(thing):
    """returns False if thing is 0/[]/""/etc or None
    returns True for anything else, aka value isn't empty/0/None"""
    if thing is not None:
        return bool(thing)
    else:
        return False


def check_positive_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(
            "%s is an invalid positive float value" % value
        )
    return fvalue


def try_remove(thing, from_list):
    try:
        from_list.remove(thing)
    except ValueError:
        pass


################################################################################
# MASKING FUNCTIONS:


def array_from_mask_params(nint, nchan, zap_ints, zap_chans, zap_chans_per_int):
    """Return the mask as a numpy array of size (nint, nchan)
    (1 = masked, 0 = unmasked)

    nint = number of intervals
    nchan = number of channels
    zap_ints (set) - intervals to zap
    zap_chans (set) - channels to zap
    zap_chans_per_int - list (of length nint) of sets such that
                        zap_chans_per_int[i] gives a set of channels to mask in interval i

    NB if using original rfifind.py note that some type conversion will be needed as
    mask_zap_ints is an array and mask_zap_chans_per_int is a list of arrays
    """
    mask = np.zeros((nint, nchan), dtype=bool)
    if len(zap_ints) != 0:
        mask[tuple(zap_ints), :] = 1
    if len(zap_chans) != 0:
        mask[:, tuple(zap_chans)] = 1
    for i in range(nint):
        if len(zap_chans_per_int[i]) != 0:
            mask[i, tuple(zap_chans_per_int[i])] = 1

    return mask


class Mask:
    def __init__(self, maskfile, invertband=True):
        """invertband=True for sigproc filterbank convention where highest frequency is first"""
        rfimask = rfifind.rfifind(maskfile)
        self.invertband = invertband
        self.nchan = rfimask.nchan
        self.nint = rfimask.nint
        self.ptsperint = rfimask.ptsperint
        self.dtint = rfimask.dtint
        # original rfimask has this as an array, convert to set if necessary
        self.mask_zap_ints = set(rfimask.mask_zap_ints)
        if invertband:
            self.mask_zap_chans = set(
                list(self.nchan - 1 - np.array(list(rfimask.mask_zap_chans)))
            )
        else:
            self.mask_zap_chans = rfimask.mask_zap_chans
        # original rfimask has this as a list of arrays, convert to set if necessary
        if invertband:
            self.mask_zap_chans_per_int = [
                set(list(self.nchan - 1 - np.array(list(x))))
                for x in rfimask.mask_zap_chans_per_int
            ]
        else:
            self.mask_zap_chans_per_int = [
                set(x) for x in rfimask.mask_zap_chans_per_int
            ]

        self.mask = array_from_mask_params(
            self.nint,
            self.nchan,
            self.mask_zap_ints,
            self.mask_zap_chans,
            self.mask_zap_chans_per_int,
        )


def clip(
    intensities,
    clip_sigma,
    running_avg_std=None,
    chan_running_avg=None,
    droptot_sig=None,
    chan_running_std=None,
):
    """
    Attempt to replicate presto's clip_times in clipping.c
    Time-domain clpping of raw data, based on the zero-DM time series.
    Channel averages are calculated only using points that are within 3 std of the median

    Input:
        intensities: Raw data with shape (ptsperint, numchan)
        clip_sigma: Clipping is done at clip_sigma above/below the (pseudo) running mean
            (if don't want to actually do the clipping, pass in 0 or None)
        running_avg_std: [running_avg, running_std], stores running average and std of DM0 time series
        chan_running_avg: a numpy array of length numchan, stores running average of each channel
        droptot_sig, chan_running_std: see last paragraph


    Returns:
        the clipped intensities, dict(running_avg_std=[running_avg, running_std], chan_running_avg=chan_running_avg)
        The latter two can then be passed on to the next block

    Example usage:
    running_dict = dict()
    read in intensities_block0
    intensities_block0, running_dict = clip(intensities_block0, 3, **running_dict)
    # do other stuff
    read in intensities_block1
    intensities_block1, running_dict = clip(intensities_block1, 3, **running_dict)


    droptot_sig and chan_running_std are for CHIME filterbank data where dropped packets weren't handled perfectly
    This resulted in times where intensities would suddenly plummet, but not always to 0, in 4 channels within the band.
    A dip in 4 channels appears to mostly not set off the normal clipping routine, so they need to be clipped independently.
    This is done AFTER the normal clipping.
    Data < chan_running_avg - droptot_sigma * chan_running_std are clipped
    They are filled with the running channel average (aka the same values as for normal clipping)

    Example usage is exactly the same as above, just add in e.g. droptot_sigma=4.5
    """
    ptsperint, numchan = intensities.shape

    if running_avg_std is None and chan_running_avg is None:
        first_block = True
        running_avg = 0
        running_std = 0
        chan_running_avg = np.zeros((numchan))
        chan_running_std = np.zeros((numchan))
    else:
        first_block = False
        if running_avg_std is None or chan_running_avg is None:
            raise RuntimeError(
                "running_avg_std and chan_running_avg should either both be None (when it's the first block) or neither"
            )
        running_avg, running_std = running_avg_std

    # Calculate the zero DM time series
    zero_dm_time_series = intensities.sum(axis=-1)
    current_avg = zero_dm_time_series.mean()
    current_std = zero_dm_time_series.std()
    current_med = np.median(zero_dm_time_series)

    # Calculate the current standard deviation and mean
    # but only for data points that are within a certain
    # fraction of the median value.  This removes the
    # really strong RFI from the calculation.
    lo_cutoff = current_med - 3.0 * current_std
    hi_cutoff = current_med + 3.0 * current_std
    chan_avg_temp = np.zeros((numchan))

    # Find the "good" points
    good_pts_idx = np.where(
        (zero_dm_time_series > lo_cutoff) & (zero_dm_time_series < hi_cutoff)
    )[0]
    numgoodpts = good_pts_idx.size

    if numgoodpts < 1:
        current_avg = running_avg
        current_std = running_std
        chan_avg_temp = chan_running_avg
        chan_std_temp = chan_running_std
    else:
        current_avg = zero_dm_time_series[good_pts_idx].mean()
        current_std = zero_dm_time_series[good_pts_idx].std()
        chan_avg_temp = intensities[good_pts_idx, :].mean(axis=0)
        chan_std_temp = intensities[good_pts_idx, :].std(
            axis=0
        )  # for CHIME dropped-packet clipping

    # Update a pseudo running average and stdev
    # (exponential moving average)
    if not first_block:
        running_avg = 0.9 * running_avg + 0.1 * current_avg
        running_std = 0.9 * running_std + 0.1 * current_std
        chan_running_avg = 0.9 * chan_running_avg + 0.1 * chan_avg_temp
        if droptot_sig is not None:
            chan_running_std = 0.9 * chan_running_std + 0.1 * chan_std_temp
    else:
        running_avg = current_avg
        running_std = current_std
        chan_running_avg = chan_avg_temp
        chan_running_std = chan_std_temp  # for CHIME dropped-packet clipping
        if current_avg == 0:
            logging.warning("Warning: problem with clipping in first block!!!\n\n")

    # See if any points need clipping
    if not_zero_or_none(clip_sigma):
        trigger = clip_sigma * running_std
        where_clip = np.where((np.abs(zero_dm_time_series) - running_avg) > trigger)[0]

        # Replace the bad channel data with running channel average
        if where_clip.size:
            intensities[
                where_clip, :
            ] = chan_running_avg  # this edits intensities in place, might want to change

    # CHIME dropped-packet clipping
    if not_zero_or_none(droptot_sig):
        where_droptot_clip = np.where(
            intensities < (chan_running_avg - droptot_sig * chan_running_std)
        )
        intensities[where_droptot_clip] = chan_running_avg[where_droptot_clip[1]]
        return intensities, dict(
            running_avg_std=[running_avg, running_std],
            chan_running_avg=chan_running_avg,
            chan_running_std=chan_running_std,
        )
    else:
        return intensities, dict(
            running_avg_std=[running_avg, running_std],
            chan_running_avg=chan_running_avg,
        )


def clip_mask_subbase_gulp(
    nint,
    ptsperint,
    data,
    running_dict,
    clipsig,
    droptotsig,
    gulp,
    maxDT,
    current_int,
    mask,
):
    """Clip, mask and subtract the running_avg from a gulp"""
    for interval in range(nint):
        try:
            slc = slice(interval * ptsperint, (interval + 1) * ptsperint)
            data[slc, :], running_dict = clip(
                data[slc, :],
                clipsig,
                droptot_sig=droptotsig,
                **running_dict,
            )
        except IndexError:  # in case on leftover partial-interval  # this seems to not get activated ever?
            logging.debug(
                f"Last interval detected: length {data.shape[0]} where gulp is {gulp} and maxDT {maxDT}",
            )
            slc = slice(interval * ptsperint, None)
            data[slc, :], running_dict = clip(
                data[slc, :],
                clipsig,
                droptot_sig=droptotsig,
                **running_dict,
            )

        data[slc, :] -= running_dict["chan_running_avg"]
        data[slc, list(mask.mask_zap_chans_per_int[current_int])] = 0
        current_int += 1


def clip_subbase_gulp(
    nint,
    ptsperint,
    data,
    running_dict,
    clipsig,
    droptotsig,
    gulp,
    maxDT,
    current_int,
    *args,
):
    """Same as clip_mask_subbase_gulp but no mask"""
    for interval in range(nint):
        try:
            slc = slice(interval * ptsperint, (interval + 1) * ptsperint)
            data[slc, :], running_dict = clip(
                data[slc, :],
                clipsig,
                droptot_sig=droptotsig,
                **running_dict,
            )
        except IndexError:  # in case on leftover partial-interval  # this seems to not get activated ever?
            logging.debug(
                f"Last interval detected: length {data.shape[0]} where gulp is {gulp} and maxDT {maxDT}",
            )
            slc = slice(interval * ptsperint, None)
            data[slc, :], running_dict = clip(
                data[slc, :],
                clipsig,
                droptot_sig=droptotsig,
                **running_dict,
            )

        data[slc, :] -= running_dict["chan_running_avg"]
        current_int += 1


################################################################################
# DM FUNCTIONS:


def DM_delay(DM, freq, *ref_freq):
    """
    Pass in DM (cm-3pc), freq (MHz), and optionally a reference frequency
    Returns time delay in seconds
    Uses Manchester-Taylor 1/2.4E-4 convention
    """
    K = 1.0 / 2.41e-4  # Manchester-Taylor convention, units in MHz cm-3pc s
    if ref_freq:
        return K * DM * (1 / freq**2 - 1 / ref_freq[0] ** 2)
    else:
        return K * DM / freq**2


def inverse_DM_delay(delay_s, freq, *ref_freq):
    """Inverse of DM_delay
    Pass in delay (s), freq (MHz), and optionally a reference frequency
    (if not included it's referenced to infinite frequency)
    Returns the DM which would produce that delay in cm-3pc
    Uses Manchester-Taylor 1/2.4E-4 convention"""
    K = 1.0 / 2.41e-4  # Manchester-Taylor convention, units in MHz cm-3pc s
    if ref_freq:
        return delay_s / (1 / freq**2 - 1 / ref_freq[0] ** 2) / K
    else:
        return delay_s * freq**2 / K


def get_fs(fmin, fmax, nchans, type="center", invertband=True, **kwargs):
    """Get channel frequencies for a band with edges fmin, fmax
    type can be "center", "lower", "upper"

    Done via np.linspace, can pass in retstep=True.

    Returns a numpy array
    starting with the lowest-freq channel if invertband = False
    starting with the highest-freq channel if invertband = True
    (invertband = True is the default as that's sigproc filterbank convention)
    """
    df = (fmax - fmin) / nchans
    if type == "lower":
        fs = np.linspace(fmin, fmax, nchans, endpoint=False, **kwargs)
    elif type == "center":
        fs = np.linspace(fmin + df / 2, fmax - df / 2, nchans, endpoint=True, **kwargs)
    elif type == "upper":
        fs = np.linspace(fmin + df, fmax, nchans, endpoint=True, **kwargs)
    else:
        raise AttributeError(
            f"type {type} not recognised, must be one of 'center', 'lower', 'upper'"
        )

    if invertband:
        return fs[::-1]
    else:
        return fs


def round_to_samples(delta_t, sampling_time):
    """Round a time delay to the nearest number of time samples"""
    return np.round(delta_t / sampling_time).astype(int)


def get_maxDT_DM(DM, maxDT, tsamp, fs):
    if fs[0] > fs[-1]:
        fhi = fs[0]
        flo = fs[-1]
    else:
        fhi = fs[-1]
        flo = fs[0]
    if not_zero_or_none(DM):
        outDM = DM
        max_delay_s = DM_delay(outDM, flo, fhi)
        outmaxDT = round_to_samples(max_delay_s, tsamp)
    elif not_zero_or_none(maxDT):
        outmaxDT = maxDT
        max_delay_s = outmaxDT * tsamp
        outDM = inverse_DM_delay(max_delay_s, flo, fhi)
    else:
        outDM = 0
        outmaxDT = 0
        max_delay_s = 0
    return outDM, outmaxDT, max_delay_s


def shift_and_stack(data, shifts, prev_array, maxDT):
    ilength, nchans = data.shape
    prev_array = copy.copy(prev_array)  # might be uneccessary, haven't checked
    mid_array_shape = (ilength - maxDT, nchans)
    mid_array = np.zeros(mid_array_shape, dtype=prev_array.dtype)
    end_array = np.zeros_like(prev_array)

    for i in range(0, nchans):
        dt = shifts[i]
        prev_array[maxDT - dt :, i] += data[:dt, i]
        mid_array[:, i] = data[dt : ilength - (maxDT - dt), i]
        end_array[: (maxDT - dt), i] = data[ilength - (maxDT - dt) :, i]

    return prev_array, mid_array, end_array


def approx_size_shifted_arrays(data, maxDT):
    nbytes = get_nbits(data.dtype) // 4
    ilength, nchans = data.shape
    prev_sz = nbytes * maxDT * nchans
    end_sz = prev_sz
    mid_sz = nbytes * (ilength - maxDT) * nchans
    return 2 * prev_sz + mid_sz + end_sz


def get_gulp(nsamples, ptsperint, maxDT, mingulp, desired_gulp):
    if mingulp == 0:  # DM = 0 case
        gulp = (int(desired_gulp // ptsperint) + 1)*ptsperint
        return gulp, 0
    if desired_gulp < mingulp:
        gulp = mingulp
        leftovers = nsamples % gulp
        if leftovers > maxDT:
            nsamp_cut_off = 0
        else:
            nsamp_cut_off = leftovers
        return gulp, nsamp_cut_off
    else:
        # What do we want here?
        # to make sure the last bit of data doesn't get cut off
        # ideally want a multiple of ptsperint AND that the last block has >maxDT samples
        all_intspergulp = np.arange((nsamples // ptsperint) + 1)
        ipg_over_maxDT = all_intspergulp[all_intspergulp >= mingulp / ptsperint]
        # check there are some possible values
        if ipg_over_maxDT.size == 0:
            raise RuntimeError(
                f"No possible gulp sizes over mingulp in {all_intspergulp*ptsperint}"
            )

        # number of time samples in final read
        leftovers = np.array([nsamples % (ptsperint * ipg) for ipg in ipg_over_maxDT])

        # divide exactly, quite unlikely
        good_ipg = ipg_over_maxDT[leftovers == 0]
        if good_ipg.size:
            logging.debug(f"Found gulps with no leftovers: {good_ipg*ptsperint}")
            ipg = find_nearest(good_ipg, desired_gulp / ptsperint)
            nsamp_cut_off = 0
            return ipg * ptsperint, nsamp_cut_off

        # leftover > maxDT
        good_ipg = ipg_over_maxDT[leftovers > maxDT]
        if good_ipg.size:
            logging.debug(
                    f"Found {good_ipg.size} gulps which preserve all the data: {good_ipg*ptsperint}",
                )
            ipg = find_nearest(good_ipg, desired_gulp / ptsperint)
            nsamp_cut_off = 0
            return ipg * ptsperint, nsamp_cut_off
        else:
            logging.debug(
                    f"No gulps preseve all the data, leftovers are all < maxDT and will be cut off"
                )
            # logging.debug(f"Picking gulp which minimizes leftover")
            # ipg = ipg_over_maxDT[leftovers == leftovers.min()]
            # if not isinstance(ipg, int):  # multiple options have the same leftover
            #    ipg = find_nearest(ipg, desired_gulp / ptsperint)
            logging.debug(
                    f"Gulp which minimizes leftover is {ipg_over_maxDT[leftovers == leftovers.min()]*ptsperint}",
                )
            ipg = find_nearest(ipg_over_maxDT, desired_gulp / ptsperint)
            nsamp_cut_off = leftovers[ipg_over_maxDT == ipg][0]
            return ipg * ptsperint, nsamp_cut_off


################################################################################
################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="""Incoherently dedisperse a filterbank file (in chunks), outputs another filterbank
        Note:
            - Uses the Manchester-Taylor 1/2.4E-4 convention for dedispersion
            - Aligns other channels to the frequency of the uppermost channel (fch1)
              (whether it's the center, upper edge or lower edge is given by <where_channel_ref>, default=center)
              such that tstart in the header is referenced to that same frequency
            - if gulp is optimized, the number of time samples in the output file will be N - maxdt
        Limitations:
            - untested on a reversed band (positive foff); I assume it'll break
            - not written to deal with multiple polarization data""",
    )
    parser.add_argument("filename", type=str, help="Filterbank file to dedisperse")
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=".",
        help="Directory in which to write the output",
    )
    parser.add_argument(
        "gulp",
        type=int,
        help="""Desired number of spectra (aka number of time samples) to read in at once
            This is gets optimized but will try to pick a gulp as close to the one passed in as possible""",
    )

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "-d",
        "--dm",
        type=float,
        default=0,
        help="DM (cm-3pc) to dedisperse to, must be positive",
    )
    g.add_argument(
        "-t",
        "--maxdt",
        type=check_positive_float,
        default=0,
        help="Number of time samples corresponding to the DM delay between the lowest and highest channel\n(must be positive)",
    )

    parser.add_argument(
        "--where_channel_ref",
        default="center",
        choices=["center", "lower", "upper"],
        help="Where within the channel ",
    )
    parser.add_argument(
        "--dmprec",
        type=int,
        default=2,
        help="DM precision (only used when writing filename)",
    )

    # masking options, this will definitely break for an inverted band
    parser.add_argument(
        "-m", "--mask", type=str, default="", help="rfifind .mask file to apply"
    )

    parser.add_argument(
        "--ignorechan",
        type=str,
        help="Comma separated string (no spaces!) of channels to ignore. PRESTO convention (lowest frequency channel = 0)",
        default="",
    )

    parser.add_argument(
        "--clipsig",
        type=float,
        default=6.0,
        help="sigma at which to clip the data a la presto (to turn off set to 0). Default: 6.0",
    )
    parser.add_argument(
        "--droptotsig",
        type=float,
        help="""CHIME filterbanks don't handle dropped packets well, this looks for where data is below running_avg - <dropsig> * std
                (set to 0 to turn this off). Default: 4.5""",
        default=4.5,
    )

    parser.add_argument(
        "--log", type=str, help="name of file to write log to", default=None,
    )

    parser.add_argument(
        '-v', '--verbose',
        help="Increase logging level to debug",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO,
    )


    args = parser.parse_args()

    if args.log is not None:
        logging.basicConfig(
            filename=args.log,
            filemode='w',
            format='%(asctime)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S',
            level=args.loglevel,
            )
    else:
        logging.basicConfig(
            format='%(asctime)s - %(message)s',
            datefmt='%d-%b-%y %H:%M:%S',
            level=args.loglevel,
            stream=sys.stdout,
            )

    # log unhandled exception
    sys.excepthook = handle_exception

    t0 = time.perf_counter()

    # being too lazy to refactor
    dmprec = args.dmprec
    where_channel_ref_freq = "center"

    # define gulp preprocessing based on args
    if args.clipsig or args.droptotsig or args.mask:
        if not_zero_or_none(args.mask):
            "Preprocess is: clipping/computing running averages, subtracting baseline, masking"

            def preprocess(*args, **kwargs):
                clip_mask_subbase_gulp(*args, **kwargs)

        else:
            "Preprocess is: clipping/computing running averages, subtracting baseline, NOT masking"

            def preprocess(*args, **kwargs):
                clip_subbase_gulp(*args, **kwargs)

    else:
        logging.info(
            "Preprocess is: NOT clipping/computing running averages, NOT subtracting baseline, NOT masking"
        )

        def preprocess(*args, **kwargs):
            pass

    logging.info(f"Working on file: {args.filename}")
    header, hdrlen = sigproc.read_header(args.filename)
    nsamples = int(sigproc.samples_per_file(args.filename, header, hdrlen))
    logging.debug(header)

    if header["nifs"] != 1:
        raise ValueError(f"Code not written to deal with unsummed polarization data")

    # calculate/get parameters from header
    tsamp = header["tsamp"]
    nchans = header["nchans"]

    # calculate fmin and fmax FROM HEADER
    fmin, fmax, invertband = get_fmin_fmax_invert(header)

    logging.info(
        f"fmin: {fmin}, fmax: {fmax}, nchans: {nchans} tsamp: {tsamp} nsamples: {nsamples}",
    )

    ## actual code
    # get ignorechans
    if not_zero_or_none(args.ignorechan):
        # convert string to list of ints, and invert so index 0 = highest freq channel as per sigproc filterbank convention
        ignorechans = set(
            [nchans - 1 - int(chan) for chan in args.ignorechan.split(",")]
        )
    else:
        ignorechans = set()

    # Load the mask
    if not_zero_or_none(args.mask):
        logging.info(f"Loading mask from {args.mask}")
        mask = Mask(args.mask)
        logging.info(f"Mask loaded")
        ptsperint = mask.ptsperint
        zerochans = mask.mask_zap_chans | ignorechans
        logging.info(f"ptsperint of {ptsperint} read from mask")
        # if data has been downsampled wrt mask, adjust ptsperint accordingly
        maskdt = mask.dtint / mask.ptsperint
        if np.isclose(maskdt, tsamp, atol=1e-10):
            logging.info(f"tsamp {tsamp} matches mask dt {maskdt}, no downsampling of ptsperint required")
        elif tsamp > maskdt:
            if np.isclose(tsamp % maskdt, 0):
                downsamp = tsamp / maskdt
                ptsperint /= downsamp
                if ptsperint % 1:
                    logging.error(f"Tried to downsample ptsperint {mask.ptsperint} by {downsamp} and did not get an integer ({ptsperint})")
                ptsperint = int(ptsperint)
                logging.info(f"tsamp {tsamp} is {downsamp} x maskdt ({maskdt}), downsampling ptsperint from {mask.ptsperint} to {ptsperint}")
            else:
                logging.error(f"tsamp > maskdt, but not by an integer factor. tsamp/maskdt = {tsamp}/{maskdt} = {tsamp/maskdt}")
        else:
            if np.isclose(maskdt % tsamp, 0):
                upsamp = maskdt / tsamp
                ptsperint = int(upsamp * ptsperint)
                logging.info(f"tsamp {tsamp} is 1/{upsamp} x maskdt ({maskdt}), upsampling ptsperint from {mask.ptsperint} to {ptsperint}")
            else:
                logging.error(f"maskdt > tsamp, but not by an integer factor. maskdt/tsamp = {maskdt}/{tsamp} = {maskdt/tsamp}")


        # check mask covers all data
        if not ((mask.nint - 1) * ptsperint) < nsamples <= (mask.nint * ptsperint):
            logging.error(f"Mask has {mask.nint} intervals and using {ptsperint} ptsperint. Data is {nsamples} samples but mask covers {(mask.nint - 1) * ptsperint} < samples <= {mask.nint * ptsperint}")

    else:
        mask = None
        ptsperint = 2400  # presto default
        zerochans = ignorechans
        logging.info(f"Using presto default for ptsperint")

    logging.info(f"Clipping etc will be done in intervals of {ptsperint}")

    logging.info(
        f"clipping etc will be done in intervals of {ptsperint} as per mask/presto default",
    )

    # Select gulp
    #######################################################################
    # Get the maximum brute force DM delay, and the delays for each channel #
    fs = get_fs(fmin, fmax, nchans, type=where_channel_ref_freq, invertband=invertband)

    DM, maxDT, max_delay_s = get_maxDT_DM(args.dm, args.maxdt, tsamp, fs)
    logging.info(f"Brute force incoherent DM is {DM}")
    logging.info(
        f"Maximum brute force incoherent DM delay need to shift by is {max_delay_s} s"
    )
    logging.info(f"This corresponds to {maxDT} time samples\n")
    if DM == 0:
        logging.warning("DM=0, will likely break")
    #    sys.exit("DM=0, why are you running this?")

    # Find minimum number of samples need to read in, must be a multiple of ptsperint
    if maxDT % ptsperint:
        mingulp = ((maxDT // ptsperint) + 1) * ptsperint
    else:
        mingulp = maxDT

    logging.info(
        f"Minimum gulp is {mingulp} time samples (= {mingulp / ptsperint:.1f} intervals)",
    )

    gulp, nsamp_cut_off = get_gulp(
        nsamples, ptsperint, maxDT, mingulp, args.gulp,
    )
    logging.info(f"Selected gulp of {gulp}")
    logging.info(f"Approx {nsamples // gulp} gulps (+1 if no samples cut off)")
    logging.info(f"The last {nsamp_cut_off} samples will be cut off")

    if gulp % ptsperint:
        raise ValueError(
            f"CODE BUG: gulp ({gulp}) does not divide into ptsperint ({ptsperint}), gulp/ptsperint = {gulp/ptsperint}"
        )
    else:
        intspergulp = gulp // ptsperint
        logging.debug(f"intspergulp: {intspergulp}")

    # initialize things that need to survive multiple gulps
    current_int = 0
    current_gulp = 0
    running_dict = dict(
        running_avg_std=None, chan_running_avg=None, chan_running_std=None
    )
    arr_dtype = get_dtype(header["nbits"])
    arr_outdtype = np.float32  # as subtracting average in clipping/masking

    # precompute DM shifts
    # align it to the highest frequency channel
    if fs[0] > fs[-1]:
        fref = fs[0]
    else:
        fref = fs[-1]
    shifts = round_to_samples(DM_delay(DM, fs, fref), tsamp)

    # check all positive
    assert (
        np.array(shifts) >= 0
    ).all(), "Some shifts are negative, indexing may go wonky"
    # check max(shifts) = maxDT
    if max(shifts) != maxDT:
        raise ValueError(
            f"Maximum shift ({max(shifts)}) does not match maxDT ({maxDT}),"
            f" something went wrong with DM<->maxDT conversion or in the shift calculation"
        )

    # Update and write header
    header["nbits"] = 32
    if header.get("nsamples", ""):
        logging.debug(
            f"Updating header, nsamples ({header['nsamples']}) will be decreased by {maxDT + nsamp_cut_off}",
        )
        header["nsamples"] -= maxDT + nsamp_cut_off
        logging.info(f"Updated header, nsamples = {header['nsamples']}")


    out_filename = args.filename[:-4] + f"_DM{DM:.{dmprec}f}.fil"
    outf = open(os.path.join(args.outdir, out_filename), "wb")

    logging.info(f"Writing header to {out_filename}\n")
    write_header(header, outf)

    t1 = time.perf_counter()
    filfile = open(args.filename, "rb")
    filfile.seek(hdrlen)

    # Read in first gulp
    # subtracting off the channel averages so everything needs to be converted to floats
    intensities = (
        np.fromfile(filfile, count=gulp * nchans, dtype=arr_dtype)
        .reshape(-1, nchans)
        .astype(arr_outdtype)
    )
    logging.debug(f"Read in first chunk {intensities.shape}")
    logging.debug(f"Size of chunk: {sys.getsizeof(intensities)/1000/1000} MB")
    logging.debug(
        f"Approximate size of dedispersion arrays: {approx_size_shifted_arrays(intensities, maxDT)/1000/1000} MB"
    )

    # Process first gulp separately
    intensities[:, list(zerochans)] = 0
    preprocess(
        intspergulp,
        ptsperint,
        intensities,
        running_dict,
        args.clipsig,
        args.droptotsig,
        gulp,
        maxDT,
        current_int,
        mask,
    )
    # logging.debug("First gulp, initializing prev_array")
    prev_array = np.zeros((maxDT, nchans), dtype=intensities.dtype)
    # logging.debug(f"prev_array size {sys.getsizeof(prev_array)/1000/1000}MB")
    prev_array, mid_array, end_array = shift_and_stack(
        intensities, shifts, prev_array, maxDT
    )
    # logging.debug(f"shifted and stacked first gulp")
    # logging.debug(f"array sizes: {sys.getsizeof(prev_array)/1000000}, {sys.getsizeof(mid_array)/1000000}, {sys.getsizeof(end_array)/1000000} MB")
    outf.write(mid_array.ravel().astype(arr_outdtype))
    current_gulp += 1

    # reset for next loop
    prev_array = end_array
    intensities = (
        np.fromfile(filfile, count=gulp * nchans, dtype=arr_dtype)
        .reshape(-1, nchans)
        .astype(arr_outdtype)
    )

    # test if need to do next loop, or if on last gulp
    if intensities.shape[0] > maxDT:
        if intensities.shape[0] < gulp:  #  on last gulp
            if intensities.shape[0] % ptsperint:  # last int is weirdly sized
                intspergulp = (gulp // ptsperint) + 1
                logging.debug(f"intspergulp changed to {intspergulp}")

        while True:
            # tt0 = time.perf_counter()
            intensities[:, list(zerochans)] = 0
            preprocess(
                intspergulp,
                ptsperint,
                intensities,
                running_dict,
                args.clipsig,
                args.droptotsig,
                gulp,
                maxDT,
                current_int,
                mask,
            )
            # tt1 = time.perf_counter()
            # logging.debug(f"Clipped and masked gulp {current_gulp} in {tt1 - tt0} s")

            # Brute-force dedisperse whole gulp
            prev_array, mid_array, end_array = shift_and_stack(
                intensities, shifts, prev_array, maxDT
            )
            outf.write(prev_array.ravel().astype(arr_outdtype))
            outf.write(mid_array.ravel().astype(arr_outdtype))
            # tt2 = time.perf_counter()
            # log.debug(f"Dedispersed in {tt2-tt1} s")
            # log.debug(f"Processed gulp {current_gulp}")
            logging.info(f"Processed gulp {current_gulp}")
            current_gulp += 1

            # reset for next loop
            prev_array = end_array
            intensities = (
                np.fromfile(filfile, count=gulp * nchans, dtype=arr_dtype)
                .reshape(-1, nchans)
                .astype(arr_outdtype)
            )

            # Test if on last interval or end of file
            if intensities.shape[0] < gulp:
                if intensities.size == 0 or intensities.shape[0] < maxDT:
                    break
                elif intensities.shape[0] % ptsperint:
                    intspergulp = (intensities.shape[0] // ptsperint) + 1
                    logging.debug(f"last gulp detected, intspergulp changed to {intspergulp}")

    outf.close()
    filfile.close()

    t2 = time.perf_counter()

    logging.info("\nDone")
    logging.info(f"TIME initialize: {t1-t0} s")
    logging.info(f"TIME clip, mask, dedisperse all gulps: {t2-t1} s")
    logging.info(f"TIME total: {t2-t0} s")

    sys.exit()
