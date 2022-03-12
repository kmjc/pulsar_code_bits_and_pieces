import numpy as np
from presto_without_presto import sigproc, rfifind
from scipy.ndimage import generic_filter
import copy
import time
import argparse
import sys


#######################################################################
########################## DEFINE FUNCTIONS ###########################

# GENERAL FUNCTIONS:


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


# FILTERBANK FUNCTIONS:

# presto's filterbank has a get_dtype could use too, but haven't without_presto-ed that yet
def get_dtype(nbits):
    """
    Returns:
        dtype of the data
    """
    if nbits == 8:
        return np.uint8
    elif nbits == 16:
        return np.uint16
    elif nbits == 32:
        return np.float32
    else:
        raise RuntimeError(f"nbits={nbits} not supported")

def get_nbits(dtype):
    """
    Returns:
        number of bits of the data
    """
    if dtype == np.uint8:
        return 8
    elif dtype == np.unit16:
        return 16
    elif dtype == np.float32:
        return 32
    else:
        raise RuntimeError(f"dtype={dtype} not supported")

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


def get_fs(fmin, fmax, nchans, type="center", **kwargs):
    """Get channel frequencies for a band with edges fmin, fmax
    type can be "center", "lower", "upper"

    Done via np.linspace, can pass in retstep=True.

    Returns a numpy array startin with the lowest channel
    """
    df = (fmax - fmin) / nchans
    if type == "lower":
        return np.linspace(fmin, fmax, nchans, endpoint=False, **kwargs)
    if type == "center":
        return np.linspace(
            fmin + df / 2, fmax - df / 2, nchans, endpoint=True, **kwargs
        )
    if type == "upper":
        return np.linspace(fmin + df, fmax, nchans, endpoint=True, **kwargs)
    else:
        raise AttributeError(
            f"type {type} not recognised, must be one of 'center', 'lower', 'upper'"
        )


def round_to_samples(delta_t, sampling_time):
    """Round a time delay to the nearest number of time samples"""
    return np.round(delta_t / sampling_time).astype(int)


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


# don't want to have a whole mask with all the stats in memory
# extract relevant bits, make the running median
# have some functions to return those things for particular time samples/chunks
class ThinnedMask:
    def __init__(self, rfimask, medlen=17):
        """rfimask is a presto rfifind.rfifind object"""
        self.medlen = medlen
        self.nchan = rfimask.nchan
        self.nint = rfimask.nint
        self.ptsperint = rfimask.ptsperint
        # original rfimask has this as an array, convert to set if necessary
        self.mask_zap_ints = set(rfimask.mask_zap_ints)
        self.mask_zap_chans = rfimask.mask_zap_chans
        # original rfimask has this as a list of arrays, convert to set if necessary
        self.mask_zap_chans_per_int = [set(x) for x in rfimask.mask_zap_chans_per_int]

        self.mask = array_from_mask_params(
            self.nint,
            self.nchan,
            self.mask_zap_ints,
            self.mask_zap_chans,
            self.mask_zap_chans_per_int,
        )

        masked_avgs = np.ma.array(rfimask.avg_stats, mask=self.mask)

        # found generic_filter, filling with nans and using np.nanmedian
        # calculates the median in a window centered on the index, and deals with nans properly
        self.running_medavg = generic_filter(
            masked_avgs.filled(fill_value=np.nan),
            np.nanmedian,
            size=(medlen, 1),
            mode="nearest",
        )

    def time_sample_to_interval(self, time_sample):
        """for a given time sample,return the index of the corresponding interval"""
        starts = np.arange(self.nint) * self.ptsperint
        interval_index = np.where(starts <= time_sample)[0][-1]
        return interval_index

    def time_sample_mask(self, time_sample):
        "get the mask for a single time sample (result has size nchan)"
        interval_id = self.time_sample_to_interval(time_sample)
        return self.mask[interval_id, :]

    def time_sample_medavg(self, time_sample):
        "get the running median of the avg_stats for a single time sample (result has size nchan)"
        interval_id = self.time_sample_to_interval(time_sample)
        return self.running_medavg[interval_id, :]

    def chunk_mask(self, start_tsamp, end_tsamp):
        """get the mask for a chunk of time samples start_tsamp:end_tsamp
        result has shape (end_tsamp - start_tsamp, nchan)"""
        time_samples = np.arange(start_tsamp, end_tsamp)
        interval_ids = np.array(list(map(self.time_sample_to_interval, time_samples)))
        return self.mask[interval_ids, :]

    def chunk_medavg(self, start_tsamp, end_tsamp):
        """get the running median of the avg_stats for a chunk of time samples start_tsamp:end_tsamp
        result has shape (end_tsamp - start_tsamp, nchan)"""
        time_samples = np.arange(start_tsamp, end_tsamp)
        interval_ids = np.array(list(map(self.time_sample_to_interval, time_samples)))
        return self.running_medavg[interval_ids, :]

    def mask_chunk(self, start_tsamp, end_tsamp, data, ignorechans=set()):
        """
        data must be of shape (end_tsamp - start_tsamp, nchan)
        ignorechans should be a set of channel indices

        * subtract off the (per-channel) running median of the avg_stats from the data
        * fill any masked values with 0,
        * set any channels given in igorechans to 0

        returns a new array of masked (and median avg subtracted) data
        """
        if end_tsamp - start_tsamp != data.shape[0]:
            raise ValueError(
                f"Time sample range ({end_tsamp} - {start_tsamp} = {end_tsamp - start_tsamp}) does not match data shape {data.shape}"
            )

        msk = self.chunk_mask(start_tsamp, end_tsamp)
        medavg = self.chunk_medavg(start_tsamp, end_tsamp)
        out = data - medavg
        out[msk] = 0
        out[:, list(ignorechans)] = 0
        return out


class EmptyThinnedMask:
    def __init__(self):
        pass

    def time_sample_to_interval(self, time_sample):
        return None

    def time_sample_mask(self, time_sample):
        return None

    def time_sample_medavg(self, time_sample):
        return None

    def chunk_mask(self, start_tsamp, end_tsamp):
        return None

    def chunk_medavg(self, start_tsamp, end_tsamp):
        return None

    def mask_chunk(self, start_tsamp, end_tsamp, data, ignorechans=set()):
        """
        * EmptyThinnedMask method*
        returns a new array with any ignorechans set to 0
        """
        out = copy.copy(data)
        out[:, list(ignorechans)] = 0
        return out


########################## DEFINE FUNCTIONS ###########################
#######################################################################


if __name__ == '__main__':
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
        "--out_filename",
        type=str,
        default=None,
        help="Filename to write the output to (otherwise will append _DM<DM>.fil)",
    )
    parser.add_argument(
        "gulp",
        type=int,
        help="""Number of spectra (aka number of time samples) to read in at once
                        NOTE: this is also the length over which the median is calculated for masking""",
    )
    parser.add_argument(
        "-g",
        "--dont_optimize_gulp",
        action="store_true",
        help="""Don't optimize gulp. (Generally a good idea to optimize but option exists in case of memory constraints)
    Optimization
      - finds the factors of the total number of samples in the file, N
      - disregards any less than the maximum time delay (maxdt)
      - selects the value closest to <gulp>

    Note, without optimization, if <gulp> is not a factor of N, you'll be discarding some data and the end""",
    )

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "-d", "--dm", type=float, default=0, help="DM (cm-3pc) to dedisperse to"
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
        default=3,
        help="DM precision (only used when writing filename if <out_filename> not given)",
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="""-v = some information
    -vv = more information
    -vvv = the most information""",
    )

    # masking options, this will definitely break for an inverted band
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        default="",
        help="""rfifind .mask file to apply.
                        * a per-channel running median of avg_stats is calcualted, ignoring masked values
                        * this running median is sutracted from the data
                        * data to be masked are filled with 0s""",
    )
    # parser.add_argument('--noclip', action='store_true',
    #                    help="Don't apply clipping (a la presto) time samples to clip are based on the running median and standard deviation of the DM0 time series")
    parser.add_argument(
        "--ignorechan",
        type=str,
        help="Comma separated string (no spaces!) of channels to ignore. PRESTO convention (lowest frequency channel = 0)",
        default="",
    )

    args = parser.parse_args()

    # being too lazy to refactor
    verbosity = args.verbosity
    verbose = verbosity > 2  # for the stdin=verbose things
    filename = args.filename
    out_filename = args.out_filename
    DM = args.dm
    maxDT = args.maxdt
    dmprec = args.dmprec
    gulp = args.gulp
    optimize_gulp = not args.dont_optimize_gulp
    where_channel_ref_freq = args.where_channel_ref

    def verbose_message(verbosity_level, message):
        global verbosity
        if verbosity_level <= verbosity:
            print(message)
        else:
            pass

    t0 = time.perf_counter()
    #######################################################################
    ################## Get header and derive some stuff from it ###########
    verbose_message(0, f"Working on file: {filename}")
    header, hdrlen = sigproc.read_header(filename)
    nsamples = int(sigproc.samples_per_file(filename, header, hdrlen))

    verbose_message(2, header)

    if header["nifs"] != 1:
        raise ValueError(f"Code not written to deal with unsummed polarization data")

    # calculate/get parameters from header
    tsamp = header["tsamp"]
    nchans = header["nchans"]

    # calculate fmin and fmax FROM HEADER
    fmax = header["fch1"] - header["foff"] / 2
    fmin = fmax + nchans * header["foff"]
    verbose_message(
        0,
        f"fmin: {fmin}, fmax: {fmax}, nchans: {nchans} tsamp: {tsamp} nsamples: {nsamples}",
    )

    #######################################################################
    ########################## Set up for masking #########################
    if not_zero_or_none(args.mask):
        t001 = time.perf_counter()
        verbose_message(0, f"Loading mask from {args.mask}")
        rfimask = rfifind.rfifind(args.mask)
        t002 = time.perf_counter()
        verbose_message(1, f"TIME to load mask: {t002-t001} s")
        mask = ThinnedMask(rfimask)
        t003 = time.perf_counter()
        verbose_message(
            0,
            f"Calculated the running median of avg_stats over {mask.medlen} intervals. All set",
        )
        verbose_message(
            1,
            f"TIME to get running median etc: {t003-t002} s for {rfimask.nint} intervals",
        )
        del rfimask
    else:
        mask = EmptyThinnedMask()

    # get indices of channels not included in ignorechan
    if not_zero_or_none(args.ignorechan):
        # convert string to list of ints, and invert so index 0 = highest freq channel as per sigproc filterbank convention
        ignorechans = set(
            [nchans - 1 - int(chan) for chan in args.ignorechan.split(",")]
        )
    else:
        ignorechans = set()

    #######################################################################
    ###### Get the maximum DM delay, and the delays for each channel ######

    fs = get_fs(fmin, fmax, nchans, type=where_channel_ref_freq)

    # If given use DM, else use maxDT
    if not_zero_or_none(DM):
        maxdelay_s = DM_delay(DM, fs[0], fs[-1])
        maxDT = round_to_samples(maxdelay_s, tsamp)
    elif not_zero_or_none(maxDT):
        # maxDT unchanged
        maxdelay_s = maxDT * tsamp
        DM = inverse_DM_delay(maxdelay_s, fs[0], fs[-1])
    else:
        raise AttributeError(f"Must set either DM ({DM}) or maxDT{maxDT}")

    verbose_message(0, f"Max DM is {DM}")
    verbose_message(1, f"Maximum DM delay need to shift by is {maxdelay_s} s")
    verbose_message(0, f"This corresponds to {maxDT} time samples")

    # align it to to center of the highest frequency channel
    shifts = round_to_samples(DM_delay(DM, fs, fs[-1]), tsamp)
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

    #######################################################################
    ######################### Optimize gulp  ##############################

    # NOT updating tstart
    #    aka tstart is now the reference time for the center of the highest frequency channel since that doesn't get shifted
    # BUT are cutting off maxDT samples, so need to adjust nsamples
    # iff it's in the header!! a bunch of them don't have it

    # always chuck the last maxDT of data since it's partial
    if optimize_gulp:  # pick the closest factor of nsamples (which is also >maxDT)
        factors = np.array(factorize(nsamples))
        factors_over_maxDT = factors[
            factors >= maxDT
        ]  # I think the "=" case should work fine
        if factors_over_maxDT.size == 0:
            raise ValueError(f"No factors ({factors}) found over maxDT ({maxDT})")
        gulp = factors_over_maxDT[abs(factors_over_maxDT - gulp).argmin()]
        verbose_message(0, f"Optimized gulp to {gulp}")
        cut_off_extra = 0
    else:
        verbose_message(0, f"Using input gulp of {gulp}")
        cut_off_extra = nsamples % gulp
        if cut_off_extra:
            verbose_message(
                1,
                f"{cut_off_extra} extra samples will be cut off at the end of the file",
            )
    #    leftover = nsamples % gulp
    #    if leftover > maxDT:
    #        cut_off_extra = 0
    #    else:
    #        cut_off_extra = leftover
    # Hmmmm I think this might cause me some issues
    # Need to think of a way to do this that doesn't require evaluating an if in every sweep

    if gulp < maxDT:
        raise AttributeError(f"gulp {gulp} less than maxDT {maxDT}. Increase gulp")
    nchunks = nsamples // gulp
    verbose_message(1, f"{nchunks} chunks to process")

    if header.get("nsamples", ""):
        verbose_message(
            2,
            f"Updating header, nsamples ({header['nsamples']}) will be decreased by {maxDT + cut_off_extra}",
        )
        header["nsamples"] -= maxDT + cut_off_extra
        verbose_message(1, f"Updated header, nsamples = {header['nsamples']}")



    #######################################################################
    ##################### More Dedispersion setup #########################

    # Initialize arrays and deal with types
    arr_dtype = get_dtype(header["nbits"])
    verbose_message(3, f"{header['nbits']} in header -> dtype {arr_dtype}")
    # If masking then intensities end up as float32 though!
    if not_zero_or_none(args.mask):
        out_arr_dtype = mask.running_medavg.dtype
        verbose_message(3, f"Masking, so all arrays will endup with dtype {out_arr_dtype}")
        prev_array = np.zeros((maxDT, nchans), dtype=out_arr_dtype)
        mid_array = np.zeros((gulp - maxDT, nchans), dtype=out_arr_dtype)
        end_array = np.zeros_like(prev_array)
        num_bits = get_nbits(out_arr_dtype)
        header["nbits"] = num_bits
        verbose_message(0, f"Due to masking, writing output data with nbits={header['nbits']}")
    else:
        prev_array = np.zeros((maxDT, nchans), dtype=arr_dtype)
        mid_array = np.zeros((gulp - maxDT, nchans), dtype=arr_dtype)
        end_array = np.zeros_like(prev_array)
        out_arr_dtype = arr_dtype


    #######################################################################
    ########################### Write Header ##############################
    # has to be here because need to change the nbits in the header if masking
    # Open file
    if zero_or_none(out_filename):
        out_filename = filename[:-4] + f"_DM{DM:.{dmprec}f}.fil"
    outf = open(out_filename, "wb")

    # Write header
    verbose_message(0, f"Writing header to {out_filename}")
    # outf.write(sigproc.addto_hdr("HEADER_START", None))
    header_list = list(header.keys())
    manual_head_start_end = False
    if header_list[0] != "HEADER_START" or header_list[-1] != "HEADER_END":
        verbose_message(
            3,
            f"HEADER_START not first and/or HEADER_END not last in header_list"
            f"removing them from header_list (if present) and writing them manually",
        )
        try_remove("HEADER_START", header_list)
        try_remove("HEADER_END", header_list)
        manual_head_start_end = True

    if manual_head_start_end:
        outf.write(sigproc.addto_hdr("HEADER_START", None))
    for paramname in header_list:
        if paramname not in sigproc.header_params:
            # Only add recognized parameters
            continue
        verbose_message(3, "Writing header param (%s)" % paramname)
        value = header[paramname]
        outf.write(sigproc.addto_hdr(paramname, value))
    if manual_head_start_end:
        outf.write(sigproc.addto_hdr("HEADER_END", None))


    #######################################################################
    ############################ Dedisperse ###############################
    t1 = time.perf_counter()
    verbose_message(1, f"TIME to intialize: {t1-t0} s")

    verbose_message(0, f"Starting dedispersion")
    filfile = open(filename, "rb")
    filfile.seek(hdrlen)
    # read in FIRST chunk
    # separate as you need to not write the prev_array, and I didn't want an extra if clause in my loop
    k = 0  # tracks which gulp I'm in
    intensities = np.fromfile(filfile, count=gulp * nchans, dtype=arr_dtype).reshape(
        -1, nchans
    )
    intensities = mask.mask_chunk(
        k * gulp, (k + 1) * gulp, intensities, ignorechans=ignorechans
    )

    for i in range(1, nchans - 1):
        dt = shifts[i]
        mid_array[:, i] = intensities[dt : gulp - (maxDT - dt), i]
        end_array[: (maxDT - dt), i] = intensities[gulp - (maxDT - dt) :, i]

    t2 = time.perf_counter()
    verbose_message(1, f"TIME to dedisperse first chunk: {t2-t1} s")

    # write mid_array ONLY
    outf.write(mid_array.ravel().astype(out_arr_dtype))
    t3 = time.perf_counter()
    verbose_message(0, f"Processed chunk 1 of {nchunks}")
    verbose_message(1, f"TIME to write first chunk: {t3-t2} s")

    # set up next chunk
    prev_array[:, :] = end_array[:, :]
    end_array[:, :] = 0
    intensities = np.fromfile(filfile, count=gulp * nchans, dtype=arr_dtype).reshape(
        -1, nchans
    )

    k += 1
    if gulp != nsamples:
        while True:
            intensities = mask.mask_chunk(
                k * gulp, (k + 1) * gulp, intensities, ignorechans=ignorechans
            )

            for i in range(1, nchans - 1):
                dt = shifts[i]
                prev_array[maxDT - dt :, i] += intensities[:dt, i]
                mid_array[:, i] = intensities[dt : gulp - (maxDT - dt), i]
                end_array[: (maxDT - dt), i] = intensities[gulp - (maxDT - dt) :, i]

            # write prev_array
            outf.write(prev_array.ravel().astype(out_arr_dtype))
            # write mid_array
            outf.write(mid_array.ravel().astype(out_arr_dtype))
            verbose_message(0, f"Processed chunk {k} of {nchunks}")

            # set up next chunk
            prev_array[:, :] = end_array[:, :]
            end_array[:, :] = 0
            intensities = np.fromfile(
                filfile, count=gulp * nchans, dtype=arr_dtype
            ).reshape(-1, nchans)

            if (
                intensities.shape[0] < gulp
            ):  # fromfile doesn't detect EOF, have to do it manually
                break

            k += 1

    t4 = time.perf_counter()
    verbose_message(1, f"TIME for other chunks: {t4-t3} s")
    verbose_message(0, "Done")
    verbose_message(0, f"TIME total: {t4-t0} s")

    outf.close()
    filfile.close()
    sys.exit()
