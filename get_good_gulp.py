import numpy as np
from presto_without_presto import sigproc
import copy
import time
import argparse
import sys


t0 = time.perf_counter()

#######################################################################
########################## DEFINE FUNCTIONS ###########################


def sizeof_fmt(num, use_kibibyte=False, dp=1):
    """Make number of bytes human-readable, default is 1kB = 1000B
    To use 1024, set use_kibibyte=True"""
    base, suffix = [(1000.0, "B"), (1024.0, "iB")][use_kibibyte]
    for x in ["B"] + list(map(lambda x: x + suffix, list("kMGTP"))):
        if -base < num < base:
            return f"{num:3.{dp}f} {x}"
        num /= base
    return f"{num:3.{dp}f} {x}"


def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]


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


########################## DEFINE FUNCTIONS ###########################
#######################################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="""Give some info to choose a good gulp size for chunk_dedisperse""",
)
parser.add_argument("filename", type=str, help="Filterbank file to dedisperse")

g = parser.add_mutually_exclusive_group(required=True)
g.add_argument("-d", "--dm", type=float, default=0, help="DM (cm-3pc) to dedisperse to")
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
    "-m", "--max", default=1e9, type=float, help="max number of bytes to allow"
)
parser.add_argument(
    "-f",
    "--fudge",
    default=1,
    type=float,
    help=f"fudge factor for max bytes to allow\n"
    f"aka if pass in 512E6 for max -> 512MB\n"
    f"and pass in a fudge of 2\n"
    f"it'll stop when 2 * estimated byte size > 512MB",
)
parser.add_argument(
    "-l",
    "--last_only",
    action="store_true",
    help="only print last line (useful with <max>)",
)

parser.add_argument(
    "--fdmt", action="store_true", help="Get good gulp size for fdmt instead"
)

parser.add_argument(
    "--thin",
    action="store_true",
    help="Thin the output gulps to only the max and min, for that number of chunks, under the max size limit",
)

parser.add_argument("-v", "--verbosity", action="count", default=0)

parser.add_argument(
    "--tophalf",
    action="store_true",
    help="only use with --fdmt. Run on only top half of the band",
)

args = parser.parse_args()

# being too lazy to refactor
verbosity = args.verbosity
verbose = verbosity > 2  # for the stdin=verbose things
filename = args.filename
DM = args.dm
maxDT = args.maxdt
where_channel_ref_freq = args.where_channel_ref


def verbose_message(verbosity_level, message):
    global verbosity
    if verbosity_level <= verbosity:
        print(message)
    else:
        pass


#######################################################################
################## Get header and derive some stuff from it ###########
header, hdrlen = sigproc.read_header(filename)
nsamples = int(sigproc.samples_per_file(filename, header, hdrlen))

verbose_message(1, header)
verbose_message(1, f"nsamples: {nsamples}")

# calculate/get parameters from header
tsamp = header["tsamp"]
nchans = header["nchans"]


# calculate fmin and fmax FROM HEADER
fmax = header["fch1"] - header["foff"] / 2
fmin = fmax + nchans * header["foff"]
if args.fdmt and args.tophalf:
    fmin = fmin + (fmax - fmin) / 2
    nchans = nchans // 2
verbose_message(2, f"fmin: {fmin}, fmax: {fmax}, nchans: {nchans} tsamp: {tsamp}")


#######################################################################
###### Get the maximum DM delay, and the delays for each channel ######


if args.fdmt:
    from fdmt.cpu_fdmt import FDMT

    fs = np.linspace(fmin, fmax, nchans, endpoint=True)
else:
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


verbose_message(1, f"Max DM is {DM}")
verbose_message(1, f"Maximum DM delay need to shift by is {maxdelay_s} s")
verbose_message(1, f"This corresponds to {maxDT} time samples")
print(
    f"Minimum gulp size is {maxDT}"
)  # I think mid_array being 0 size won't break anything? Don't know how the write handles that

if args.fdmt:
    fd1 = FDMT(fmin=fmin, fmax=fmax, nchan=nchans, maxDT=maxDT)

# OLD: based on factors of nsamples
# factors = np.array(factorize(nsamples))
# factors_over_maxDT = factors[factors > maxDT]
# possible_gulps = factors_over_maxDT
# verbose_message(1, f"\ngulps > maxDT ({maxDT}) which are also factors of nsamples ({nsamples}): \n{factors_over_maxDT}")

# NEW: what actually want is for all gulps to be > maxDT
print(f"Number of samples: {nsamples}")
gulps_over_maxDT = np.arange(
    maxDT + 1, nsamples
)  # haven't tested whether it throws a hissy fit if gulp=maxDT, so being safe
leftovers = nsamples % gulps_over_maxDT
no_leftovers = gulps_over_maxDT[leftovers == 0]
no_data_lost = gulps_over_maxDT[leftovers > maxDT]
possible_gulps = list(no_data_lost)
possible_gulps.extend(list(no_leftovers))
possible_gulps.sort()
left = [nsamples % g for g in possible_gulps]


verbose_message(
    1,
    "Memory size below is an approximation and based on the size of the main arrays for the calculation",
)
# verbose_message(1, "(Based on very minimal testing on a small observation!) it can be 2x this")

print("")
print(
    f"{'gulp':<20} {'nchunks':<10} {'approx size':<14} {'x1.25':<10} {'x1.5':<10} {'x2':<10}"
)
nbytes = header["nbits"] // 8
if nbytes < 4:
    verbose_message(
        0,
        f"WARNING {nbytes} bytes used for size estimation BUT if using a mask everyhing gets converted to floats - times by {int(4/nbytes)}",
    )
# prev = []

dt = [("gulp", int), ("nchunks", int), ("leftover", int), ("byte_size_data", float)]
data = []
for i in range(len(possible_gulps)):
    gulp = possible_gulps[i]
    leftover = left[i]
    nchunks = int(nsamples / gulp)

    if args.fdmt:
        byte_size_intensities = gulp * nchans * nbytes
        byte_size_prev_arr = maxDT * maxDT * nbytes

        numRowsA = (fd1.subDT(fd1.fs)).sum()
        numRowsB = (fd1.subDT(fd1.fs[::2], fd1.fs[2] - fd1.fs[0])).sum()
        byte_size_A = (gulp + 2 * maxDT) * numRowsA * nbytes
        byte_size_B = (gulp + 2 * maxDT) * numRowsB * nbytes

        byte_size_data = (
            byte_size_A + byte_size_B + byte_size_intensities + byte_size_prev_arr
        )

    else:
        byte_size_data = (2 * gulp * nchans + maxDT * nchans) * nbytes

    data.append((gulp, nchunks, leftover, byte_size_data))
gulps_and_info = np.array(data, dtype=dt)

within_size_lim = gulps_and_info[
    gulps_and_info["byte_size_data"] * args.fudge <= args.max
]

prec = 2

if args.last_only:
    gulp, nchunks, leftover, byte_size_data = within_size_lim[-1]
    print(
        f"{gulp:<20} {nchunks:<10} {sizeof_fmt(byte_size_data, dp=prec):<14} "
        f"{sizeof_fmt(1.25*byte_size_data, dp=prec):<10} "
        f"{sizeof_fmt(1.5*byte_size_data, dp=prec):<10} "
        f"{sizeof_fmt(2*byte_size_data, dp=prec):<10}"
    )
elif args.thin:
    for nchunk in np.unique(within_size_lim["nchunks"])[::-1]:
        same_nchunk = within_size_lim[within_size_lim["nchunks"] == nchunk]
        mingulp_idx = np.where(same_nchunk["gulp"] == same_nchunk["gulp"].min())[0]
        maxgulp_idx = np.where(same_nchunk["gulp"] == same_nchunk["gulp"].max())[0]
        for id in [mingulp_idx, maxgulp_idx]:
            gulp, nchunks, leftover, byte_size_data = same_nchunk[id][0]
            print(
                f"{gulp:<20} {nchunks:<10} {sizeof_fmt(byte_size_data, dp=prec):<14} "
                f"{sizeof_fmt(1.25*byte_size_data, dp=prec):<10} "
                f"{sizeof_fmt(1.5*byte_size_data, dp=prec):<10} "
                f"{sizeof_fmt(2*byte_size_data, dp=prec):<10}"
            )
else:
    for (gulp, nchunks, leftover, byte_size_data) in within_size_lim:
        print(
            f"{gulp:<20} {nchunks:<10} {sizeof_fmt(byte_size_data, dp=prec):<14} "
            f"{sizeof_fmt(1.25*byte_size_data, dp=prec):<10} "
            f"{sizeof_fmt(1.5*byte_size_data, dp=prec):<10} "
            f"{sizeof_fmt(2*byte_size_data, dp=prec):<10}"
        )
