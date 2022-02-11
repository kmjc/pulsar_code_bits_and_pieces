import numpy as np
from presto_without_presto import sigproc
import copy
import time
import argparse
import sys

t0 = time.perf_counter()

#######################################################################
########################## DEFINE FUNCTIONS ###########################


def sizeof_fmt(num, use_kibibyte=False):
    """Make number of bytes human-readable, default is 1kB = 1000B
    To use 1024, set use_kibibyte=True"""
    base, suffix = [(1000.,'B'),(1024.,'iB')][use_kibibyte]
    for x in ['B'] + list(map(lambda x: x+suffix, list('kMGTP'))):
        if -base < num < base:
            return "%3.1f %s" % (num, x)
        num /= base
    return "%3.1f %s" % (num, x)


def factorize(num):
    return [n for n in range(1, num + 1) if num % n == 0]

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

def DM_delay(DM, freq, *ref_freq):
    """
    Pass in DM (cm-3pc), freq (MHz), and optionally a reference frequency
    Returns time delay in seconds
    Uses Manchester-Taylor 1/2.4E-4 convention
    """
    K = 1.0 / 2.41E-4  # Manchester-Taylor convention, units in MHz cm-3pc s
    if ref_freq:
        return K * DM * (1/freq**2 - 1/ref_freq[0]**2)
    else:
        return K * DM / freq**2

def inverse_DM_delay(delay_s, freq, *ref_freq):
    """Inverse of DM_delay
    Pass in delay (s), freq (MHz), and optionally a reference frequency
    (if not included it's referenced to infinite frequency)
    Returns the DM which would produce that delay in cm-3pc
    Uses Manchester-Taylor 1/2.4E-4 convention"""
    K = 1.0 / 2.41E-4  # Manchester-Taylor convention, units in MHz cm-3pc s
    if ref_freq:
        return delay_s / (1/freq**2 - 1/ref_freq[0]**2) / K
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
        return np.linspace(fmin + df/2, fmax - df/2, nchans, endpoint=True, **kwargs)
    if type == "upper":
        return np.linspace(fmin + df, fmax, nchans, endpoint=True, **kwargs)
    else:
        raise AttributeError(f"type {type} not recognised, must be one of 'center', 'lower', 'upper'")

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
        raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return fvalue

########################## DEFINE FUNCTIONS ###########################
#######################################################################

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="""Give some info to choose a good gulp size for chunk_dedisperse"""
)
parser.add_argument('filename', type=str,
                    help='Filterbank file to dedisperse')

g = parser.add_mutually_exclusive_group(required=True)
g.add_argument('-d', '--dm', type=float, default=0,
               help="DM (cm-3pc) to dedisperse to")
g.add_argument('-t', '--maxdt', type=check_positive_float, default=0,
               help="Number of time samples corresponding to the DM delay between the lowest and highest channel\n(must be positive)")

parser.add_argument('--where_channel_ref', default='center', choices=['center', 'lower', 'upper'],
                     help='Where within the channel ')

parser.add_argument('--max', default=1E9, help="max number of bytes to allow")

parser.add_argument('-v', '--verbosity', action='count', default=0)

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
fmax = header["fch1"] - header["foff"]/2
fmin = fmax + nchans * header["foff"]
verbose_message(2, f"fmin: {fmin}, fmax: {fmax}, nchans: {nchans} tsamp: {tsamp}")


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


verbose_message(1, f"Max DM is {DM}")
verbose_message(1, f"Maximum DM delay need to shift by is {maxdelay_s} s")
verbose_message(1, f"This corresponds to {maxDT} time samples")
print(f"Minimum gulp size is {maxDT}")  # I think mid_array being 0 size won't break anything? Don't know how the write handles that


factors = np.array(factorize(nsamples))
factors_over_maxDT = factors[factors > maxDT]
verbose_message(1, f"\ngulps > maxDT ({maxDT}) which are also factors of nsamples ({nsamples}):\n", factors_over_maxDT)


verbose_message(1, "Memory size below is an approximation and based on the size of the main arrays for the calculation")
verbose_message(1, "(Based on very minimal testing on a small observation!) it can be 2x this")

print("")
print(f"{'gulp':<20} {'nchunks':<10} {'approx size':<14} {'x1.25':<10} {'x1.5':<10} {'x2':<10}")
overhead = 464  # no idea what determines this or what makes it vary
nbytes = header['nbits'] // 8
for gulp in factors_over_maxDT:
    nchunks = int(nsamples / gulp)
    byte_size_data = (2*gulp*nchans + maxDT*nchans)*nbytes + overhead

    if byte_size_data > args.max:
        break
    # from (very minimal!!) testing with mprun, it runs at ~ double this
    fudge_factor = 2
    print(f"{gulp:<20} {nchunks:<10} {sizeof_fmt(byte_size_data):<14} "
          f"{sizeof_fmt(1.25*byte_size_data):<10} "
          f"{sizeof_fmt(1.5*byte_size_data):<10} "
          f"{sizeof_fmt(2*byte_size_data):<10}")
