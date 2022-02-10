import numpy as np
from presto_without_presto import sigproc
import copy
import time

t0 = time.perf_counter()

######################## SET SOME PARAMETERS #########################

verbosity = 3
verbose = verbosity > 2
filename = "test_fil/SURVEYv0_point122_DM103_59599_pow_middle0.1min.fil"
out_filename = None
# If set DM it'll use that, otherwise it uses maxDT
DM = 20
maxDT = 0
dmprec = 3

gulp = 20000
optimize_gulp = True
# this will find the factors of the number of samples in the file,
# chucks anything < maxDT, and picks the one closest to gulp

# where want the reference frequency for dedispersion to be for a channel
where_channel_ref_freq = "center"  # "lower" "upper"


#######################################################################
########################## DEFINE FUNCTIONS ###########################

def verbose_message(verbosity_level, message):
    global verbosity
    if verbosity_level <= verbosity:
        print(message)
    else:
        pass

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

########################## DEFINE FUNCTIONS ###########################
#######################################################################


#######################################################################
################## Get header and derive some stuff from it ###########
header, hdrlen = sigproc.read_header(filename)
nsamples = int(sigproc.samples_per_file(filename, header, hdrlen))

verbose_message(1, header)
verbose_message(1, f"nsamples: {nsamples}")

filfile = open(filename, 'rb')
filfile.seek(hdrlen)


if header["nifs"] != 1:
    raise ValueError(f"Code not written to deal with unsummed polarization data")

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

# align it to to center of the highest frequency channel
shifts = round_to_samples(DM_delay(DM, fs, fs[-1]), tsamp)
# check all positive
assert (np.array(shifts) >= 0).all(), "Some shifts are negative, indexing may go wonky"
# check max(shifts) = maxDT
if max(shifts) != maxDT:
    raise ValueError(
        f"Maximum shift ({max(shifts)}) does not match maxDT ({maxDT}),"
        f" something went wrong with DM<->maxDT conversion or in the shift calculation"
    )


#######################################################################
######################### Update header ###############################

# NOT updating tstart
#    aka tstart is now the reference time for the center of the highest frequency channel since that doesn't get shifted
# BUT are cutting off maxDT samples, so need to adjust nsamples
# iff it's in the header!! a bunch of them don't have it

# always chuck the last maxDT of data since it's partial
if optimize_gulp:  # pick the closest factor of nsamples (which is also >maxDT)
    factors = np.array(factorize(nsamples))
    factors_over_maxDT = factors[factors > maxDT]
    gulp = factors_over_maxDT[abs(factors_over_maxDT - gulp).argmin()]
    verbose_message(1, f"Optimized gulp to {gulp}")
    cut_off_extra = 0
else:
    cut_off_extra = nsamples % gulp
#    leftover = nsamples % gulp
#    if leftover > maxDT:
#        cut_off_extra = 0
#    else:
#        cut_off_extra = leftover
# Hmmmm I think this might cause me some issues
# Need to think of a way to do this that doesn't require evaluating an if in every sweep

if gulp < maxDT:
    raise AttributeError(f"gulp {gulp} less than maxDT {maxDT}. Increase gulp")

if header.get("nsamples", ""):
    verbose_message(2, f"Updating header, nsamples ({header['nsamples']}) will be decreased by {maxDT + cut_off_extra}")
    header["nsamples"] -= (maxDT + cut_off_extra)
    verbose_message(1, f"Updated header, nsamples = {header['nsamples']}")


# Open out file
if zero_or_none(out_filename):
    out_filename = filename[:-4] + f"_DM{DM:.{dmprec}f}.fil"
outf = open(out_filename, 'wb')


# Write header
verbose_message(1, f"Writing header to {out_filename}")
outf.write(sigproc.addto_hdr("HEADER_START", None))
for paramname in list(header.keys()):
    if paramname not in sigproc.header_params:
        # Only add recognized parameters
        continue
    verbose_message(3, "Writing header param (%s)" % paramname)
    value = header[paramname]
    outf.write(sigproc.addto_hdr(paramname, value))
outf.write(sigproc.addto_hdr("HEADER_END", None))


#######################################################################
########################### Dedisperse ################################

# Initialize arrays
arr_dtype = get_dtype(header['nbits'])
prev_array = np.zeros((maxDT, nchans), dtype=arr_dtype)
mid_array = np.zeros((gulp-maxDT, nchans), dtype=arr_dtype)
end_array = np.zeros_like(prev_array)

t1 = time.perf_counter()
print(f"TIME to intialize: {t1-t0} s")


# read in FIRST chunk
# separate as you need to not write the prev_array, and I didn't want an extra if clause in my loop
intensities = np.fromfile(filfile, count=gulp*nchans, dtype=arr_dtype).reshape(-1, nchans)
for i in range(1, nchans - 1 ):
    dt = shifts[i]
    mid_array[:, i] = intensities[dt:gulp-(maxDT - dt), i]
    end_array[:(maxDT - dt), i] = intensities[gulp - (maxDT - dt):, i]

t2 = time.perf_counter()
print(f"TIME to dedisperse first chunk: {t2-t1} s")

# write mid_array ONLY
outf.write(mid_array.ravel().astype(arr_dtype))
t3 = time.perf_counter()
print(f"TIME to write first chunk: {t3-t2} s")

# set up next chunk
prev_array[:,:] = end_array[:,:]
end_array[:,:] = 0
intensities = np.fromfile(filfile, count=gulp*nchans, dtype=arr_dtype).reshape(-1, nchans)


while True:
    for i in range(1, nchans - 1 ):
        dt = shifts[i]
        prev_array[maxDT - dt:, i] += intensities[:dt, i]
        mid_array[:, i] = intensities[dt:gulp-(maxDT - dt), i]
        end_array[:(maxDT - dt), i] = intensities[gulp - (maxDT - dt):, i]

    # write prev_array
    outf.write(prev_array.ravel().astype(arr_dtype))
    # write mid_array
    outf.write(mid_array.ravel().astype(arr_dtype))

    # set up next chunk
    prev_array[:,:] = end_array[:,:]
    end_array[:,:] = 0
    intensities = np.fromfile(filfile, count=gulp*nchans, dtype=arr_dtype).reshape(-1, nchans)

    if intensities.shape[0] < gulp:  # fromfile doesn't detect EOF, have to do it manually
        break
t4 = time.perf_counter()
print(f"TIME for other chunks: {t4-t3} s")
print(f"TIME total: {t4-t0} s")

outf.close()
filfile.close()
