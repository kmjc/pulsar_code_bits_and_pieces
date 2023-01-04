# Adam's fdp.py script
# Edits:
#     argparse
#     use shape broadcasting to cut down on memory useage
#     don't use presto - no forced floats and hopefully speedier as less back and forth
#     is a bit speedier ~0.7x
#     downsample at the same time if desired
#       NB if using --downsamp need to have you filterbank file be an argument before that,
#       or the nargs='*' tries to package it in with downsamp
#     refactored chunk_size to gulp to be consistent with my other scripts

# edited this to compute some other stats too - skew, kurtosis, and the s1 and s2 for SK

# calculate medians and std for thresholding excluding any 0s in the data
# option to calcuate fdp mask on tscrunched data
# . . . there are too many masked arrays here, it's going to hog memory

import numpy as np
from presto_without_presto import sigproc
from sigproc_utils import get_dtype, get_nbits, write_header
import argparse
import copy
from scipy.stats.mstats import skew, kurtosis
import sys


def get_fdp_mask(arr, axis=0, sigma=4.5):
    """
    Get a mask of where arr is < median - sigma*std
    where the median and std have been calulates after masking out any 0s.
    Returns a boolean array with the same shape as arr.

    axis = time axis of array, e.g. for (nspec,nchan) axis=0"""
    tmp_arr = np.ma.array(arr, mask=(arr == 0))
    meds = np.ma.median(tmp_arr, axis=axis)
    stds = np.ma.median(tmp_arr, axis=axis)
    del tmp_arr
    threshold = meds - sigma * stds
    return arr < threshold


def tscrunch(arr, fac):
    """Scrunch (by summing) a (nspec,nchan) array in time by <fac>
    Returns array of shape (nspec/fac, nchan)"""
    return arr.reshape(-1, fac, arr.shape[-1]).sum(axis=1)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="""Correct artefacts due to dropped packets in CHIME data.
    These show up as sudden dips (but not necessarily to 0) in 4 channels at the same time.
    This script
        - reads in a chunk of data,
        - calculates the median and standard deviation for each frequency channel,
        - replaces any values less than (median - <thresh_sig> * std) with the median
        - repeat
    A new file ending in _fdp.fil will be written

    If downsampling factors are supplied additional _fdp_t<downsamp>.fil files will be written
    """,
)

parser.add_argument(
    "fn",
    type=str,
    help="filterbank file to process",
)

parser.add_argument(
    "--gulp",
    type=int,
    default=30674,
    help="Number of samples to operate on at once",
)

parser.add_argument(
    "--thresh_sig",
    type=float,
    default=4.5,
    help="Replace values this number of standard deviations under the median",
)

parser.add_argument(
    "--fdp_tscrunch",
    type=int,
    default=1,
    help="""
    With this argument
    - the data is scrunched (summed) in time by this factor first,
    - the dropout sections are identified based on the median and std of the scrunched data
    - this mask is upsampled back to the original time resolution and applied to the original data

    Why? With 40.96us data fdp has trouble catching all the dips. For 327.68us it's fine'.
    Recommend using this on high-time-resolution data"""
    # you're less likely to detect dips <fdp_tscrunch time bins in this case
    # but a) we weren't detecting them anyway
    # b) Shorter things will just blend into the noise
    # c) this tends to happen on longer timescales so it's fine
)

parser.add_argument(
    "--downsamp",
    type=int,
    nargs="*",
    help="Downsampling factor/s to apply (multiple factors should be separated by spaces)",
)

parser.add_argument(
    "--stats",
    action="store_true",
    help="Also calculate and save some stats: skewness, kurtosis, s1 & s2 for spectral kurtosis, number of unmasked time samples summed, total number of time samples",
)

args = parser.parse_args()

if args.fdp_tscrunch < 1:
    raise AttributeError(f"fdp_tscrunch ({args.fdp_tscrunch}) must be >=1")
if args.gulp % args.fdp_tscrunch:
    print(f"Gulp {args.gulp} is not divisible by fdp_tscrunch {args.fdp_tscrunch}")
    gulp = (int(args.gulp // args.fdp_tscrunch) + 1) * args.fdp_tscrunch
    print(f"Gulp adjusted to {gulp}")
else:
    gulp = args.gulp

header, hdrlen = sigproc.read_header(args.fn)
nspecs = int(sigproc.samples_per_file(args.fn, header, hdrlen))
nchans = header["nchans"]
arr_dtype = get_dtype(header["nbits"])

if header["nifs"] != 1:
    raise AttributeError(f"Code not written to deal with unsummed polarization data")


# loop through chunks
loop_iters = int(nspecs // gulp)
if nspecs % gulp:
    loop_iters += 1
fn_clean = args.fn.strip(".fil")
fdp_fn = f"{fn_clean}_fdp.fil"
new_fil = open(fdp_fn, "wb")
write_header(header, new_fil)
print(f"Output will be written to {fdp_fn}")
fil = open(args.fn, "rb")
fil.seek(hdrlen)

if args.stats:
#    skews = np.zeros((loop_iters, nchans))
#    kurtoses = np.zeros((loop_iters, nchans))
    s1 = np.zeros((loop_iters, nchans))
    s2 = np.zeros((loop_iters, nchans))
    num_unmasked_points = np.zeros((loop_iters, nchans), dtype=int)
    n = np.zeros((loop_iters), dtype=int)

additional_fils = []
if args.downsamp is not None:
    for i, d in enumerate(args.downsamp):
        add_fn = f"{fn_clean}_fdp_t{d}.fil"
        print(f"Also outputting downsampled file: {add_fn}")
        additional_fils.append(open(add_fn, "wb"))
        add_header = copy.deepcopy(header)
        add_header["tsamp"] = header["tsamp"] * d
        if header.get("nsamples", ""):
            add_header["nsamples"] = int(header["nsamples"] // d)
        write_header(add_header, additional_fils[i])

for i in range(loop_iters):
    print(f"{i+1}/{loop_iters}", end="\r", flush=True)
    spec = np.fromfile(fil, count=gulp * nchans, dtype=arr_dtype).reshape(-1, nchans)
    # has shape (nspec, nchans) so it plays nice with brodcasting

    mzeros = spec == 0

    # tscrunch if necessary
    if args.fdp_tscrunch != 1:
        working_spec = tscrunch(spec, args.fdp_tscrunch)
    else:
        working_spec = spec

    # get the (tscrunched) dropout mask
    mfdp = get_fdp_mask(working_spec, axis=0, sigma=args.thresh_sig).data

    # convert mask to full time resolution if necessary, combine with zeros mask
    if args.fdp_tscrunch != 1:
        mtot = np.repeat(mfdp, args.fdp_tscrunch, axis=0) | mzeros
    else:
        mtot = mfdp | mzeros

    del mfdp
    del mzeros

    # get medians to replace masked values with, not including to-be-masked values in the calculation
    tmp = np.ma.array(spec, mask=(mtot)).astype(int)
    meds_fullres = np.ma.median(tmp, axis=0)
    if not args.stats:
        del tmp

    # replace masked values with the median
    spec[mtot] = meds_fullres[np.where(mtot)[1]]

    new_fil.write(spec.ravel().astype(arr_dtype))
    if additional_fils:
        for ii, d in enumerate(args.downsamp):
            additional_fils[ii].write(spec[::d, :].ravel().astype(arr_dtype))

    # calc stats
    if args.stats:
        del spec
 #       skews[i, :] = skew(tmp, axis=0, bias=False)
 #       kurtoses[i, :] = kurtosis(tmp, axis=0, bias=False)
        s1[i, :] = tmp.sum(axis=0)
        s2[i, :] = (tmp**2).sum(axis=0)
        num_unmasked_points[i, :] = (~tmp.mask).sum(axis=0)
        n[i] = tmp.shape[0]


fil.close()
new_fil.close()
# save stats
if args.stats:
    print(f"Writing stats to {fdp_fn[:-4]}_stats.npz")
    np.savez(
        f"{fdp_fn[:-4]}_stats.npz",
#        skew=skews,
#        kurtosis=kurtoses,
        s1=s1,
        s2=s2,
        num_unmasked_points=num_unmasked_points,
        n=n,
    )
if additional_fils:
    for add_fil in additional_fils:
        add_fil.close()

sys.exit(0)
