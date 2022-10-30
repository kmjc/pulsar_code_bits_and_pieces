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

import numpy as np
from presto_without_presto import sigproc
from sigproc_utils import get_dtype, get_nbits, write_header
import argparse
import copy
from scipy.stats.mstats import skew, kurtosis


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
    "--downsamp",
    type=int,
    nargs="*",
    help="Downsampling factor/s to apply (multiple factors should be separated by spaces)",
)

parser.add_argument(
    "--stats",
    action='store_true',
    help="Also calculate and save some stats: skewness, kurtosis, s1 & s2 for spectral kurtosis, number of unmasked time samples summed, total number of time samples"
)

args = parser.parse_args()

header, hdrlen = sigproc.read_header(args.fn)
nspecs = int(sigproc.samples_per_file(args.fn, header, hdrlen))
nchans = header["nchans"]
arr_dtype = get_dtype(header["nbits"])

if header["nifs"] != 1:
    raise AttributeError(f"Code not written to deal with unsummed polarization data")


#loop through chunks
loop_iters = int(nspecs//args.gulp)
if nspecs % args.gulp:
    loop_iters += 1
fn_clean = args.fn.strip('.fil')
fdp_fn = f"{fn_clean}_fdp.fil"
new_fil = open(fdp_fn, "wb")
write_header(header, new_fil)
print(f"Output will be written to {fdp_fn}")
fil = open(args.fn, "rb")
fil.seek(hdrlen)

if args.stats:
    skews = np.zeros((loop_iters, nchans))
    kurtoses = np.zeros((loop_iters, nchans))
    s1 = np.zeros((loop_iters, nchans))
    s2 = np.zeros((loop_iters, nchans))
    num_unmasked_points = np.zeros((loop_iters, nchans), dtype=np.int)
    n = np.zeros((loop_iters), dtype=np.int)

additional_fils = []
if args.downsamp is not None:
    for i, d in enumerate(args.downsamp):
        add_fn = f"{fn_clean}_fdp_t{d}.fil"
        print(f"Also outputting downsampled file: {add_fn}")
        additional_fils.append(open(add_fn, "wb"))
        add_header = copy.deepcopy(header)
        add_header["tsamp"] =  header["tsamp"] * d
        if header.get("nsamples", ""):
            add_header["nsamples"] = int(header["nsamples"] // d)
        write_header(add_header, additional_fils[i])

for i in range(loop_iters):
    print(f"{i+1}/{loop_iters}", end='\r', flush=True)
    spec = (
        np.fromfile(fil, count=args.gulp*nchans, dtype=arr_dtype)
        .reshape(-1, nchans)
    )
    # has shape (nspec, nchans) so it plays nice with brodcasting
    med = np.median(spec,axis=0)
    std = np.std(spec,axis=0)
    #set the threshold
    threshold = med - args.thresh_sig*std
    #find values below threshold and replace with the median
    mask = np.where(spec < threshold)
    spec[mask] = med[mask[1]]
    new_fil.write(spec.ravel().astype(arr_dtype))
    # calc stats
    if args.stats:
        tmp = np.ma.array(spec, mask=(mask | (spec==0)))
        skews[i,:] = skew(tmp, axis=0, bias=False).filled(np.nan)
        kurtoses[i,:] = kurtosis(tmp, axis=0, bias=False).filled(np.nan)
        s1[i,:] = tmp.sum(axis=0).filled(np.nan)
        s2[i,:] = (tmp**2).sum(axis=0).filled(np.nan)
        num_unmasked_points[i,:] = (~tmp.mask).sum(axis=0)
        n[i] = tmp.shape[0]

    if additional_fils:
        for i,d in enumerate(args.downsamp):
            additional_fils[i].write(spec[::d,:].ravel().astype(arr_dtype))


fil.close()
new_fil.close()
# save stats
if args.stats:
    np.savez(
        f"{fdp_fn[:-4]}_stats.npz",
        skew=skews,
        kurtosis=kurtoses,
        s1=s1,
        s2=s2,
        num_unmasked_points=num_unmasked_points,
        n=n
    )
if additional_fils:
    for add_fil in additional_fils:
        add_fil.close()
