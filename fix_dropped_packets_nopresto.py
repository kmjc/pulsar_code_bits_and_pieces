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

# added mask "wiggle"
# this thing is due to dropped packets and is time dependent - it starts before and finishes after it 
# is detectable via this method
# to overcome this also mask +-<fdp_tscrunch> time samples either side of masked range

# realized with uint8 I'm probably hitting overflow errors with the tscrunching
# already factored that into the stats stuff
# and np.add.reduce changes the dtype if there isn't sufficient precision
# But my write for the tscrunched fils was definitely overflowing

# add logging
# add handle_exception

import numpy as np
from presto_without_presto import sigproc
from sigproc_utils import get_dtype, get_nbits, write_header
import argparse
import copy
from scipy.stats.mstats import skew, kurtosis
import sys
from gen_utils import handle_exception
import logging

# log unhandled exception
sys.excepthook = handle_exception

def get_fdp_mask(arr, axis=0, sigma=4.5):  #, debug=False):
    """
    Get a mask of where arr is < median - sigma*std
    where the median and std have been calulates after masking out any 0s.
    Returns a boolean array with the same shape as arr.

    axis = time axis of array, e.g. for (nspec,nchan) axis=0"""
    tmp_arr = np.ma.array(arr, mask=(arr == 0))
    meds = np.ma.median(tmp_arr, axis=axis)
    stds = np.ma.std(tmp_arr, axis=axis)
    del tmp_arr
    threshold = meds - sigma * stds
    out = arr < threshold
#    if debug:
#        print("saving things from mfdp calcs for debug")
#        np.savez("fdp_debug_mfdp_stages.npz", tmp_arr=np.ma.array(arr, mask=(arr == 0)), meds=meds, stds=stds, threshold=threshold, out=out)
    return out


def tscrunch(arr, fac):
    """Scrunch (by summing) a (nspec,nchan) array in time by <fac>
    Returns array of shape (nspec/fac, nchan)
    If nspec does not divide nicely into fac you get (nspec//fac+1, nchan)"""
    remainder = arr.shape[0] % fac
    if remainder:
        tmp = arr[:-remainder,:]
        excess = arr[-remainder:,:]
        return_nint = arr.shape[0] // fac + 1
        mout = np.zeros((return_nint, arr.shape[1]), dtype=arr.dtype)
        mout[:-1,:] = np.add.reduce(tmp.reshape(-1, fac, tmp.shape[-1]), 1)
        mout[-1,:] = np.add.reduce(excess, 0)
        return mout
    else:
        return np.add.reduce(arr.reshape(-1,fac,arr.shape[-1]), 1)
    
def tscrunch1d(arr, fac):
    """Scrunch (by summing) a (nspec) array in time by <fac>
    Returns array of shape (nspec/fac)
    If nspec does not divide nicely into fac you get (nspec//fac+1)"""
    remainder = arr.shape[0] % fac
    if remainder:
        tmp = arr[:-remainder]
        excess = arr[-remainder:]
        return_nint = arr.shape[0] // fac + 1
        mout = np.zeros((return_nint), dtype=arr.dtype)
        mout[:-1] = np.add.reduce(tmp.reshape(-1, fac), 1)
        mout[-1] = np.add.reduce(excess, 0)
        return mout
    else:
        return np.add.reduce(arr.reshape(-1,fac), 1)


if  __name__ == "__main__":
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
        help="Downsampling factor/s to apply. Multiple factors should be separated by spaces. (NB this sums the time samples together)",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Also calculate and save some stats: skewness, kurtosis, s1 & s2 for spectral kurtosis, number of unmasked time samples summed, total number of time samples",
    )

    parser.add_argument(
        "--rfifind_gulp",
        type=int,
        help="gulp using for rfifind (generally blocks x 2400), if gulp is a mulitple of rfifind_gulp this will be used for the stats calculations"
    )

    #parser.add_argument(
    #    "--debug",
    #    action='store_true',
    #    help="Output more messages for debugging purposes. Only do first gulp and output many intermediary products into fdp_debug*.npz files",
    #)

    parser.add_argument(
        "--log", type=str, help="name of file to write log to", default=None
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="Increase logging level to debug",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.INFO,
    )


    args = parser.parse_args()

    if args.log is not None:
        logging.basicConfig(
            filename=args.log,
            filemode="w",
            format="%(asctime)s %(levelname)s:%(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            level=args.loglevel,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s:%(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            level=args.loglevel,
            stream=sys.stdout,
        )


    logging.info(f"Running fdp with args:\n{args}")

    if args.fdp_tscrunch < 1:
        raise AttributeError(f"fdp_tscrunch ({args.fdp_tscrunch}) must be >=1")
    if args.gulp % args.fdp_tscrunch:
        logging.info(f"Gulp {args.gulp} is not divisible by fdp_tscrunch {args.fdp_tscrunch}")
        gulp = (int(args.gulp // args.fdp_tscrunch) + 1) * args.fdp_tscrunch
        logging.info(f"Gulp adjusted to {gulp}")
    else:
        gulp = args.gulp

    if args.rfifind_gulp is not None and args.stats:
        if gulp % args.rfifind_gulp:
            logging.warning(f"gulp {gulp} is not divisible by rfifind_gulp {args.rfifind_gulp}. Stats will be written using gulp")
            stats_gulp = gulp
            stats_gulp_per_gulp = 1
        else:
            stats_gulp = args.rfifind_gulp
            stats_gulp_per_gulp = int(gulp/stats_gulp)
        logging.info(f"There will be {stats_gulp_per_gulp} stats-intervals in every gulp")


    header, hdrlen = sigproc.read_header(args.fn)
    nspecs = int(sigproc.samples_per_file(args.fn, header, hdrlen))
    nchans = header["nchans"]
    arr_dtype = get_dtype(header["nbits"])

    # make tscrunch_dtype if need it
    if header["nbits"] == 8:
        tscrunch_dtype =  np.uint16
    elif header["nbits"] == 16:
        tscrunch_dtype = np.float32  # ths one probably isn't necessary, but maybe?
    else:
        tscrunch_dtype = arr_dtype


    if header["nifs"] != 1:
        raise AttributeError(f"Code not written to deal with unsummed polarization data")

    logging.debug(f"Read filterbank {args.fn} header:\n{header}")

    # loop through chunks
    loop_iters = int(nspecs // gulp)
    if nspecs % gulp:
        loop_iters += 1
    logging.info(f"Loop through in {loop_iters} iterations")
    # find shape for stats
    if args.stats:
        if stats_gulp_per_gulp == 1:
            stats_loop_iters = loop_iters
        else:
            stats_loop_iters = int(nspecs // stats_gulp)
            if nspecs % stats_gulp:
                stats_loop_iters += 1
        logging.info(f"Will be {stats_loop_iters} intervals in stats file")


    fn_clean = args.fn.strip(".fil")  # for some incredibly strange reason I cannot figure out for my filename beginning with fake they -> ake!!
    fn_clean = args.fn.split(".fil")[0]  # this doesn not
    #if args.debug:
    #    fn_clean += "_debug"
    fdp_fn = f"{fn_clean}_fdp.fil"
    new_fil = open(fdp_fn, "wb")
    write_header(header, new_fil)
    logging.info(f"Output will be written to {fdp_fn}")
    fil = open(args.fn, "rb")
    fil.seek(hdrlen)

    if args.stats:
    #    skews = np.zeros((loop_iters, nchans))
    #    kurtoses = np.zeros((loop_iters, nchans))
        s1 = np.zeros((stats_loop_iters, nchans))
        s2 = np.zeros((stats_loop_iters, nchans))
        num_unmasked_points = np.zeros((stats_loop_iters, nchans), dtype=int)
        n = np.zeros((stats_loop_iters), dtype=int)

        logging.debug(f"Initilaized stats arrays with shape {(stats_loop_iters, nchans)}")

    additional_fils = []
    if args.downsamp is not None:
        for i, d in enumerate(args.downsamp):
            add_fn = f"{fn_clean}_fdp_t{d}.fil"
            logging.info(f"Also outputting downsampled file: {add_fn}")
            additional_fils.append(open(add_fn, "wb"))
            add_header = copy.deepcopy(header)
            add_header["tsamp"] = header["tsamp"] * d
            if header.get("nsamples", ""):
                add_header["nsamples"] = int(header["nsamples"] // d)
            write_header(add_header, additional_fils[i])

    #if args.debug:
    #    loop_iters = 1

    for i in range(loop_iters):
        logging.info(f"{i+1}/{loop_iters}")
        spec = np.fromfile(fil, count=gulp * nchans, dtype=arr_dtype).reshape(-1, nchans)
        # has shape (nspec, nchans) so it plays nice with brodcasting

        logging.debug(f"Read data into shape {spec.shape}")

        mzeros = spec == 0
        logging.debug(f"number of zeros in data: {mzeros.sum()} ({mzeros.sum()/mzeros.size})")

        # tscrunch if necessary
        if args.fdp_tscrunch != 1:
            logging.info(f"tscrunching by a factor of {args.fdp_tscrunch} when searching for drops")
            working_spec = tscrunch(spec, args.fdp_tscrunch)
        else:
            working_spec = spec

        logging.debug(f"working spectra of shape {working_spec.shape}")

        # get the (tscrunched) dropout mask
        #mfdp = get_fdp_mask(working_spec, axis=0, sigma=args.thresh_sig, debug=args.debug).data
        mfdp = get_fdp_mask(working_spec, axis=0, sigma=args.thresh_sig).data
        logging.debug(f"mfdp mask of shape {mfdp.shape}\nNumber masked by mfdp is {mfdp.sum()} ({mfdp.sum()/mfdp.size})")

        # wiggle mfdp
        logging.debug("wiggling fdp mask")
        mfdp[0:-1,:] = (mfdp[0:-1,:] | mfdp[1:,:])
        mfdp[1:,:] = (mfdp[1:,:] | mfdp[0:-1,:])

        # convert mask to full time resolution if necessary, combine with zeros mask
        if args.fdp_tscrunch != 1:
            mtot = np.repeat(mfdp, args.fdp_tscrunch, axis=0) | mzeros
        else:
            mtot = mfdp | mzeros

        logging.debug(f"mtot made, of shape {mtot.shape}, mask amount {mtot.sum()} ({mtot.sum()/mtot.size} of data)")
    #    if args.debug:
    #        print("Saving spec, mzeros, working_spec, mfdp, mtot for debug to fdp_debug.npz")
    #        np.savez("fdp_debug.npz", spec=spec, working_spec=working_spec, mfdp=mfdp, mtot=mtot)

        del mfdp
        del mzeros

        # get medians to replace masked values with, not including to-be-masked values in the calculation
        tmp = np.ma.array(spec, mask=(mtot)).astype(int)
        meds_fullres = np.ma.median(tmp, axis=0)
        if not args.stats:
            del tmp

        # replace masked values with the median
        spec[mtot] = meds_fullres[np.where(mtot)[1]]

    #    if args.debug:
    #        print("saving output spectra")
    #        np.savez("fdp_debug_spec_out.npz", spec_out=spec)

        new_fil.write(spec.ravel().astype(arr_dtype))
        if additional_fils:
            for ii, d in enumerate(args.downsamp):
                scrunched_arr =  tscrunch(spec, d)
                additional_fils[ii].write(scrunched_arr.ravel().astype(tscrunch_dtype))
                del scrunched_arr

        # calc stats
        if args.stats:
            del spec
            jmin = i*stats_gulp_per_gulp
            jmax = min((i+1)*stats_gulp_per_gulp, stats_loop_iters)
    #       skews[i, :] = skew(tmp, axis=0, bias=False)
    #       kurtoses[i, :] = kurtosis(tmp, axis=0, bias=False)
            s1[jmin:jmax, :] = tscrunch(tmp, stats_gulp)
            s2[jmin:jmax, :] = tscrunch(tmp**2, stats_gulp)
            num_unmasked_points[jmin:jmax, :] = tscrunch((~tmp.mask).astype(int), stats_gulp)
            n[jmin:jmax] = tscrunch1d(np.ones((tmp.shape[0])), stats_gulp)


    fil.close()
    new_fil.close()
    # save stats
    if args.stats:  # and not args.debug:
        logging.info(f"Writing stats to {fdp_fn[:-4]}_stats.npz")
        np.savez(
            f"{fdp_fn[:-4]}_stats.npz",
    #        skew=skews,
    #        kurtosis=kurtoses,
            s1=s1,
            s2=s2,
            num_unmasked_points=num_unmasked_points,
            n=n,
            gulp=stats_gulp,
        )
    if additional_fils:
        for add_fil in additional_fils:
            add_fil.close()

    sys.exit(0)
