# chunk fdmt a filterbank and save as hdf5
# don't worry about the inf file,
# thinking tht after I've got this bit sorted I can nab the hdf5 1 row at a time
# pad, write the .dat's and the .inf's
import sys, os
import numpy as np
from fdmt.cpu_fdmt import FDMT
import argparse
import time
import copy
import logging

from presto_without_presto import sigproc
from presto_without_presto.psr_utils import choose_N
from presto_without_presto.sigproc import ids_to_telescope, ids_to_machine
from presto_without_presto.infodata import infodata2
from sigproc_utils import radec2string, get_fmin_fmax_invert, get_dtype

from chunk_dedisperse import (
    inverse_DM_delay,
    get_maxDT_DM,
    check_positive_float,
    not_zero_or_none,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""
        FDMT incoherently dedispersion a filterbank.
        (You probably want to pre-mask and remove the baseline from the filterbank)

        Outputs presto-style .dat and .inf files

        Uses the Manchester-Taylor 1/2.4E-4 convention for dedispersion
        """,
    )

    parser.add_argument("filename", type=str, help="Filterbank file to dedisperse")

    parser.add_argument(
        "gulp",
        type=int,
        help="Number of spectra (aka number of time samples) to read in at once",
    )

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "-d",
        "--dm",
        type=float,
        default=0,
        help="Max DM (cm-3pc) to dedisperse to, must be positive",
    )
    g.add_argument(
        "-t",
        "--maxdt",
        type=check_positive_float,
        default=0,
        help="Number of time samples corresponding to the max DM delay between the lowest and highest channel\n(must be positive)",
    )

    parser.add_argument(
        "--atdm",
        help="DM to which filterbank has already been incoherently dedispersed",
        default=0,
        type=float,
    )

    parser.add_argument(
        "-o", "--outfilename", type=str, default=None, help="Output .h5 file"
    )

    parser.add_argument(
        "--tophalf", action="store_true", help="Only run on the top half of the band"
    )

    parser.add_argument(
        "--dmprec", type=int, default=3, help="DM precision (for filenames)"
    )

    parser.add_argument(
        "--pad",
        action="store_true",
        help="Pad the .dat files to a highly factorable length",
    )

    parser.add_argument(
        "--log",
        type=str,
        help="name of file to write log to",
        default="chunk_fdmt_fb2datinf.log",
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

    logging.basicConfig(
        filename=args.log,
        filemode="w",
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=args.loglevel,
    )

    logging.info(f"Working on file: {args.filename}")
    header, hdrlen = sigproc.read_header(args.filename)
    nsamples = int(sigproc.samples_per_file(args.filename, header, hdrlen))
    logging.debug(header)

    if header["nifs"] != 1:
        raise ValueError(f"Code not written to deal with unsummed polarization data")

    # calculate/get parameters from header
    tsamp = header["tsamp"]
    nchans = header["nchans"]
    arr_dtype = get_dtype(header["nbits"])
    fmin, fmax, invertband = get_fmin_fmax_invert(header)

    logging.info(
        f"Read from file:\n"
        f"fmin: {fmin}, fmax: {fmax}, nchans: {nchans} tsamp: {tsamp} nsamples: {nsamples}\n",
    )

    # define fs, for CD/maxDT calculation
    # FDMT computes based on shift between fmin and fmax
    if args.tophalf:
        logging.info("Only using top half of the band")
        fs = np.linspace(fmin + (fmax - fmin) / 2, fmax, nchans // 2, endpoint=True)
    else:
        fs = np.linspace(fmin, fmax, nchans, endpoint=True)
    DM, maxDT, max_delay_s = get_maxDT_DM(args.dm, args.maxdt, tsamp, fs)

    logging.info(f"FDMT incoherent DM is {DM}")
    logging.info(f"Maximum delay need to shift by is {max_delay_s} s")
    logging.info(f"This corresponds to {maxDT} time samples\n")
    if DM == 0:
        sys.exit("DM=0, why are you running this?")

    ngulps = nsamples // args.gulp
    if nsamples % args.gulp:
        if (nsamples % args.gulp) < maxDT:
            raise RuntimeWarning(
                f"gulp ({args.gulp}) is not ideal. Will cut off {nsamples % args.gulp} samples at the end.\n"
                f"Try running get_good_gulp.py --fdmt --maxdt {maxDT} {args.filename}\n"
            )
        else:
            ngulps += 1
            logging.info(
                f"\nWill process the file in {ngulps-1} gulps of {args.gulp}, and one of {nsamples % args.gulp}"
            )
    else:
        logging.info(f"\nWill process the file in {ngulps} gulps of {args.gulp}")

    if args.gulp <= maxDT:
        raise RuntimeError(
            f"gulp ({args.gulp}) must be larger than maxDT ({maxDT})\n"
            f"Try running get_good_gulp.py -t {maxDT} {args.filename}\n"
        )

    # initialize FDMT class object
    if args.tophalf:
        fd = FDMT(
            fmin=fmin + (fmax - fmin) / 2, fmax=fmax, nchan=nchans // 2, maxDT=maxDT
        )
    else:
        fd = FDMT(fmin=fmin, fmax=fmax, nchan=nchans, maxDT=maxDT)
    logging.info(
        f"FDMT initialized with fmin {fd.fmin}, fmax {fd.fmax}, nchan {fd.nchan}, maxDT {fd.maxDT}\n"
    )

    # Define slices to return intensities in read_gulp
    if args.tophalf:
        read_inv_slc = slice(nchans // 2 - 1, None, -1)
        read_slc = slice(nchans // 2, None, None)
    else:
        read_inv_slc = slice(None, None, -1)
        read_slc = slice(None, None, None)

    # Don't want an "if invertband:" in my loop, define function to return data as flipped/not
    # also need to transpose anyway for FDMT so it's (nchans, gulp)
    # (NB FDMT needs the lowest freq channel to be at index 0)
    if invertband:

        def read_gulp(filfile, gulp, nchans, arr_dtype):
            """Read in next gulp and prep it to feed into fdmt"""
            data = np.fromfile(filfile, count=gulp * nchans, dtype=arr_dtype).reshape(
                -1, nchans
            )
            return data[:, read_inv_slc].T

    else:

        def read_gulp(filfile, gulp, nchans, arr_dtype):
            """Read in next gulp and prep it to feed into fdmt"""
            data = np.fromfile(filfile, count=gulp * nchans, dtype=arr_dtype).reshape(
                -1, nchans
            )
            return data[:, read_slc].T

    # initialize outfile
    if not_zero_or_none(args.outfilename):
        outfilename = args.outfilename
        if outfilename[-3] != ".h5":
            outfilename += ".h5"
    else:
        # choosing to save it as the actual max DM in the h5 output, not the DM corresponding to maxDT
        outfilename = (
            args.filename[:-4]
            + f"_fdmtDM{inverse_DM_delay((maxDT-1)*tsamp, fmin, fmax):.3f}.h5"
        )

    if os.path.exists(outfilename):
        logging.info(f"{outfilename} already exists, deleting\n")
        os.remove(outfilename)

    # HERE compute and store DMs
    # check it's arange(maxDT) and not arange(1, maxDT + 1)
    # Hao said that's correct
    if args.tophalf:
        flo = fmin + (fmax - fmin) / 2
    else:
        flo = fmin
    DMs = inverse_DM_delay(np.arange(maxDT) * tsamp, flo, fmax)
    DMs += args.atdm
    logging.info(
        f"DMs in h5 are from {DMs[0]} to {DMs[-1]} in steps of {DMs[1] - DMs[0]}\n"
    )

    # name all output dat files
    dat_names = [f"{args.filename[:-4]}_DM{aDM:.{args.dmprec}f}.dat" for aDM in DMs]
    dm_indices = range(len(DMs))

    # read in data
    filfile = open(args.filename, "rb")
    filfile.seek(hdrlen)

    logging.info("Reading in first gulp")
    # Do first gulp separately
    intensities = read_gulp(filfile, args.gulp, nchans, arr_dtype)
    fd.reset_ABQ()
    logging.info(f"Starting gulp 0")
    logging.debug(f"Size of chunk: {sys.getsizeof(intensities.base)/1000/1000} MB")
    t0 = time.perf_counter()
    out = fd.fdmt(intensities, padding=True, frontpadding=True, retDMT=True)
    logging.debug(f"Size of fdmt A, {fd.A.shape}: {sys.getsizeof(fd.A)/1000/1000} MB")
    logging.debug(f"Size of fdmt B, {fd.B.shape}: {sys.getsizeof(fd.B)/1000/1000} MB")
    t1 = time.perf_counter()
    logging.info(f"Writing gulp 0")
    # write mid_arr
    for i in dm_indices:
        with open(dat_names[i], "wb") as fout:
            fout.write(out[i, maxDT:-maxDT])

    t2 = time.perf_counter()
    logging.info(f"Completed gulp 0 in {t1-t0} s, wrote in {t2-t1} s\n")

    # setup for next iteration
    prev_arr = np.zeros((maxDT, maxDT), dtype=intensities.dtype)
    prev_arr += out[:, -maxDT:]
    intensities = read_gulp(filfile, args.gulp, nchans, arr_dtype)

    if ngulps > 1:
        for g in np.arange(1, ngulps):
            intensities = read_gulp(filfile, args.gulp, nchans, arr_dtype)
            fd.reset_ABQ()
            out = fd.fdmt(intensities, padding=True, frontpadding=True, retDMT=True)
            prev_arr += out[:, :maxDT]

            # write prev_arr and mid_arr
            for i in dm_indices:
                with open(dat_names[i], "ab") as fout:
                    fout.write(prev_arr[i, :])
                    fout.write(out[i, maxDT:-maxDT])
            logging.debug(f"Completed gulp {g}")

            # reset for next gulp
            # setting it to 0 and using += stops prev_arr changing when out does
            prev_arr[:, :] = 0
            prev_arr += out[:, -maxDT:]
    t3 = time.perf_counter()
    logging.info(f"FDMT completed in {t3-t0} s")

    # find length of data written
    if (nsamples % args.gulp) < maxDT:
        origNdat = int(nsamples // args.gulp) * args.gulp - maxDT
    else:
        origNdat = nsamples - maxDT

    PAD_MEDIAN_NSAMP = 4096
    if args.pad:
        logging.info("\nPadding dat files")
        # find good N
        N = choose_N(origNdat)
        # get medians of last PAD_MEDIAN_NSAMP samples if possible
        if out.shape[1] - maxDT < PAD_MEDIAN_NSAMP:
            logging.info(
                f"Warning padding using median over last {out.shape[1] - maxDT} samples rather than {PAD_MEDIAN_NSAMP}"
            )
            meds = np.median(out[:, :-maxDT])
        else:
            logging.info(f"Padding using median over last {PAD_MEDIAN_NSAMP} samples")
            meds = np.median(out[:, -(PAD_MEDIAN_NSAMP + maxDT) : -maxDT])

        for i in dm_indices:
            with open(dat_names[i], "ab") as fout:
                padding = np.zeros((N - origNdat,), dtype=intensities.dtype) + meds[i]
                fout.write(padding)
        t4 = time.perf_counter()
        logging.info(f"Padding completed in {t4-t3} s")
    else:
        N = origNdat
        t4 = time.perf_counter()

    logging.info(f"{len(fout_indicesices)} dat files written")

    # write all the inf files:
    logging.info(f"\nWriting inf files:")
    lofreq = fmin + abs(header["foff"]) / 2
    infdict = dict(
        basenm=args.filename[:-4],
        telescope=ids_to_telescope[header["telescope_id"]],
        instrument=ids_to_machine[header["machine_id"]],
        object=header.get("source_name", "Unknown"),
        RA=radec2string(header["src_raj"]),
        DEC=radec2string(header["src_dej"]),
        observer="unset",
        epoch=header["tstart"],
        bary=0,
        dt=header["tsamp"],
        lofreq=lofreq,
        BW=abs(header["nchans"] * header["foff"]),
        N=N,
        numchan=header["nchans"],
        chan_width=abs(header["foff"]),
        analyzer=os.environ.get("USER"),
    )

    if args.pad:
        infdict["breaks"] = 1
        infdict["onoff"] = [(0, origNdat - 1), (N - 1, N - 1)]
    else:
        infdict["breaks"] = 0

    for aDM in DMs:
        inf_fname = f"{args.filename[:-4]}_DM{aDM:.{args.dmprec}f}.inf"
        specific_infdict = copy.copy(infdict)
        specific_infdict["DM"] = aDM
        inf = infodata2(specific_infdict)
        inf.to_file(inf_fname, notes="fdmt")
        logging.info(f"Wrote {inf_fname}")
    logging.info(f"Done")
