import sys, os
import numpy as np
import argparse
import time
import copy
import yaml
import math
import logging

import pyfdmt

from presto_without_presto import sigproc
from presto_without_presto.sigproc import ids_to_telescope, ids_to_machine
from sigproc_utils import radec2string, get_fmin_fmax_invert, get_dtype

from chunk_dedisperse import (
    inverse_DM_delay,
    get_maxDT_DM,
    check_positive_float,
    not_zero_or_none,
    get_fs,
)

from gen_utils import handle_exception

BEAM_DIAM = 6182  # CHIME-specific, grabbed from a presto-generated inf file, haven't looked it up to confirm if it's accurate
# hackey, should have a dict/function which selects this based on the machine id etc
# but riptide breaks without this in the inf file

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="""
    FDMT incoherent dedispersion of a filterbank.
    (You probably want to pre-mask and remove the baseline from the filterbank)

    Outputs a binary file storing the chunked transform (a stream of float32s),
    and a yaml with the data necessary to process this and write an inf file.
    Process with bin_to_datinf.py

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
    "-o",
    "--outdir",
    type=str,
    default=".",
    help="Directory in which to write the output .fdmt and .fdmt.yaml files",
)

parser.add_argument(
    "--atdm",
    help="DM to which filterbank has already been incoherently dedispersed",
    default=0,
    type=float,
)

parser.add_argument(
    "--tophalf", action="store_true", help="Only run on the top half of the band"
)

parser.add_argument(
    "--dmprec", type=int, default=3, help="DM precision (for filenames)"
)

parser.add_argument(
    "--split_file",
    action="store_true",
    help="Split the output file into manageable smaller files. Must set --max_size",
)

parser.add_argument(
    "--max_size", type=float, help="Set maximum size for each .fdmt file (in bytes)"
)

parser.add_argument(
    "--yaml_only", action="store_true", help="Don't write fdmt files, only their yamls"
)

parser.add_argument(
    "--outdatbase",
    default="",
    type=str,
    help="Basename for dat files, e.g. 'base' would give a bunch of base_DMx.xx.dat files",
)

# currently no way to change the DM step in FDMT but can choose not to write all DMs out
parser.add_argument(
    "--mindmstep",
    type=float,
    help="""Minimum DM stepsize to write.
    e.g. if FDMT's DM stepsize is 0.002 and mindmstep is set to 0.004, only every other FDMT DM will be written to file""",
)

parser.add_argument(
    "--padto",
    type=int,
    help="Pad to this number of samples (actual padding done by bin_to_datinf, this records the N in the .fdmt.yaml)"
)

#parser.add_argument(
#    "--num_threads",
#    type=int,
#    default=1,
#    help="Number of threads to use in fdmt. Default is 1",
#)

parser.add_argument(
    "--fdmt_corr_fac",
    type=float,
    default=1,
    help="corr_fac to use in fdmt (corr = df * corr_fac)",
)

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
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=args.loglevel,
        stream=sys.stdout,
    )

# log unhandled exception
sys.excepthook = handle_exception

if args.split_file and args.max_size is None:
    parser.error("--split_file requires --max_size")


logging.info(f"Ran script in {os.getcwd()}")
############################################################################
# A LOT of setup

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
# pyfdmt uses center channels and fs[0] = highest freq
if args.tophalf:
    logging.info("Only using top half of the band")
    fs = get_fs(fmin, fmax, nchans, invertband=invertband)[:int(nchans/2)]
    #fs = np.linspace(fmin + (fmax - fmin) / 2, fmax, nchans // 2, endpoint=True)
else:
    #fs = np.linspace(fmin, fmax, nchans, endpoint=True)
    fs = get_fs(fmin, fmax, nchans, invertband=invertband)

DM, maxDT, max_delay_s = get_maxDT_DM(args.dm, args.maxdt, tsamp, fs)


logging.info(f"pyfdmt incoherent DM is {DM}")
logging.info(f"Maximum delay need to shift by is {max_delay_s} s")
logging.info(f"This corresponds to {maxDT} time samples\n")
if DM == 0:
    sys.exit("DM=0, why are you running this?")

# checks and other stuff based on gulp size
ngulps = nsamples // args.gulp
weird_last_gulp = False
if nsamples % args.gulp:
    if (nsamples % args.gulp) < maxDT:
        logging.warning(
            f"gulp ({args.gulp}) is not ideal. Will cut off {nsamples % args.gulp} samples at the end"
        )
        logging.warning(
            f"Try running get_good_gulp.py --fdmt --maxdt {maxDT} {args.filename}\n"
        )
        logging.info(f"Will process the file in {ngulps} gulps of {args.gulp}")
    else:
        weird_last_gulp = True
        ngulps += 1
        logging.info(
            f"Will process the file in {ngulps-1} gulps of {args.gulp}, and one of {nsamples % args.gulp}"
        )
else:
    logging.info(f"Will process the file in {ngulps} gulps of {args.gulp}")

if args.gulp <= maxDT:
    raise RuntimeError(
        f"gulp ({args.gulp}) must be larger than maxDT ({maxDT})\n"
        f"Try running get_good_gulp.py -t {maxDT} {args.filename}\n"
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

# pyfdmt is the opposite. Needs highest freq channel to be 0
if invertband:
    logging.debug(f"Frequency slice used to read data: {read_inv_slc}\n")

    def read_gulp(filfile, gulp, nchans, arr_dtype):
        """Read in next gulp and prep it to feed into fdmt"""
        data = np.fromfile(filfile, count=gulp * nchans, dtype=arr_dtype).reshape(
            -1, nchans
        ).astype(np.float32)
        #return data[:, read_inv_slc].T
        return data[:, read_slc].T

else:
    logging.debug(f"Frequency slice used to read data: {read_slc}\n")

    def read_gulp(filfile, gulp, nchans, arr_dtype):
        """Read in next gulp and prep it to feed into fdmt"""
        data = np.fromfile(filfile, count=gulp * nchans, dtype=arr_dtype).reshape(
            -1, nchans
        ).astype(np.float32)
        #return data[:, read_slc].T
        return data[:, read_inv_slc].T

# Compute and store DMs
DMs = inverse_DM_delay(np.arange(maxDT) * tsamp, fs.min(), fs.max())
DMs += args.atdm
logging.info(f"FDMT DMs are from {DMs[0]} to {DMs[-1]} in steps of {DMs[1] - DMs[0]}")

if not_zero_or_none(args.mindmstep):
    DM_downsamp = int(args.mindmstep // (DMs[1] - DMs[0]))
    logging.info(f"Downsampling output DMs by {DM_downsamp}")
else:
    DM_downsamp = 1
logging.info(
    f"DMs to be written are from {DMs[0]} to {DMs[::DM_downsamp][-1]} in steps of {DMs[DM_downsamp] - DMs[0]}"
)
numDMs = math.ceil(maxDT / DM_downsamp)
logging.info(f"{numDMs} DMs to be written")

# find length of data to be  written
if (nsamples % args.gulp) < maxDT:
    origNdat = int(nsamples // args.gulp) * args.gulp - maxDT
else:
    origNdat = nsamples - maxDT
logging.debug(f"Length of data to be written is {origNdat} samples")

# set up output file/s
if args.split_file:
    nbytes = header["nbits"] // 8
    tot_filesize = origNdat * numDMs * ()
    dms_per_file = math.floor(args.max_size / (origNdat * nbytes))
    # adjust if not a multiple of DM_downsamp
    if dms_per_file % DM_downsamp:
        dms_per_file = int((dms_per_file // DM_downsamp) * DM_downsamp)

    nfiles = math.ceil(numDMs / dms_per_file)
    logging.info(f"Splitting the output into {nfiles} files of {dms_per_file} DMs")
    if nfiles > 1000:
        logging.warning(
            f"number of files to write ({nfiles}) is over 1000, might get OSError"
        )

    fouts_indices = list(range(nfiles))
    fouts_names = []  # contains names of files
    dm_slices = (
        []
    )  # contains dm_slices, so for file with fouts_indices i DMs[dm_slices[i]] will give you the DMs it contains
    for ii in fouts_indices:
        start = ii * dms_per_file * DM_downsamp
        if ii == fouts_indices[-1]:
            end = maxDT
        else:
            end = (ii + 1) * dms_per_file * DM_downsamp
        fout_name = f"{args.filename[:-4]}_{start}-{end-1}.fdmt"
        fouts_names.append(fout_name)
        dm_slices.append(slice(start, end, DM_downsamp))
else:
    fouts_indices = [0]
    fouts_names = [f"{args.filename[:-4]}.fdmt"]
    dm_slices = [slice(None, None, DM_downsamp)]

# make outdir if it doesn't exist
os.makedirs(args.outdir, exist_ok=True)
logging.info(f"Output will be written in directory {args.outdir}")
logging.info(f"Outfiles:\n{fouts_names}")
logging.debug(f"DM slices:\n{dm_slices}")

if not args.yaml_only:
    fouts = [
        open(os.path.join(args.outdir, fout_name), "wb") for fout_name in fouts_names
    ]

dm_indices = range(len(DMs))


############################################################################
# Do FDMT

if not args.yaml_only:
    # read in data
    filfile = open(args.filename, "rb")
    filfile.seek(hdrlen)

    logging.info("Reading in first gulp")
    # Do first gulp separately
    intensities = read_gulp(filfile, args.gulp, nchans, arr_dtype)
    logging.info(f"Starting gulp 0")
    logging.debug(f"Shape of first chunk read: {intensities.shape}")
    logging.debug(f"Size of chunk: {sys.getsizeof(intensities.base)/1000/1000} MB")
    t0 = time.perf_counter()
    out = pyfdmt.transform(intensities, fs[0], fs[-1], header["tsamp"], 0, DM, frontpad=True)
    logging.debug(f"intensities shape: {intensities.shape}, out shape: {out.data.shape}")
    logging.debug(f"ndms: {len(out.dms)}")
    logging.debug(f"some dms: {out.dms[:3]}, {out.dms[-3:]}")
    t1 = time.perf_counter()

    logging.debug(f"Check: maxDT matches pyfdmt ymax: {maxDT==out.ymax}, maxDT:{maxDT}, ymax:{out.ymax}")
    # as delete out after each gulp for memory reasons, need to preserve some info
    ymax = out.ymax
    actual_DMs = out.dms + args.atdm
    logging.debug(f"Checking DMs match what expected: {(DMs == actual_DMs).all()}")

    logging.info(f"Writing gulp 0")
    # write mid_arr
    logging.debug(f"FDMT transform shape: {out.data.shape}")
    logging.debug(
        f"Only writing {out.ymax}:-{out.ymax} slice in time, should be {out.data.shape[1] - 2*out.ymax} samples"
    )
    for ii in fouts_indices:
        fouts[ii].write(out.data[dm_slices[ii], out.ymax:-out.ymax].ravel())

    t2 = time.perf_counter()
    logging.info(f"Completed gulp 0 in {t1-t0} s, wrote in {t2-t1} s\n")

    # setup for next iteration
    # for fdmt he shape is (maxDT, maxDT), for pyfdmt it's (ymax+1, ymax)
    prev_arr = np.zeros((out.ymax+1, out.ymax), dtype=intensities.dtype)
    prev_arr += out.data[:, -out.ymax:]
    out = None

    if ngulps > 1:
        for g in np.arange(1, ngulps):
            intensities = read_gulp(filfile, args.gulp, nchans, arr_dtype)
            out = pyfdmt.transform(intensities, fs[0], fs[-1], header["tsamp"], 0, DM, frontpad=True)
            prev_arr += out.data[:, :out.ymax]

            # write prev_arr and mid_arr
            # logging.debug(f"gulp {g} out array shape {out.data.shape}")
            # logging.debug(f"gulp {g} writing {prev_arr.shape[1] + out.data.shape[1] - 2*out.ymax} time samples")
            for ii in fouts_indices:
                fouts[ii].write(prev_arr[dm_slices[ii], :].ravel())
                fouts[ii].write(out.data[dm_slices[ii], out.ymax:-out.ymax].ravel())
            logging.debug(f"Completed gulp {g}")

            # reset for next gulp
            # setting it to 0 and using += stops prev_arr changing when out does
            prev_arr[:, :] = 0
            prev_arr += out.data[:, -out.ymax:]
            out = None

    for ii in fouts_indices:
        fouts[ii].close()
        logging.info(f"FDMT data written to {fouts_names[ii]}")
    filfile.close()
    t3 = time.perf_counter()
    logging.info(f"FDMT completed in {(t3-t0)/60/60} hrs\n")

else:
    logging.info("Yaml-only")
    t3 = time.perf_counter()


############################################################################
# Write useful information to a yaml file
# Construct a dictionary containing all the information necessary to make an inf file
logging.info(f"Writing yaml:")
lofreq = fmin + abs(header["foff"]) / 2
if args.outdatbase:
    basename = args.outdatbase
else:
    basename = args.filename[:-4]

inf_dict = dict(
    basenm=basename,
    telescope=ids_to_telescope[header.get("telescope_id", 0)],
    instrument=ids_to_machine[header.get("machine_id", 0)],
    object=header.get("source_name", "Unknown"),
    RA=radec2string(header.get("src_raj", 0)),
    DEC=radec2string(header.get("src_dej", 0)),
    observer="unset",
    epoch=header["tstart"],
    bary=0,
    dt=header["tsamp"],
    lofreq=lofreq,
    BW=abs(header["nchans"] * header["foff"]),
    numchan=header["nchans"],
    chan_width=abs(header["foff"]),
    analyzer=os.environ.get("USER"),
    waveband="Radio",
    beam_diam=BEAM_DIAM,  # hackey, see note at top of script where BEAM_DIAM is defined
)

if args.padto is not None:
    assert args.padto > origNdat
    inf_dict["N"] = args.padto
    inf_dict["breaks"] = 1
else:
    inf_dict["N"] = int(origNdat)
    inf_dict["breaks"] = 0
# set onoff in bin_to_datinf (yaml doesn't like tuples and this was easier)

# Construct a dictionary with extra information needed to assemble the dat and inf files
# General stuff that goes into every file:
yaml_dict = dict(
    ngulps=ngulps,
    gulp=args.gulp,
    inf_dict=inf_dict,
    maxDT=int(ymax),  # int is leftover from fdmt, not sure if still necessary
    origNdat=int(origNdat),
)

if weird_last_gulp:
    yaml_dict["ngulps"] = ngulps - 1
    yaml_dict["last_gulp"] = int(nsamples % args.gulp)

logging.debug("Dict values to go into every yaml file:")
logging.debug(f"{yaml_dict}")

# loop through each split file and write a yaml for each
inf_names = [f"{basename}_DM{aDM:.{args.dmprec}f}.inf" for aDM in actual_DMs]
for ii in fouts_indices:
    specific_yaml_dict = copy.copy(yaml_dict)

    slc = dm_slices[ii]
    specific_yaml_dict["inf_names"] = inf_names[slc]
    specific_yaml_dict["DMs"] = [float(aDM) for aDM in actual_DMs[slc]]

    # write yaml
    with open(os.path.join(args.outdir, f"{fouts_names[ii]}.yaml"), "w") as fyaml:
        yaml.dump(specific_yaml_dict, fyaml)
    logging.info(f"yaml written to {fouts_names[ii]}.yaml")

t4 = time.perf_counter()
logging.info(f"yamls written in {t4 - t3} seconds")

logging.info(f"Done")
