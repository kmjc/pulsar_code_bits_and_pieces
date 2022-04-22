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
import yaml
import math
import logging

from presto_without_presto import sigproc
from presto_without_presto.psr_utils import choose_N
from presto_without_presto.sigproc import (ids_to_telescope, ids_to_machine)
from sigproc_utils import (radec2string, get_fmin_fmax_invert, get_dtype)

from chunk_dedisperse import (
    inverse_DM_delay,
    get_maxDT_DM,
    check_positive_float,
    not_zero_or_none,
)



parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="""
    FDMT incoherently dedispersion a filterbank.
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
    "--pad", action='store_true', help="Compute numbers needed to pad the .dat files to a highly factorable length (untested)"
)

parser.add_argument(
    "--split_file", action='store_true', help="Split the output file into manageable smaller files. Must set --max_size"
)

parser.add_argument(
    "--max_size", type=float, help="Set maximum size for each file (in bytes)"
)

parser.add_argument(
    "--log", type=str, help="name of file to write log to", default="chunk_fdmt_fb2bin.log"
)

parser.add_argument(
    '-v', '--verbose',
    help="Increase logging level to debug",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.INFO,
)


args = parser.parse_args()

logging.basicConfig(
    filename=args.log,
    filemode='w',
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=args.loglevel,
    )


if args.split_file and args.max_size is None:
    parser.error("--split_file requires --max_size")


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
if args.tophalf:
    logging.info("Only using top half of the band")
    fs = np.linspace(
        fmin + (fmax - fmin)/2, fmax, nchans // 2, endpoint=True
    )
else:
    fs = np.linspace(
        fmin, fmax, nchans, endpoint=True
    )
DM, maxDT, max_delay_s = get_maxDT_DM(args.dm, args.maxdt, tsamp, fs)

logging.info(f"FDMT incoherent DM is {DM}")
logging.info(f"Maximum delay need to shift by is {max_delay_s} s")
logging.info(f"This corresponds to {maxDT} time samples\n")
if DM == 0:
    sys.exit("DM=0, why are you running this?")

# checks and other stuff based on gulp size
ngulps = nsamples // args.gulp
if nsamples % args.gulp:
    if (nsamples % args.gulp) < maxDT:
        log.warning(f"gulp ({args.gulp}) is not ideal. Will cut off {nsamples % args.gulp} samples at the end")
        log.warning(f"Try running get_good_gulp.py --fdmt --maxdt {maxDT} {args.filename}\n")
    else:
        weird_last_gulp = True
        ngulps += 1
        logging.info(f"\nWill process the file in {ngulps-1} gulps of {args.gulp}, and one of {nsamples % args.gulp}")
else:
    logging.info(f"\nWill process the file in {ngulps} gulps of {args.gulp}")

if args.gulp <= maxDT:
    raise RuntimeError(
        f"gulp ({args.gulp}) must be larger than maxDT ({maxDT})\n"
        f"Try running get_good_gulp.py -t {maxDT} {args.filename}\n"
    )

# initialize FDMT class object
if args.tophalf:
    fd = FDMT(fmin=fmin + (fmax - fmin)/2, fmax=fmax, nchan=nchans//2, maxDT=maxDT)
else:
    fd = FDMT(fmin=fmin, fmax=fmax, nchan=nchans, maxDT=maxDT)
logging.info(f"FDMT initialized with fmin {fd.fmin}, fmax {fd.fmax}, nchan {fd.nchan}, maxDT {fd.maxDT}\n")

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

# HERE compute and store DMs
# check it's arange(maxDT) and not arange(1, maxDT + 1)
# Hao said that's correct
if args.tophalf:
    flo = fmin + (fmax - fmin)/2
else:
    flo = fmin
DMs = inverse_DM_delay(np.arange(maxDT) * tsamp, flo, fmax)
DMs += args.atdm
logging.info(
    f"DMs in h5 are from {DMs[0]} to {DMs[-1]} in steps of {DMs[1] - DMs[0]}\n"
)

# find length of data to be  written
if (nsamples % args.gulp) < maxDT:
    origNdat = int(nsamples // args.gulp) * args.gulp - maxDT
else:
    origNdat = nsamples - maxDT
logging.debug(f"Length of data to be written is {origNdat} samples")

# set up output file/s
if args.split_file:
    tot_filesize = origNdat * maxDT * (header["nbits"] // 8)
    dms_per_file = math.floor(maxDT / tot_filesize * args.max_size)
    nfiles = math.ceil(maxDT / dms_per_file)
    logging.info(f"Splitting the output into {nfiles} files of {dms_per_file} DMs")
    if nfiles > 1000:
        log.warning(f"number of files to write ({nfiles}) is over 1000, might get OSError")

    fouts = []  # contains open file objects
    fouts_indices = list(range(nfiles))
    fouts_names = []  # contains names of files
    dm_slices = []  # contains dm_slices, so for file with fouts_indices i DMs[dm_slices[i]] will give you the DMs it contains
    for ii in fouts_indices:
        start = ii * dms_per_file
        if ii == fouts_indices[-1]:
            end = maxDT
        else:
            end = (ii + 1) * dms_per_file
        fout_name = f"{args.filename[:-4]}_{start}-{end-1}.fdmt"
        fouts_names.append(fout_name)
        fouts.append(open(fout_name, "wb"))
        dm_slices.append(slice(start, end))
    logging.info(f"Outfiles:\n{fouts_names}")
    logging.debug(f"DM slices:\n{dm_slices}")
else:
    fouts = [open(f"{args.filename[:-4]}.fdmt", "wb")]
    fouts_indices = [0]
    fout_names = [f"{args.filename[:-4]}.fdmt"]
    dm_slices = [slice(None)]

dm_indices = range(len(DMs))


############################################################################
# Do FDMT

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
logging.debug(
    f"Size of fdmt A, {fd.A.shape}: {sys.getsizeof(fd.A)/1000/1000} MB"
)
logging.debug(
    f"Size of fdmt B, {fd.B.shape}: {sys.getsizeof(fd.B)/1000/1000} MB"
)
t1 = time.perf_counter()
logging.info(f"Writing gulp 0")
# write mid_arr
for ii in fouts_indices:
    fouts[ii].write(out[dm_slices[ii], maxDT:-maxDT].ravel())

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
        for ii in fouts_indices:
            fouts[ii].write(prev_arr[dm_slices[ii],:].ravel())
            fouts[ii].write(out[dm_slices[ii], maxDT:-maxDT].ravel())
        logging.debug(f"Completed gulp {g}")

        # reset for next gulp
        # setting it to 0 and using += stops prev_arr changing when out does
        prev_arr[:, :] = 0
        prev_arr += out[:, -maxDT:]

for ii in fouts_indices:
    fouts[ii].close()
    logging.info(f"FDMT data written to {fouts_names[ii]}")
filfile.close()
t3 = time.perf_counter()
logging.info(f"FDMT completed in {(t3-t0)/60/60} hrs")



############################################################################
# Write useful information to a yaml file
# Construct a dictionary containing all the information necessary to make an inf file
logging.info(f"\nWriting yaml:")
lofreq = fmin + abs(header['foff'])/2
inf_dict = dict(
    basenm=args.filename[:-4],
    telescope=ids_to_telescope[header['telescope_id']],
    instrument=ids_to_machine[header['machine_id']],
    object=header.get('source_name', 'Unknown'),
    RA=radec2string(header['src_raj']),
    DEC=radec2string(header['src_dej']),
    observer='unset',
    epoch= header['tstart'],
    bary=0,
    dt=header['tsamp'],
    lofreq=lofreq,
    BW=abs(header['nchans'] * header['foff']),
    N=int(N),  # otherwise it's a numpy int which does not play nice with yaml
    numchan=header['nchans'],
    chan_width=abs(header['foff']),
    analyzer=os.environ.get( "USER" ),
)

# add padding-dependent inf_dict variables
PAD_MEDIAN_NSAMP = 4096
if args.pad:
    logging.info("\nRecording padding parameters for assembling dat files")
    # find good N
    N = choose_N(origNdat)
    logging.debug(f"In next stage of process, data will be padded from {origNdat} to {N} samples")
    # get medians of last PAD_MEDIAN_NSAMP samples if possible
    if out.shape[1] - 2*maxDT < PAD_MEDIAN_NSAMP:
        logging.warning(f"Padding using median over last {out.shape[1] - 2*maxDT} samples rather than {PAD_MEDIAN_NSAMP}")
        meds = np.median(out[:,maxDT:-maxDT])
    else:
        logging.info(f"Padding using median over last {PAD_MEDIAN_NSAMP} samples")
        meds = np.median(out[:, -(PAD_MEDIAN_NSAMP+maxDT):-maxDT])

    inf_dict['breaks'] = 1
    inf_dict['onoff'] = [(0, origNdat - 1), (N - 1, N - 1)]

else:
    N = origNdat
    logging.debug(f"\nNot padding data, will stay as {N} time samples")
    inf_dict['breaks'] = 0

inf_dict['N'] = int(N)


# Construct a dictionary with extra information needed to assemble the dat and inf files
# General stuff that goes into every file:
yaml_dict = dict(
    ngulps=ngulps,
    gulp=args.gulp,
    inf_dict=inf_dict,
    maxDT=int(maxDT),  # otherwise numpy int
)

if weird_last_gulp:
    yaml_dict['gulp'] = args.gulp - 1
    yaml_dict['last_gulp'] = int(nsamples % args.gulp)

logging.debug("\nDict values to go into every yaml file:")
logging.debug(f"{yaml_dict})

# loop through each split file and write a yaml for each
inf_names = [f"{args.filename[:-4]}_DM{aDM:.{args.dmprec}f}.inf" for aDM in DMs]
for ii in fouts_indices:
    specific_yaml_dict = copy.copy(yaml_dict)

    slc = dm_slices[ii]
    specific_yaml_dict["medians"] = [float(md) for md in meds[slc]]
    specific_yaml_dict["inf_names"] = inf_names[slc]
    specific_yaml_dict["DMs"] = [float(aDM) for aDM in DMs[slc]]

    # write yaml
    with open(f"{fouts_names[i]}.yaml", "w") as fyaml:
        yaml.dump(specific_yaml_dict, fyaml)
    logging.info(f"yaml written to {fouts_names[ii]}.yaml")

t4 = time.perf_counter()
logging.info(f"yamls written in {t4 - t3} seconds")

logging.info(f"\nDone")
