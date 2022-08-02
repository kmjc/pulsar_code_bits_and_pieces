# Adam's fdp.py script
# Edits:
#     argparse
#     use shape broadcasting to cut down on memory useage
#     don't use presto - no forced floats and hopefully speedier as less back and forth

import numpy as np
from presto_without_presto import rfifind, sigproc
from sigproc_utils import get_dtype, get_nbits, write_header
import logging
import argparse

#!/usr/bin/env python3
from presto import filterbank as fb
import sys
import numpy as np
import logging
import matplotlib.pyplot as plt
from presto import filterbank as fb
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="""Correct artefacts due to dropped packets in CHIME data.
    These show up as sudden dips (but not necessarily to 0) in 4 channels at the same time.
    This script
        - reads in a chunk of data,
        - calculates the median and standard deviation for each frequency channel,
        - replaces any values less than (median - <thresh_sig> * std) with the median
        - repeat
    A new file ending in _fdp.fil will be writted

    Note:
    There's a hidden forced conversion of the data into floats in the get_spectra
    method of presto.filterbank.FilterbankFile. If the script is using more memory
    than it seems like it should from the <chunk_size> this is likely the reason.
    """,
)

parser.add_argument(
    "fn",
    type=str,
    help="filterbank file to process",
)

parser.add_argument(
    "--chunk_size",
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

args = parser.parse_args()

log = logging.getLogger(__name__)

header, hdrlen = sigproc.read_header(args.fn)
nspecs = int(sigproc.samples_per_file(args.fn, header, hdrlen))
nchans = header["nchans"]
arr_dtype = get_dtype(header["nbits"])

if header["nifs"] != 1:
    raise log.error(f"Code not written to deal with unsummed polarization data")


#loop through chunks
loop_iters = int(nspecs/args.chunk_size)
fn_clean = args.fn.strip('.fil')
fdp_fn = f"{fn_clean}_fdp2.fil"
new_fil = open(os.path.join(fdp_fn), "wb")
write_header(header, new_fil)
fil = open(args.fn, "rb")
fil.seek(hdrlen)

for i in range(loop_iters):
    print(f"{i}/{loop_iters}\r")
    spec = (
        np.fromfile(fil, count=args.chunk_size*nchans, dtype=arr_dtype)
        .reshape(-1, nchans)
    )
    # has shape (nspec, nchans) so it plays nice with brodcasting
    med = np.median(spec,axis=0)
    std = np.std(spec,axis=0)
    #set the thresold
    threshold = med - args.thresh_sig*std
    #find values below threshold and replace with the median
    mask = np.where(spec < threshold)
    spec[mask] = med[mask[1]]
    new_fil.write(spec.ravel().astype(arr_dtype))

fil.close()
new_fil.close()
