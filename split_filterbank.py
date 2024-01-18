import numpy as np
from presto_without_presto import sigproc
from sigproc_utils import get_dtype, write_header
import argparse
import copy
import sys


def split_fil(fname, nfiles):
    header, hdrlen = sigproc.read_header(fname)
    tsamp = header['tsamp']
    nchans = header['nchans']
    arr_dtype = get_dtype(header["nbits"])
    nsamp = int(sigproc.samples_per_file(fname, header, hdrlen))

    gulp_init = nsamp // nfiles
    gulps = [gulp_init] * nfiles
    gulps[-1] += nsamp % nfiles

    filfile = open(fname, "rb")
    filfile.seek(hdrlen)

    for i, gulp in enumerate(gulps):
        out_fname = f"{fname.rstrip('.fil')}_{i}.fil"
        header_out = copy.deepcopy(header)
        header_out['tstart'] = header['tstart'] + sum(gulps[:i+1]) * tsamp / 60 / 60 / 24

        outf = open(out_fname, "wb")
        write_header(header_out, outf)

        intensities = np.fromfile(filfile, count=gulp*nchans, dtype=arr_dtype)
        outf.write(intensities.ravel().astype(arr_dtype))

        outf.close()
        print(f"Wrote {out_fname}")

    filfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Split a filterbank into multiple files",
    )

    parser.add_argument(
        "fname",
        type=str,
        help="filterbank file to process",
    )

    parser.add_argument(
        "nfiles",
        type=int,
        help="Split filterbank into <nfiles> files",
    )

    args = parser.parse_args()

    split_fil(args.fname, args.nfiles)
    sys.exit(0)

