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

    for i, gulp in enumerate(gulps, subgulp=100000):
        out_fname = f"{fname.rstrip('.fil')}_{i}.fil"
        header_out = copy.deepcopy(header)
        header_out['tstart'] = header['tstart'] + sum(gulps[:i+1]) * tsamp / 60 / 60 / 24

        outf = open(out_fname, "wb")
        write_header(header_out, outf)

        subgulps = [subgulp]*int(gulp//subgulp)
        if gulp % subgulp:
            subgulps.append(int(gulp % subgulp))

        for subgulpp in subgulps:
            intensities = np.fromfile(filfile, count=subgulpp*nchans, dtype=arr_dtype)
            outf.write(intensities.ravel().astype(arr_dtype))

        outf.close()
        print(f"Wrote {out_fname}")
        del intensities

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

    parser.add_argument(
        "--subgulp",
        default=100000,
        help="how many time samples to read at once (100000 with 1024 channels and uint8 => ~100 MB)"
    )

    args = parser.parse_args()

    split_fil(args.fname, args.nfiles, subgulp=args.subgulp)
    sys.exit(0)

