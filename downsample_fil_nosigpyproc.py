import numpy as np
from presto_without_presto import sigproc
from sigproc_utils import get_dtype, write_header
import argparse
import copy
from math import ceil
import sys

def downsample_fil(fname, downsamps, gulp):
    header, hdrlen = sigproc.read_header(fname)
    tsamp = header['tsamp']
    nchans = header['nchans']
    arr_dtype = get_dtype(header["nbits"])
    nsamp = int(sigproc.samples_per_file(fname, header, hdrlen))

    out_filenames = []
    outfs = []
    for d, downsamp in enumerate(downsamps):
        # check downsamp is compatible with gulp
        if gulp % downsamp:
            for outf in outfs:
                outf.close()
            suggest = 1
            for dd in downsamps:
                suggest *= dd
            raise RuntimeError(f"gulp ({gulp}) is not divisible by downsample factor {downsamp}. This will do weird things. Pick a gulp divisible by {suggest}.")

        # open file
        out_filenames.append(f"{fname[:-4]}_t{downsamp}.fil")
        outfs.append(open(out_filenames[d], "wb"))

        # write header
        new_header = copy.deepcopy(header)
        new_header["tsamp"] = tsamp*downsamp
        write_header(new_header, outfs[d])

    filfile = open(fname, "rb")
    filfile.seek(hdrlen)

    ngulps = ceil(nsamp/gulp)
    print(f"Processing {fname} in {ngulps} gulps of {gulp}")
    for g in range(ngulps):
        intensities = np.fromfile(filfile, count=gulp*nchans, dtype=arr_dtype).reshape(-1, nchans)
        for d, downsamp in enumerate(downsamps):
            outfs[d].write(intensities[::downsamp,:].ravel().astype(arr_dtype))
        print(f"Wrote gulp {g}/{ngulps}")

    for i, outf in enumerate(outfs):
        outf.close()
        print(f"Wrote downsampled file {out_filenames[i]}")

    filfile.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample a file (using sigpyproc3)")

    parser.add_argument("filename", type=str, help="Sigproc filterbank file")
    parser.add_argument(
        "downsamp",
        type=int,
        nargs="+",
        help="Downsampling factor/s to apply (multiple factors should be separated by spaces)",
    )
    parser.add_argument(
        "--gulp",
        type=int,
        default=28800,
        help="Maximum number of time samples to read at once",
    )

    args = parser.parse_args()

    downsample_fil(args.filename, args.downsamp, gulp=args.gulp)

    sys.exit()