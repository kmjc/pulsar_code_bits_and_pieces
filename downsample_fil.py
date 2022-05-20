from sigpyproc.readers import FilReader
import argparse
import sys

def downsample_fil(fil_fname, factors, gulp=2400):
    print("Starting on", fil_fname)
    fil = FilReader(fil_fname)
    if not isinstance(factors, list):
        factors = list(factors)
    for factor in factors:
        print(f"Downsampling {fil_fname} by {factor} (in time)")
        fil.downsample(tfactor=factor, filename=f"{fil_fname[:-4]}_t{factor}.fil", gulp=gulp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample a file (using sigpyproc3)"
        )

    parser.add_argument("filename", type=str, help="Sigproc filterbank file")
    parser.add_argument("downsamp", type=int, nargs="+", help="Sownsampling factor/s to apply (multiple factors should be separated by spaces)")
    parser.add_argument("--gulp", type=int, default=2400, help="Maximum number of samples to read at one time")

    args = parser.parse_args()

    downsample_fil(args.filename, args.downsamp, gulp=args.gulp)

    sys.exit()
