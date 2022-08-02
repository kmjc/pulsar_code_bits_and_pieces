import glob
import os
from presto_without_presto import sigproc
from presto_without_presto.psr_utils import choose_N

def get_common_N(dir, glob_expr="*_pow.fil"):
    """Get a common "nice" N for all files in <dir> matching <glob_expr>"""
    glob_expr_full = os.path.join(dir, glob_expr)
    raw_fils = glob.glob(glob_expr_full)

    if not raw_fils:
        raise AttributeError(f"No {glob_expr_full} filterbank files found")

    max_N = 0
    for fil in raw_fils:
        header, hdrlen = sigproc.read_header(fil)
        nsamples = int(sigproc.samples_per_file(fil, header, hdrlen))
        if nsamples > max_N:
            max_N = nsamples

    out_N = choose_N(max_N)
    return out_N

if __name__ == "__main__":
    dir = os.getcwd()
    get_common_N(dir)
