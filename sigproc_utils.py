from presto_without_presto import sigproc
import numpy as np
import logging


def radec2string(radec):
    """Convert the SIGPROC-style HHMMSS.SSSS right ascension
    to a presto-inf-style HH:MM:SS.SSSS string

    or similarly for declination, DDMMSS.SSSS -> DD.MM.SS.SS"""
    hh = int(radec // 10000)
    mm = int((radec - 10000 * hh) // 100)
    ss = int((radec - 10000 * hh - 100 * mm) // 1)
    ssss = int(((radec - 10000 * hh - 100 * mm - ss) * 10000) // 1)
    return f"{hh}:{mm}:{ss}.{ssss}"


# presto's filterbank has a get_dtype could use too, but haven't without_presto-ed that yet
def get_dtype(nbits):
    """
    Returns:
        dtype of the data
    """
    if nbits == 8:
        return np.uint8
    elif nbits == 16:
        return np.uint16
    elif nbits == 32:
        return np.float32
    else:
        raise RuntimeError(f"nbits={nbits} not supported")


def get_nbits(dtype):
    """
    Returns:
        number of bits of the data
    """
    if dtype == np.uint8:
        return 8
    elif dtype == np.uint16:
        return 16
    elif dtype == np.float32:
        return 32
    else:
        raise RuntimeError(f"dtype={dtype} not supported")


# OK so if it's a filterbank it SHOULD have HEADER_START and HEADER_END
# I think I was testing it on one that wasn't properly written, so I had to
# add them in
def write_header(header, outfile):
    header_list = list(header.keys())
    manual_head_start_end = False
    if header_list[0] != "HEADER_START" or header_list[-1] != "HEADER_END":
        logging.debug(
            f"HEADER_START not first and/or HEADER_END not last in header_list, removing them from header_list (if present) and writing them manually"
        )
        try_remove("HEADER_START", header_list)
        try_remove("HEADER_END", header_list)
        manual_head_start_end = True

    if manual_head_start_end:
        outfile.write(sigproc.addto_hdr("HEADER_START", None))
    for paramname in header_list:
        if paramname not in sigproc.header_params:
            # Only add recognized parameters
            continue
        logging.debug("Writing header param (%s)" % paramname)
        value = header[paramname]
        outfile.write(sigproc.addto_hdr(paramname, value))
    if manual_head_start_end:
        outfile.write(sigproc.addto_hdr("HEADER_END", None))


def get_fmin_fmax_invert(header):
    """Calculate band edges and whether the band is inverted from a filterbank header.
    The presto sense of inverted, aka for a normal sigproc file where foff<0 inverted=True
    Returns fmin, fmax, inverted"""

    if header["foff"] < 0:
        fmax = header["fch1"] - header["foff"] / 2
        fmin = fmax + header["nchans"] * header["foff"]
        invert = True
    else:
        fmin = header["fch1"] - header["foff"] / 2
        fmax = fmin + header["nchans"] * header["foff"]
        invert = False

    return fmin, fmax, invert
