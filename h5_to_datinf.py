import h5py
import os, sys
import copy
import numpy as np
import argparse
from presto_without_presto.psr_utils import choose_N
from presto_without_presto.sigproc import (ids_to_telescope, ids_to_machine)
from presto_without_presto.infodata import infodata2


def radec2string(radec):
    """Convert the SIGPROC-style HHMMSS.SSSS right ascension
    to a presto-inf-style HH:MM:SS.SSSS string

    or similarly for declination, DDMMSS.SSSS -> DD.MM.SS.SS"""
    hh = int(radec // 10000)
    mm = int((radec - 10000*hh) // 100)
    ss = int((radec - 10000*hh -100*mm) // 1)
    ssss = int(((radec - 10000*hh -100*mm - ss) * 10000) // 1)
    return f"{hh}:{mm}:{ss}.{ssss}"

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="""Convert a h5 output from FDMT (as per chunk_fdmt_fb2h5.py) to
    presto .dat and .inf files"""
)

parser.add_argument("fname", type=str, help="h5 filename containing data (with shape (nDMs, time)), header and DMs")
parser.add_argument("--pad", action='store_true', help="Whether to pad the data to a highly factorable length (for FFT-ing)")
parser.add_argument("--dmprec", type=int, default=2, help="DM precision (for filenames)")
parser.add_argument("-v", "--verbosity", action="count", default=0, help="Verbose output")

args = parser.parse_args()
fname = args.fname
padding = args.pad
dmprec = args.dmprec



if args.verbosity > 0:
    def verbose_message(message):
        print(message)
else:
    def verbose_message(message):
        pass

if args.verbosity > 1:
    def verbose_message1(message):
        print(message)
else:
    def verbose_message1(message):
        pass


f = h5py.File(fname, 'r')

# Extract header info
hdict = {}
for key in f['header'].keys():
    hdict[key] = f['header'][key][()]

# calculate lofreq
if hdict['foff'] < 0:  # normal sigproc format
    lofreq = hdict['fch1'] + (hdict['nchans']-1)*hdict['foff']
else:
    lofreq = hdict['fch1']


# Base inf dict that applies to all DMs
infdict = dict(
    basenm=fname[:-3],
    telescope=ids_to_telescope[hdict['telescope_id']],
    instrument=ids_to_machine[hdict['machine_id']],
    object=hdict.get('source_name', 'Unknown'),
    RA=radec2string(hdict['src_raj']),
    DEC=radec2string(hdict['src_dej']),
    observer='unset',
    epoch= hdict['tstart'],
    bary=0,
    dt=hdict['tsamp'],
    lofreq=lofreq,
    BW=abs(hdict['nchans'] * hdict['foff']),
    numchan=hdict['nchans'],
    chan_width=abs(hdict['foff']),
    analyzer=os.environ.get( "USER" ),
)

if padding:
    # find good N
    origN = f['data'].shape[1]
    N = choose_N(origN)
    infdict['N'] = N
    infdict['breaks'] = 1
    infdict['onoff'] = [(0, origN - 1), (N - 1, N - 1)]
else:
    N = f['data'].shape[1]
    infdict['N'] = N
    infdict['breaks'] = 0


# loop through DMs and write out .dat and .inf files
dms = f['DMs'][()]
data = f['data']
for i in range(dms.shape[0]):
    specific_infdict = copy.copy(infdict)
    specific_infdict['DM'] = dms[i]
    inf = infodata2(specific_infdict)
    outbasename = f"{fname[:-3]}_{round(dms[i], dmprec)}"
    inf.to_file(f"{outbasename}.inf", notes="fdmt")
    verbose_message(f"Wrote {outbasename}.inf")

    if padding:
        verbose_message1('padding')
        outdata = np.zeros((N), dtype=np.float32)
        outdata[:origN] = data[i, :]
        verbose_message1('data read')
        # pad data with median of middle 80%
        outdata[origN:] = np.median(data[i, int(0.1*origN):int(0.9*origN)])
        verbose_message1('padded values set to median')
    else:
        outdata = data[i, :]
        verbose_message1('data read')

    with open(f"{outbasename}.dat", "wb") as fout:
        fout.write(outdata)
    verbose_message(f"Wrote {outbasename}.dat")

sys.exit()
