from presto_without_presto.infodata import infodata2
import yaml
import argparse
import copy
import numpy as np
import time
import sys
import logging



parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Go from output of chunk_fdmt_fb2bin.py to presto-style .dat and .inf files"
)
parser.add_argument("filename", type=str, help=".fdmt file to process. Must have a corresponding .fdmt.yaml file")
parser.add_argument(
    "--log", type=str, help="name of file to write log to", default="bin_to_datinf.log"
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

logging.info(f"Processing {args.filename}")

with open(f"{args.filename}.yaml", "r") as fin:
    yam = yaml.safe_load(fin)
logging.info(f"yaml file {args.filename}.yaml loaded:")
for key in yam.keys():
    logging.debug(f"{key:}")
    logging.debug(f"{yam[key]}")

# write dat files
dat_names = [f"{infnm[:-4]}.dat" for infnm in yam['inf_names']]
logging.debug(f"List of dat files to write to:")
logging.debug(f"{dat_names}")

# data will be in order
# dm0t0 dm0t1 dm0t2 .... dm1t0 dm1t1
# where dm0t0 = 0th time interval for the 0th DM
# for each gulp

# gulp = number of time intervals
# so starting position of dm i is
# which_gulp_on * gulp * ndms + i * gulp

dt = np.float32
nbytes = 4  # data should be 32 bit floats
gulp = yam['gulp']
maxDT = yam['maxDT']
ndms = len(yam['DMs'])
dm_indices = range(ndms)  # if want to grab a specific DM just change dm_indices
ngulps = yam['ngulps']


last_gulp = yam.get("last_gulp", 0)
if last_gulp:
    logging.debug("Weird last gulp detected")
    #ngulps -= 1  # already do this in chunk_fdmt_fb2bin BUT . .  it is super clunky. this variable changes too much
has_breaks = yam['inf_dict'].get("breaks", 0)
if has_breaks:
    logging.debug("breaks found in inf_dict. Will pad dat files using medians in the yaml")

logging.info(f"maxDT: {maxDT}, ndms: {ndms}, gulp: {gulp}, ngulps: {ngulps}, last_gulp: {last_gulp}, has_breaks: {has_breaks}\n")


# open all dats, loop through file writing to each
logging.debug("BENCHMARKING: keep all dats open, loop through file")
t2 = time.perf_counter()

datfiles = [open(dat_name, "wb") for dat_name in dat_names]

fdmtfile = open(args.filename, "rb")

# first gulp, (gulp - maxDT) time samples
#samples_processed = 0

logging.info("Procesing gulp 0")
#logging.debug("GULP 0")
for i in dm_indices:
    dmdata = np.fromfile(fdmtfile, count=(gulp-maxDT), dtype=dt)
    datfiles[i].write(dmdata)
#    samples_processed += dmdata.size
#    logging.debug(f"\tDM {i}\tsamples processed: {samples_processed}")

# other gulps, (gulp) time samples
logging.info(f"Processing gulps {list(range(1, ngulps))[0]} to {list(range(1, ngulps))[-1]}")
for g in range(1, ngulps):
#    logging.debug(f"GULP {g}")
    for i in dm_indices:
        dmdata = np.fromfile(fdmtfile, count=gulp, dtype=dt)
        datfiles[i].write(dmdata)
#        samples_processed += dmdata.size
#        logging.debug(f"\tDM {i}\tsamples processed: {samples_processed}")
logging.info("Finished normal gulps")

# last gulp is weird size
if last_gulp:
    logging.debug("LAST GULP")
    for i in dm_indices:
        dmdata = np.fromfile(fdmtfile, count=yam['last_gulp'], dtype=dt)
        datfiles[i].write(dmdata)
        samples_processed += dmdata.size
        logging.debug(f"\tDM {i}\tsamples processed: {samples_processed}")

# file has been padded
# UNTESTED
if has_breaks:
    logging.info("Padding data")
    onoff = yam['inf_dict']['onoff']
    padby = onoff[1][0] - onoff[0][1]
    for i in dm_indices:
        med = yam['medians'][i]
        logging.debug(f"Padding DM {i} by {padby} with {med}")
        padding = np.zeros((padby), dtype=dt) + med
        datfiles[i].write(padding)

logging.debug("Closing fdmt file")
fdmtfile.close()
logging.debug("fdmt file closed")

logging.debug("Closing dat files file")
for datfile in datfiles:
    datfile.close()
logging.debug("dat files closed")


t3 = time.perf_counter()
logging.debug(f"BENCHMARKING: keep all dats open, loop through file - {t3 - t2} seconds for {len(dm_indices)} dats")



t8 = time.perf_counter()
# write .inf files
logging.info("\nWriting inf files")
for i in dm_indices:
    specific_infdict = copy.copy(yam['inf_dict'])
    specific_infdict['DM'] = yam['DMs'][i]
    inf = infodata2(specific_infdict)
    inf.to_file(yam['inf_names'][i], notes="fdmt")
    logging.info(f"Wrote {yam['inf_names'][i]}")

t9 = time.perf_counter()
logging.debug(f"\nWrote all .inf files in {t9-t8} s")

sys.exit()
