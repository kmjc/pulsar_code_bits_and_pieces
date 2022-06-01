import sys
from presto_without_presto.infodata import infodata2
from presto_without_presto.psr_utils import choose_N
import yaml
import argparse
import copy
import numpy as np
import time
import logging
import os
from gen_utils import handle_exception


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Go from output of chunk_fdmt_fb2bin.py to presto-style .dat and .inf files",
)

parser.add_argument(
    "filename",
    type=str,
    help=".fdmt file to process. Must have a corresponding .fdmt.yaml file",
)

parser.add_argument(
    "--pad",
    action="store_true",
    help="Pad the .dat files to a highly factorable length using their mean (untested)",
)

parser.add_argument(
    "--log", type=str, help="name of file to write log to", default=None
)

parser.add_argument(
    "-v",
    "--verbose",
    help="Increase logging level to debug",
    action="store_const",
    dest="loglevel",
    const=logging.DEBUG,
    default=logging.INFO,
)


args = parser.parse_args()

if args.log is not None:
    logging.basicConfig(
        filename=args.log,
        filemode="w",
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=args.loglevel,
    )
else:
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=args.loglevel,
        stream=sys.stdout,
    )

logging.info(f"Processing {args.filename}")

with open(f"{args.filename}.yaml", "r") as fin:
    yam = yaml.safe_load(fin)
logging.info(f"yaml file {args.filename}.yaml loaded:")
for key in yam.keys():
    logging.debug(f"{key:}")
    logging.debug(f"{yam[key]}")

# write dat files
dat_names = [f"{infnm[:-4]}.dat" for infnm in yam["inf_names"]]
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
gulp = yam["gulp"]
maxDT = yam["maxDT"]
ndms = len(yam["DMs"])
dm_indices = range(ndms)  # if want to grab a specific DM just change dm_indices
ngulps = yam["ngulps"]  # does NOT include last gulp if it's a weird length


last_gulp = yam.get("last_gulp", 0)
if last_gulp:
    logging.debug("Weird last gulp detected")
    # ngulps -= 1  # already do this in chunk_fdmt_fb2bin BUT . .  it is super clunky. this variable changes too much

logging.info(
    f"maxDT: {maxDT}, ndms: {ndms}, gulp: {gulp}, ngulps: {ngulps}, last_gulp: {last_gulp}\n"
)

# Make unravel_plan
logging.info("Unravel plan:")
unravel_plan = [gulp - maxDT]
for i in range(1, ngulps):
    unravel_plan.extend([maxDT, gulp - maxDT])
if last_gulp:
    unravel_plan.extend([maxDT, last_gulp - maxDT])
logging.info(unravel_plan)
logging.info("\n")


# open all dats, loop through file writing to each
logging.debug("BENCHMARKING: keep all dats open, loop through file")
t2 = time.perf_counter()

datfiles = [open(dat_name, "wb") for dat_name in dat_names]

fdmtfile = open(args.filename, "rb")

# first gulp, (gulp - maxDT) time samples
# samples_processed = 0

running_sum = np.zeros(ndms)

# I'd prefer to make unravel_plan in chunk_fdmt_fb2bin as you write the data
# since it's more likely to be accurate then, but with lots of gulps it takes too long
logging.info("Unravelling data")
for chunk in unravel_plan:
    for i in dm_indices:
        dmdata = np.fromfile(fdmtfile, count=chunk, dtype=dt)
        running_sum[i] += dmdata.sum()
        datfiles[i].write(dmdata)
logging.info("Done\n")

# Pad dat files
# UNTESTED
if args.pad:
    logging.info("Padding data")
    origNdat = yam["origNdat"]
    N = choose_N(origNdat)
    logging.debug(f"Data will be padded from {origNdat} to {N} samples")

    onoff = [(0, origNdat - 1), (N - 1, N - 1)]
    padby = onoff[1][0] - onoff[0][1]

    means = running_sum / origNdat
    for i in dm_indices:
        logging.debug(f"Padding DM {i} by {padby} with {means[i]}")
        padding = np.zeros((padby), dtype=dt) + means[i]
        datfiles[i].write(padding)

    # update inf dict
    yam["inf_dict"]["breaks"] = 1
    yam["inf_dict"]["onoff"] = onoff
    yam["inf_dict"]["N"] = N
    logging.debug(
        f"updated inf dict to N: {yam['inf_dict']['N']}, breaks:{yam['inf_dict']['breaks']}, onoff: {yam['inf_dict']['onoff']}"
    )

logging.debug("Closing fdmt file")
fdmtfile.close()
logging.debug("fdmt file closed")

logging.debug("Closing dat files file")
for datfile in datfiles:
    datfile.close()
logging.debug("dat files closed")

# check number of samples wrote
sz = os.path.getsize(dat_names[0]) / 4
logging.info(f"dat files each contain {sz} samples")
for dat_name in dat_names:
    sz_i = os.path.getsize(dat_name) / 4
    if sz_i != sz:
        logging.warning(f"{dat_name} has a different number of samples! ({sz_i})")


t3 = time.perf_counter()
logging.debug(
    f"BENCHMARKING: keep all dats open, loop through file - {t3 - t2} seconds for {ndms} dats"
)

# check number of samples wrote matches the inf file value
if yam["inf_dict"]["N"] != sz:
    logging.warning(
        f"inf N ({yam['inf_dict']['N']}) does not match size of dat files ({sz})"
    )

t8 = time.perf_counter()
# write .inf files
logging.info("\nWriting inf files")
for i in dm_indices:
    specific_infdict = copy.copy(yam["inf_dict"])
    specific_infdict["DM"] = yam["DMs"][i]
    inf = infodata2(specific_infdict)
    inf.to_file(yam["inf_names"][i], notes="fdmt")
    logging.info(f"Wrote {yam['inf_names'][i]}")

t9 = time.perf_counter()
logging.debug(f"\nWrote all .inf files in {t9-t8} s")

sys.exit()
