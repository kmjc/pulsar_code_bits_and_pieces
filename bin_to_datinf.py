from presto_without_presto.infodata import infodata2
import yaml
import argparse
import copy
import numpy as np
import time
import sys



parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Go from output of chunk_fdmt_fb2bin.py to presto-style .dat and .inf files"
)
parser.add_argument("filename", type=str, help=".fdmt file to process. Must have a corresponding .fdmt.yaml file")
args = parser.parse_args()

def verbose_message0(message):
    print(message)

with open(f"{args.filename}.yaml", "r") as fin:
    yam = yaml.safe_load(fin)

# write dat files
dat_names = [f"{infnm[-4]}.dat" for infnm in yam['inf_names']]
# quicker to loop through and write one by one or write same chunk for all?
fdmtfile = open(args.filename, "rb")

# data will be in order
# dm0t0 dm0t1 dm0t2 .... dm1t0 dm1t1
# where dm0t0 = 0th time interval for the 0th DM
# for each gulp

# gulp = number of time intervals
# so starting position of dm i is
# which_gulp_on * gulp * maxDT + i * gulp

dt = np.float32
nbytes = 4  # data should be 32 bit floats
gulp = yam['gulp']
maxDT = yam['maxDT']
ngulps = yam['ngulps']

t0 = time.perf_counter()
# loop through file for each DM and write one dat file at a time
for i in range(maxDT):
    t1 = time.perf_counter()
    print(f"Writing {i}th DM, {yam['DMs'][i]}")
    datfile = open(dat_names[i], "wb")
    offset = i * gulp
    fdmtfile.seek(offset*nbytes)
    for g in range(ngulps):
        dmdata = np.fromfile(fdmtfile, count=gulp, dtype=dt)
        datfile.write(dmdata)
        fdmtfile.seek((maxDT - 1)*gulp*nbytes, 1)

    # last gulp is weird size
    last_gulp = yam.get("last_gulp", 0)
    if last_gulp:
        last_gulp_offset = gulp*maxDT*ngulps + i*last_gulp
        fdmtfile.seek(last_gulp_offset*nbytes)
        dmdata = np.fromfile(fdmtfile, count=yam['last_gulp'], dtype=dt)
        datfile.write(dmdata)
        padding_offset = gulp*maxDT*ngulps + maxDT*last_gulp
    else:
        padding_offset = 0

    # file has been padded
    if yam.get("breaks", 0):
        onoff = yam['inf_dict']['onoff']
        padding = onoff[1][0] - onoff[0][1]
        padding_offset += i * padding
        fdmtfile.seek(padding_offset*nbytes)
        dmdata = np.fromfile(fdmtfile, count=padding, dtype=dt)
        datfile.write(dmdata)

    t2 = time.perf_counter()
    print(f"Wrote {dat_names[i]} in {t2-t1} s")

    datfile.close()

t3 = time.perf_counter()
print(f"\nWrote all .dat files in {t3-t0} s")



fdmtfile.close()

# write .inf files
verbose_message0("\nWriting inf files")
for i in range(maxDT):
    specific_dict = copy.copy(yam['inf_dict'])
    specific_infdict['DM'] = yam['DMs'][i]
    inf = infodata2(specific_infdict)
    inf.to_file(yam['inf_names'][i], notes="fdmt")
    verbose_message0(f"Wrote {yam['inf_names'][i]}")

t4 = time.perf_counter()
print(f"\nWrote all .inf files in {t4-t3} s")

sys.exit()
