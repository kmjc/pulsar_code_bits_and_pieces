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

last_gulp = yam.get("last_gulp", 0)
if last_gulp:
    ngulps -= 1
has_breaks = yam.get("breaks", 0)


# loop through file for each DM and write one dat file at a time
print("BENCHMARKING: loop through file for each dat")
t0 = time.perf_counter()

fdmtfile = open(args.filename, "rb")
first_gulp_offset = maxDT*(gulp-maxDT)
for i in range(10):
    datfile = open(dat_names[i], "wb")
    # first chunk is  gulp-maxDT
    offset = i * (gulp-maxDT)
    fdmtfile.seek(offset*nbytes)
    dmdata = np.fromfile(fdmtfile, count=(gulp-maxDT), dtype=dt)
    datfile.write(dmdata)

    dm_offset = i*gulp
    fdmtfile.seek((first_gulp_offset + dm_offset)*nbytes)

    for g in range(1, ngulps):
        dmdata = np.fromfile(fdmtfile, count=gulp, dtype=dt)
        datfile.write(dmdata)
        fdmtfile.seek((maxDT - 1)*gulp*nbytes, 1)

    # last gulp is weird size

# skipping last gulp and padding for now. NB if not writing dats as do fdmt should just write the medians to the yaml and apply them here

#    if last_gulp:
#        last_gulp_offset = gulp*maxDT*ngulps + i*last_gulp
#        fdmtfile.seek(last_gulp_offset*nbytes)
#        dmdata = np.fromfile(fdmtfile, count=yam['last_gulp'], dtype=dt)
#        datfile.write(dmdata)
#        padding_offset = gulp*maxDT*ngulps + maxDT*last_gulp
#    else:
#        padding_offset = 0

    # file has been padded
#    if has_breaks:
#        onoff = yam['inf_dict']['onoff']
#        padding = onoff[1][0] - onoff[0][1]
#        padding_offset += i * padding
#        fdmtfile.seek(padding_offset*nbytes)
#        dmdata = np.fromfile(fdmtfile, count=padding, dtype=dt)
#        datfile.write(dmdata)

    datfile.close()
    print(f"DM {i} done")


fdmtfile.close()
t1 = time.perf_counter()
print(f"BENCHMARKING: loop through file for each dat - {t1 - t0} seconds for 10 dats")
print(f"Extrapolate to 5000 dat files => {t1-t0} x 5000 / 10 => {(t1-t0)* 5000/10 /60/60} hrs")



# do same but read in whole chunk rather than read-seek-ing
# DEFINITELY SLOWER!


print("\nBENCHMARKING: loop through file once, read whole gulp, open-write-close all dats every gulp")
t4 = time.perf_counter()

fdmtfile = open(args.filename, "rb")
# first chunk is  gulp-maxDT
dmdata = np.fromfile(fdmtfile, count=(gulp-maxDT)*maxDT, dtype=dt).reshape((maxDT, gulp-maxDT))
for i in range(10):
    with open(dat_names[i], "wb") as datfile:
        datfile.write(dmdata[i,:])


for g in range(1, ngulps):
    dmdata = np.fromfile(fdmtfile, count=gulp*maxDT, dtype=dt).reshape((maxDT, gulp))
    for i in range(10):
        with open(dat_names[i], "ab") as datfile:
            datfile.write(dmdata[i,:])

    print(f" gulp {g} done")

fdmtfile.close()

t5 = time.perf_counter()
print(f"BENCHMARKING: loop through file once, read whole gulp, open-write-close all dats every gulp - {t5 - t4} seconds for 10 dats")

"""
# should do this with seeks
print(f"\nBENCHMARKING: keep 10 files open at once, loop through filterbank")
t6 = time.perf_counter
datfiles = [open(dat_names[i], "wb") for i in range(10)]
fdmtfile = open(args.filename, "rb")

# first chunk
dmdata = np.fromfile(fdmtfile, count=(gulp-maxDT)*maxDT, dtype=dt).reshape((maxDT, gulp-maxDT))
for i in range(10):
    datfile[i].write(dmdate[i,:])

for g in range(1, ngulps):
    dmdata = np.fromfile(fdmtfile, count=gulp*maxDT, dtype=dt).reshape((maxDT, gulp))
    for i in range(10):
        datfile[i].write(dmdata[i,:])
    print(f"gulp {g} done")

for i in range(10):
    datfiles[i].close()
fdmtfile.close()

t7 = time.perf_counter()
print(f"BENCHMARKING: keep 10 files open at once, loop through filterbank - {t7-t6} seconds for 10 dats")
print(f"Extraplate to 5000 dat files => {t7-t6} * 5000 / 10 => {(t7-t6)*5000/10/60/60} hrs")
print("NB this one might get more efficient if keep more files open at once")
"""

#t8 = time.perf_counter()
# write .inf files
#verbose_message0("\nWriting inf files")
#for i in range(maxDT):
#    specific_dict = copy.copy(yam['inf_dict'])
#    specific_infdict['DM'] = yam['DMs'][i]
#    inf = infodata2(specific_infdict)
#    inf.to_file(yam['inf_names'][i], notes="fdmt")
#    verbose_message0(f"Wrote {yam['inf_names'][i]}")

#t9 = time.perf_counter()
#print(f"\nWrote all .inf files in {t9-t8} s")

sys.exit()
