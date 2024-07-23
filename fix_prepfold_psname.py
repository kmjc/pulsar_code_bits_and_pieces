import argparse
import glob
import os
from copy import deepcopy

def rename_ps(dirname, verbose=True):
    if verbose:
        print(f"# Renaming {dirname}/*.ps files if they were too long")
    pfds = sorted(glob.glob(os.path.join(dirname, "*.pfd")))
    allfnames = sorted(glob.glob(os.path.join(dirname, "*")))
    pss_existing = sorted(glob.glob(os.path.join(dirname, "*.ps")))
    pss_expect = [f"{pfd[:-3]}.ps" for pfd in pfds]
    weird_files = [f for f in allfnames if (f[-4:] != ".pfd" and f[-9:] != ".bestprof" and f[-3:] != ".ps")]

    rename = {}
    for i, pfd in reversed(list(enumerate(pfds))):
        if pss_expect[i] in pss_existing:
            del pfds[i]
            continue

        found = False
        for j, weird in reversed(list(enumerate(weird_files))):
            if weird in pss_expect[i]:
                if found:
                    raise RuntimeError(f"Duplicate match {pfd} : {weird}, existing match: {rename[pfd]})")
                rename[pfd] = weird
                del pfds[i]
                del weird_files[j]
                found = True
    if len(pfds):
        print(f"WARNING: unmatched pfds: {pfds}")

    for pfd, weird in rename.items():
        new_name = f"{pfd[:-3]}ps"
        if verbose:
            print(f"Moving {weird} to {new_name}")
        os.rename(weird, new_name)

    if verbose:
        print("DONE")

if __name__ == "__main__":
    all_tasks = ["ps_rename", "quickclf", "quickclf_parse", "plot_ffa", "plot_singlepulse", "cleanup", "tar"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Fix any overlong prepfold ps filenames where the ps etc got cut off (assumes the corresponding pfd exists)",
    )
    parser.add_argument(
        "-d",
        "--dirname",
        default=".",
        help="Directory in which to look for pfd and ps files"
    )

    args = parser.parse_args()

    rename_ps(args.dirname)

# OK so it turns out this is super annoying because the character limit is based on the dir prepfold was run in
