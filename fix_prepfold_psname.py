import argparse
import glob
import os

def rename_ps(dirname, verbose=True):
    print(f"# Renaming {dirname}/*.ps files if they were too long")
    pfds = sorted(glob.glob(os.path.join(dirname, "*.pfd")))
    allfnames = sorted(glob.glob(os.path.join(dirname, "*")))

    for i, pfd in enumerate(pfds):
        basename = os.path.basename(pfd)
        if len(f"{basename}.ps") > 73:
            new_name = f"{pfd}.ps"
            old_name = os.path.join(os.path.dirname(new_name), os.path.basename(new_name)[:73])
            # check it exists
            if new_name in allfnames:
                if verbose:
                    print(f"{new_name} already exists")
                continue
            if old_name not in allfnames:
                raise RuntimeWarning(f"Expect {old_name} to exist, but it doesn't")
            if verbose:
                print(f"Moving {old_name} to {new_name}")
            os.rename(old_name, new_name)

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