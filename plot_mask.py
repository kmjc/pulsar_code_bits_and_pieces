import numpy as np
import argparse
from matplotlib import pyplot as plt
from presto_without_presto import rfifind

def array_from_mask_params(nint, nchan, zap_ints, zap_chans, zap_chans_per_int):
    """Return the mask as a numpy array of size (nint, nchan)
    (1 = masked, 0 = unmasked)

    nint = number of intervals
    nchan = number of channels
    zap_ints (set) - intervals to zap
    zap_chans (set) - channels to zap
    zap_chans_per_int - list (of length nint) of sets such that
                        zap_chans_per_int[i] gives a set of channels to mask in interval i

    NB if using original rfifind.py note that some type conversion will be needed as
    mask_zap_ints is an array and mask_zap_chans_per_int is a list of arrays
    """
    mask = np.zeros((nint, nchan), dtype=bool)
    if len(zap_ints) != 0:
        mask[tuple(zap_ints), :] = 1
    if len(zap_chans) != 0:
        mask[:, tuple(zap_chans)] = 1
    for i in range(nint):
        if len(zap_chans_per_int[i]) != 0:
            mask[i, tuple(zap_chans_per_int[i])] = 1

    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot a rfifind mask")
    parser.add_argument("maskfile", type=str, help="rfifind .mask file")
    parser.add_argument("-c", "--compare", type=str, help="compare to this other mask")
    args = parser.parse_args()

    rfimask = rfifind.rfifind(args.maskfile)
    rfimaskarr = array_from_mask_params(
        rfimask.nint,
        rfimask.nchan,
        rfimask.mask_zap_ints,
        rfimask.mask_zap_chans,
        rfimask.mask_zap_chans_per_int,
    )

    if args.compare is None:
        im = plt.pcolormesh(rfimaskarr)
        plt.colorbar(im)
        plt.title(f"{rfimask.basename} (1 = masked)")
    else:
        rfimask2 = rfifind.rfifind(args.compare)
        rfimaskarr2 = array_from_mask_params(
            rfimask2.nint,
            rfimask2.nchan,
            rfimask2.mask_zap_ints,
            rfimask2.mask_zap_chans,
            rfimask2.mask_zap_chans_per_int,
        )

        if rfimaskarr.shape != rfimaskarr2.shape:
            raise AttributeError(f"Masks have different shapes (mask has {rfimaskarr.shape}, compare has {rfimaskarr2.shape})")

        comparearr = np.zeros(rfimaskarr.shape, dtype=float)
        comparearr[(rfimaskarr == True) & (rfimaskarr2 == True)] = 0
        comparearr[(rfimaskarr == True) & (rfimaskarr2 == False)] = +1
        comparearr[(rfimaskarr == False) & (rfimaskarr2 == True)] = -1
        comparearr[(rfimaskarr == False) & (rfimaskarr2 == False)] = np.nan

        im = plt.pcolormesh(comparearr, vmin=-1, vmax=1, cmap='cool')
        plt.colorbar(im)
        plt.title(f"A ({rfimask.basename}) compared with B({rfimask2.basename}) \n"
                  f"0 = both masked, +1 = A masked & B unmasked, -1 = A unmasked & B masked")

    plt.xlabel("channel")
    plt.ylabel("interval")

    plt.show()
