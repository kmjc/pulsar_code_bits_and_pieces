# ## Import stuff

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors
import copy
from collections.abc import Iterable

from presto_without_presto import rfifind
from iqrm import iqrm_mask
from rfifind_numpy_tools import write_new_mask_from

import sys
import argparse
from scipy.ndimage import generic_filter, generic_filter1d
from matplotlib.backends.backend_pdf import PdfPages

from gen_utils import handle_exception
import logging

# catch uncaught exceptions and put them in log too
sys.excepthook = handle_exception

# ## Define functions

# ### General utils

def output_plot(fig, pdf=None):
    if pdf is None:
        plt.show()
    else:
        fig.savefig(pdf, format='pdf')
    plt.close(fig)

def masked_frac(mask):
    return mask.sum()/mask.size

def get_ignorechans_from_mask(mask):
    nint, nchan = mask.shape
    return np.where(mask.sum(axis=0) == nint)[0]

def np_ignorechans_to_presto_string(array1d):
    return ",".join(list(array1d.astype(str)))

def write_mask_and_ignorechans(mask, outname, rfifind_obj, infstats_too=True):
    ignorechans_fname = f"{outname[:-5]}.ignorechans"
    logging.info(f"Writing ignorechans to {ignorechans_fname}")
    with open(ignorechans_fname, "w") as fignore:
        fignore.write(np_ignorechans_to_presto_string(get_ignorechans_from_mask(mask)))
    logging.info(f"Writing mask to {outname}")
    write_new_mask_from(outname, mask, rfifind_obj, infstats_too=infstats_too)

def wrap_up(mask, mask_exstats, rfifind_obj, means, var, pdf, outfilename, infstats_too):
    logging.info(f"Fraction of data masked: {masked_frac(mask)}")
    write_mask_and_ignorechans(mask, outfilename, rfifind_obj, infstats_too=infstats_too)

    logging.info(f"Making summary plots")
    make_summary_plots(mask, mask_exstats, rfifind_obj, means, var, pdf, title_insert="final")

    if pdf is not None:
        logging.info("Writing pdf")
        pdf.close()
    logging.info("Done")


def make_summary_plots(mask, mask_exstats, rfifind_obj, means, var, pdf, title_insert=""):
    """Plot the mask, and the masked pow_stats, means, and var"""
    figtmp, axtmp = plt.subplots()
    plot_mask(mask, ax=axtmp)
    figtmp.suptitle(f"{title_insert} mask")
    output_plot(figtmp, pdf=p)

    figtmp, axtmp = plot_map_plus_sums(rfifind_obj.pow_stats, mask=mask, returnplt=True)
    figtmp.suptitle(f"{title_insert} pow_stats")
    output_plot(figtmp, pdf=pdf)

    figtmp, axtmp = plot_map_plus_sums(means.data, mask=mask_exstats, returnplt=True)
    figtmp.suptitle(f"{title_insert} means")
    output_plot(figtmp, pdf=pdf)

    figtmp, axtmp = plot_map_plus_sums(var.data, mask=mask_exstats, returnplt=True)
    figtmp.suptitle(f"{title_insert} var")
    output_plot(figtmp, pdf=pdf)


# https://www.tutorialspoint.com/how-to-make-a-histogram-with-bins-of-equal-area-in-matplotlib
def equal_area(x, nbin):
   pow = 0.5
   dx = np.diff(np.sort(x))
   tmp = np.cumsum(dx ** pow)
   tmp = np.pad(tmp, (1, 0), 'constant')
   return np.interp(np.linspace(0, tmp.max(), nbin + 1), tmp, np.sort(x))


# ### Down/upsample masks

def tscrunch_mask(msk, fac):
    """
    mask of shape (int,chan)
    scrunch in time by <fac>, taking the logical or of the two rows
    returned mask will be shape (int/fac, chan)
    """
    remainder = msk.shape[0] % fac
    if remainder:
        tmp = msk[:-remainder,:].astype(int)
        excess = msk[-remainder,:].astype(int)
        return_nint = msk.shape[0] // fac + 1

        mout = np.zeros((return_nint, msk.shape[1]), dtype=bool)
        mout[:-1,:] = tmp.reshape(-1,fac,tmp.shape[-1]).sum(1) > 0
        mout[-1,:] = excess.sum(0) > 0
        return mout
    else:
        tmp = msk.astype(int)
        return tmp.reshape(-1,fac,tmp.shape[-1]).sum(1) > 0

def upsample_mask(msk, fac):
    """
    mask of shape (int, chan)
    upsample in time by <fac>
    returned mask will be shape (int*fac, chan)
    """
    return np.repeat(msk, fac, axis=0)

def reshape_extra_stats_mask(rfimask_shape, msk, fdp_gulp, ptsperint):
    """
    reshape mask derived from extra_stats to match <rfimask_shape>
    fdp_gulp = gulp used when running fdp
    ptsperint = rfimask's ptsperint
    msk = mask to be reshaped (shape is of form nint,nchan)

    will use tscrunch_mask if fdp_gulp < ptsperint
    will use upsample_mask if fdp_gulp > ptsperint
    """
    if msk.shape == rfimask_shape:
        return msk
    if fdp_gulp < ptsperint:
        logging.warning("WARNING ptsperint > fdp_gulp, unless ran into memory issues go redo fdp with higher gulp")
        if ptsperint % fdp_gulp:
            raise AttributeError(f"ptsperint ({ptsperint}) does not divide evenly into fdp_gulp ({fdp_gulp})")
        tscrunch_fac = int(ptsperint // fdp_gulp)
        return tscrunch_mask(msk, tscrunch_fac)
    else:
        if fdp_gulp % ptsperint:
            raise AttributeError(f"fdp_gulp ({fdp_gulp}) does not divide evenly into ptsperint ({ptsperint})")
        upsample_fac = int(fdp_gulp // ptsperint)
        tmp = upsample_mask(msk, upsample_fac)
        if tmp.shape != rfimask_shape:
            if tmp.shape[0] == rfimask_shape[0] + 1:
                return tmp[:-1,:]
            else:
                raise AttributeError(f"odd shape problem:\noriginal {msk.shape}, upsampled to {(upsample_fac)}, trying to match {rfimask_shape}")
        else:
            return tmp
                
def reshape_rfifind_mask(extra_stats_shape, msk, fdp_gulp, ptsperint):
    """
    reshape mask derived from rfifind stats to match <extra_stats_shape>
    fdp_gulp = gulp used when running fdp
    ptsperint = rfimask's ptsperint
    msk = mask to be reshaped (shape is of form nint,nchan)

    will use upsample_mask if fdp_gulp < ptsperint
    will use tscrunch_mask if fdp_gulp > ptsperint
    """
    if msk.shape == extra_stats_shape:
        return msk
    if fdp_gulp < ptsperint:
        logging.warning("WARNING ptsperint > fdp_gulp, unless ran into memory issues go redo fdp with higher gulp")
        if ptsperint % fdp_gulp:
            raise AttributeError(f"ptsperint ({ptsperint}) does not divide evenly into fdp_gulp ({fdp_gulp})")
        upsample_fac = int(ptsperint // fdp_gulp)
        tmp = upsample_mask(msk, upsample_fac)
        if tmp.shape != extra_stats_shape:
            if tmp.shape[0] == extra_stats_shape[0] + 1:
                return tmp[:-1,:]
            else:
                raise AttributeError(f"odd shape problem:\noriginal {msk.shape}, upsampled to {(upsample_fac)}, trying to match {extra_stats_shape}")
    else:
        if fdp_gulp % ptsperint:
            raise AttributeError(f"fdp_gulp ({fdp_gulp}) does not divide evenly into ptsperint ({ptsperint})")
        tscrunch_fac = int(fdp_gulp // ptsperint)
        return tscrunch_mask(msk, tscrunch_fac)


# ## Zapping functions

def get_zeros_mask_alt(rfifind_obj, ignorechans=[], verbose=False, plot_diagnostics=True, ax=None):
    """
    Get a mask where the std_stats = 0
    Then shuffle it +- 1 interval on each side
    Often if something went wrong, like a node going down or a network error, intervals either side are affected also
    This, hopefully catches that but also doesn't throw out whole channel if only part of it has dropped out.

    verbose and plot_diagnostics both concern where std==0 in the data in places not covered by ignorechans
    """
    tmp = (rfifind_obj.std_stats==0)

    working_mask = np.zeros_like(rfifind_obj.mask, dtype=bool)
    working_mask[:,np.array(ignorechans)] = True
    if plot_diagnostics:
        if ax is None:
            fig, ax = plt.subplots()
        ax.pcolormesh(np.ma.array(tmp, mask=working_mask).T)
        ax.set_xlabel("int")
        ax.set_ylabel("chan")
        ax.set_title("Plot of where std_stats==0, masked by the ignorechans")

    # add ignorechans to mask
    for ii in ignorechans:
        tmp[:,ii] = 1

    if verbose:
        ignorechans = set(ignorechans)
        inv_tmp = ~tmp  # so inv_tmp is 0 anywhere data is zero

        whole_ints = set(np.where(inv_tmp.sum(axis=1)==0)[0])
        whole_chans = set(np.where(inv_tmp.sum(axis=0)==0)[0])
        additional_whole_chans = whole_chans.difference(ignorechans)

        if whole_ints:
            logging.info(f"Found whole interval/s where std_stats==0: {sorted(list(whole_ints))}")
            working_mask[np.array(list(whole_ints)),:] = 1

        if additional_whole_chans:
            logging.info(f"Found whole channel/s where std_stats==0 (not covered by ignorechans): {sorted(list(additional_whole_chans))}")
            working_mask[:,np.array(list(whole_chans))] = 1


        # assume we're dealing with partial channels only and NO partial intervals
        inv_tmp2 = np.ma.array(inv_tmp, mask=working_mask)
        partial_channels = list(np.where((inv_tmp2 == 0).any(axis=0))[0])
        if partial_channels:
            logging.info(f"Found partial channel/s where std_stats==0 (not covered by ignorechans): {partial_channels}")

    # shuffling mask +-1 int
    tmp[1:,:] = (tmp[1:,:] | tmp[:-1,:])
    tmp[:-1,:] = (tmp[:-1,:] | tmp[1:,:])

    return tmp.astype(bool)


def cut_off_high_fraction(existing_mask, hard_thresh_chans=0.2, hard_thresh_ints=0.3 ,cumul_threshold_chans=0.95, cumul_threshold_ints=0.95, plot_diagnostics=True, verbose=True, ax=None, axhard=None):  #, debug=False):
    """
    First cut off channels with a fraction masked above <hard_thresh_chans> (this is intfrac in rfifind)
    And cut off intervals with a fraction masked above <hard_thresh_ints> (this is chanfrac in rfifind)

    Then:
    For each channel, calculate the fraction of intervals which have been zapped (discounting entire zapped intervals)
    Make a cumulative distribution for these fractions
    Zap fractions above where the cumulative distribution is greater than <cumul_threshold_*>
    e.g. if have 10 channels with zapped fractions of 0.1,0.1,0.4,0.1,0.1,0.1,0.1,0.2,0.1,0.2
        and set cumul_threshold_chans=0.9
        the top 10% would be zapped, in this case just channel 2 with a fraction of 0.4

    Same thing for each interval

    <cumul_threshold_chans> is used to determine which channels to zap (per-channel fractions plot)
    <cumul_threshold_ints> is used to determine which ints to zap (per-interval fractions plot)

    ax, if passed in, is an axes array of shape (2,2) (with sharex='col')
    axhard, if passed in, is an axes array of shape (1,2) (this displays the plot for the hard cut)
    these two are different as want the sharex='col' for ax, and if included the hard cut in that the scale would be annoting
    """
    nint, nchan = existing_mask.shape
    zapped_chans = np.where(existing_mask.sum(axis=0) == nint)[0]
    nzapped_chans = zapped_chans.shape[0]
    zapped_ints = np.where(existing_mask.sum(axis=1) == nchan)[0]
    nzapped_ints = zapped_ints.shape[0]
    logging.debug(f"start nzapped_chans={nzapped_chans}, nzapped_ints={nzapped_ints}")

    frac_data_chans = (existing_mask.sum(axis=0) - nzapped_ints)/(nint-nzapped_ints)
    frac_data_ints = (existing_mask.sum(axis=1) - nzapped_chans)/(nchan-nzapped_chans)

    #if debug:
    #    for cc in range(nchan):
    #        logging.debug(f"{cc}:\t{frac_data_chans[cc]}")

    # cut off things above the hard threshold
    chans_to_zap_hard = np.where((frac_data_chans > hard_thresh_chans) & (frac_data_chans != 1))[0]
    if verbose:
        logging.info(f"Channels to zap from hard fraction threshold {hard_thresh_chans}: {chans_to_zap_hard}")
    ints_to_zap_hard = np.where((frac_data_ints > hard_thresh_ints) & (frac_data_ints != 1))[0]
    if verbose:
        logging.info(f"Intervals to zap from hard fraction threshold {hard_thresh_ints}: {ints_to_zap_hard}")

    if plot_diagnostics:
        show_plot = False
        if axhard is None:
            fighard, axhard = plt.subplots(1,2, figsize=(12,4))
            show_plot = True
        axhard[0].hist(frac_data_chans[frac_data_chans != 1], bins=40, density=True)
        axhard[0].axvline(hard_thresh_chans, c='red')
        axhard[1].hist(frac_data_ints[frac_data_ints != 1], bins=40, density=True)
        axhard[1].axvline(hard_thresh_ints, c='red')
        axhard[0].set_title("chans hard threshold cut")
        axhard[1].set_title("ints hard threshold cut")
        if show_plot:
            plt.show()
            plt.close()

    assert (zapped_chans == np.where(frac_data_chans == 1)[0]).all()
    assert (zapped_ints == np.where(frac_data_ints == 1)[0]).all()

    # remake frac_data_chans
    fracs_mask_hard = np.zeros((nint, nchan), dtype=bool)
    fracs_mask_hard[:,chans_to_zap_hard] = True
    fracs_mask_hard[ints_to_zap_hard, :] = True

    nzapped_ints = len(set(zapped_ints).union(set(ints_to_zap_hard)))
    nzapped_chans = len(set(zapped_chans).union(set(chans_to_zap_hard)))
    logging.debug(f"after hard nzapped_chans={nzapped_chans}, nzapped_ints={nzapped_ints}")
    frac_data_chans = ((existing_mask|fracs_mask_hard).sum(axis=0) - nzapped_ints)/(nint - nzapped_ints)
    frac_data_ints = ((existing_mask|fracs_mask_hard).sum(axis=1) - nzapped_chans)/(nchan - nzapped_chans)

    # make cumulative distributions and apply threshold
    sorted_fracs_chan = np.sort(frac_data_chans[frac_data_chans !=1])
    N_chans = sorted_fracs_chan.size
    cumul_chans = np.arange(N_chans)/N_chans
    try:
        chan_frac_threshold = sorted_fracs_chan[cumul_chans >= cumul_threshold_chans][0]
    except IndexError:
        chan_frac_threshold = 1
    chans_to_zap = np.where((frac_data_chans >= chan_frac_threshold) & (frac_data_chans != 1))[0]
    if verbose:
        logging.info(f"Channels to zap from cumulative threshold of {cumul_threshold_chans}: {sorted(list(set(chans_to_zap).difference(set(zapped_chans))))}")


    sorted_fracs_int = np.sort(frac_data_ints[frac_data_ints !=1])
    N_ints = sorted_fracs_int.size
    cumul_ints = np.arange(N_ints)/N_ints
    try:
        int_frac_threshold = sorted_fracs_int[cumul_ints >= cumul_threshold_ints][0]
    except IndexError:
        int_frac_threshold = 1
    ints_to_zap = np.where((frac_data_ints >= int_frac_threshold) & (frac_data_ints != 1))[0]
    if verbose:
        logging.info(f"Intervals to zap from cumulative threshold of {cumul_threshold_ints}: {sorted(list(set(ints_to_zap).difference(set(zapped_ints))))}")

    if plot_diagnostics:
        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(2,2, sharex='col')
            show_plot = True
        ax[0,0].set_title(f"Per-channel fractions")
        ax[0,0].hist(frac_data_chans[frac_data_chans != 1], bins=40, density=True)
        if cumul_threshold_chans != 1:
            ax[0,0].axvline(chan_frac_threshold, c='orange')
        else:
            # the hard thresholds are not marked on the plot if cumul_threshold_chans != 1
            # because the fractions plotted are based on already having removed channels over the hard threshold
            # it's still a bit misleading to plot it when cumul_threshold_chans == 1, but a bit less so
            ax[0,0].axvline(hard_thresh_chans, c='red')

        ax[1,0].set_title(f"cumulative, threshold={cumul_threshold_chans}")
        ax[1,0].plot(sorted_fracs_chan, cumul_chans)
        ax[1,0].axhline(cumul_threshold_chans, c='orange')

        ax[0,1].set_title(f"Per-interval fractions")
        ax[0,1].hist(frac_data_ints[frac_data_ints != 1], bins=40, density=True)
        if cumul_threshold_ints != 1:
            ax[0,1].axvline(int_frac_threshold, c='orange')
        else:
            ax[0,1].axvline(hard_thresh_ints, c='red')
        ax[1,1].set_title(f"cumulative, threshold={cumul_threshold_ints}")
        ax[1,1].plot(sorted_fracs_int, cumul_ints)
        ax[1,1].axhline(cumul_threshold_ints, c='orange')

        if show_plot:
            plt.show()
            plt.close()

    fracs_mask = np.zeros((nint, nchan), dtype=bool)
    fracs_mask[:,zapped_chans] = True
    fracs_mask[zapped_ints,:] = True
    fracs_mask[:,chans_to_zap] = True
    fracs_mask[ints_to_zap,:] = True

    return fracs_mask|fracs_mask_hard

# helper function for get_step_chans
def rescale(a, b):
    """rescale a to b so that they have the same min and max"""
    mina = np.ma.min(a)
    dra = np.ma.max(a) - mina
    minb = np.ma.min(b)
    drb = np.ma.max(b) - minb

    return (a - mina)*drb/dra + minb

# ID channels with sharp steps in them
# based off https://stackoverflow.com/questions/48000663/step-detection-in-one-dimensional-data
def get_step_chans(stat, thresh=30, ignorechans=[], return_stats=False, return_plots=False, output_pdf=None):
    """stat of shape (nint, nchan) (numpy masked array)
    for each channel in stat, look for steps (via subtracting the mean and then taking the negative of np.cumsum)
    If the max of that is > thresh it gets zapped
    NB This means if a giant pulse or something happens will likely zap it

    Returns a list of channels to zap
    if return_stats returns
    <list of channels to zap> <channels> <max peak value>
    if return_plots will return a list of figs also (as last item returned)
    output_pdf is a bit weird. If return_plots is True then it does nothing
        otherwise it should be None (in which case plots are shown) or a PdfPages object
    """
    to_zap = []
    cs = []
    ms = []
    if return_plots:
        figs = []

    for c in np.arange(stat.shape[1]):
        if c not in ignorechans and not stat.mask[:,c].all():
            dary = stat[:,c]
            dary -= np.average(stat[:,c])
            dary_step = -np.ma.cumsum(dary)
            m = dary_step.max()
            cs.append(c)
            ms.append(m)
            if m > thresh:
                figtmp, axtmp = plt.subplots(2,1)
                axtmp[1].pcolormesh(stat[:,max(0,c-10):min(c+10,stat.shape[1])].T)
                axtmp[1].axhline(10, c='red')
                figtmp.suptitle(f"{c}: {dary_step.max()}")
                axtmp[0].plot(dary)
                axtmp[0].plot(rescale(dary_step, dary), c='orange')
                if return_plots:
                    figs.append(figtmp)
                else:
                    output_plot(figtmp, pdf=output_pdf)
                to_zap.append(c)
    figtmp, axtmp = plt.subplots()
    axtmp.hist(ms, equal_area(ms, 32), density=True)
    axtmp.axvline(thresh, c="orange")
    axtmp.set_title("distribution of stat used to cut on (max of a mean-subtracted cumulative sum)")
    if return_plots:
        figs.append(figtmp)
    else:
        output_plot(figtmp, pdf=output_pdf)

    if not return_stats and not return_plots:
        return to_zap
    else:
        to_return = [to_zap]
        if return_stats:
            to_return.extend([cs, ms])
        if return_plots:
            to_return.append(figs)
        return to_return


# ### iqrm

def run_iqrm_2D(data, mask, axis, r, size_fill=3):
    """Run iqrm on 1d slices of 2d data,
    axis means which direction to take the slice
    aka if data is shape (nint, nchan):
        axis=1 means for each interval, run iqrm looking for outlier channels
        axis=0 menas for each channel, run iqrm looking for outlier intervals

    size_fill = factor to use for fill_value, med + <size_fill> * (max-med)
    """
    a0,a1 = data.shape
    out = np.zeros((a0,a1), dtype=bool)
    if mask is None:
        use = data
    else:
        statistic = np.ma.array(data, mask=mask)
        fill_value = get_fill_value(statistic, size_fill, mode="max-med")
        use = statistic.filled(fill_value)

    if axis == 1:
        for i in range(a0):
            out[i,:], v = iqrm_mask(use[i,:], r)

    if axis == 0:
        for j in range(a1):
            out[:,j], v = iqrm_mask(use[:,j].T, r)

    return out

def get_fill_value(masked_statistic, factor, mode="max-med"):
    """Get fill value for filling in NaNs
    modes are
        "max" = Use factor*max
        "max-med" = Use median + factor*(max-median)
        "med" = Use median
    """
    if mode == "max":
        return factor * masked_statistic.max()
    if mode == "max-med":
        md = np.ma.median(masked_statistic)
        return md + factor * (masked_statistic.max() - md)
    if mode == "med":
        return np.ma.median(masked_statistic)
    else:
        logging.error("Invalid option for mode, must be one of ['max', 'max-med', 'med']")


def get_iqrm_chans(stat, mask, out='mask', rfac=8, flip=False, reduction_function=np.ma.median, size_fill=3, fill_mode="max-med"):
    """
    Mask channels based on (nint, nchan) stat passed in operated upon by <reduction_function> along the interval axis(=0)

    flip = do max - thing, to ID dips
    masked values filled in with 2*max

    The result can either be a set of channels (out='set')
    or the channel mask broadcast to shape (nint, nchan) (out='mask')

    the r used in iqrm is set to nchan/rfac
    IQRM documentation recommends 10 as a starting/generally-ok-for-most-setups option
    """
    if out not in ['mask', 'set']:
        raise AttributeError(f"out of {out} not recognised, must be one of 'mask', 'set'")
    if mask is None:
        masked_thing = stat
    else:
        masked_thing = np.ma.array(stat, mask=mask)
    reduced_thing = reduction_function(masked_thing, axis=0)
    non_nan = reduced_thing[np.where(np.invert(np.isnan(reduced_thing)))]
    med_non_nan = np.ma.median(non_nan)

    if flip:
        logging.debug("flipping")
        use = non_nan.max() - reduced_thing
        # setting the fill value to 3*(max-median) dist / same for flip, does improve things a bit
        # try that for other stuff too where it's still 2*max?
        fill_value1 = med_non_nan + size_fill*(med_non_nan - non_nan.min())
        fill_value = get_fill_value(use, size_fill, mode=fill_mode)
        logging.debug("fill value check", fill_value1, fill_value)
    else:
        logging.debug("not flipping")
        use = reduced_thing
        fill_value1 = med_non_nan + size_fill*(non_nan.max() - med_non_nan)
        fill_value = get_fill_value(use, size_fill, mode=fill_mode)
        logging.debug("fill value check", fill_value1, fill_value)

    r = stat.shape[1] / rfac

    #plt.plot(use.filled(fill_value))

    iqmask_stdavg, v = iqrm_mask(use.filled(fill_value), radius=r)

    chanset = set(np.where(iqmask_stdavg)[0])
    if out == 'set':
        return chanset

    iqmask_stdavg_broadcast = (iqmask_stdavg * np.ones(stat.shape)).astype(bool)
    if out == 'mask':
        return iqmask_stdavg_broadcast


def iqrm_of_median_of_means(means_data, mask, r, to_return="mask", plot=True, ax=None):
    """
    Find intervals to zap in means_data (shape (nint, nchan))
    by taking the median in each interval and running 1D iqrm to look for outliers

    r = radius to use with iqrm
    to_return governs the form of the output. Options are "array", "set", "mask", "chan_mask"
        array = array of interval indices to zap
        set = set of interval indices to zap
        mask = mask of same shape as means
        int_mask = mask of shape means.shape[0]

    plot=True, make a plot showing the median of the means with the masked version overlaid

    """
    if to_return not in ["array", "set", "mask", "int_mask"]:
        raise AttributeError(f"to_return ({to_return}) must be one of 'array', set', 'mask', 'int_mask'")
    thing = np.ma.median(np.ma.array(means_data, mask=mask), axis=1)
    q, v = iqrm_mask(thing, radius=r_int)
    if plot:
        if ax is None:
            figtmp, ax = plt.subplots()
        ax.plot(thing, "x")
        ax.plot(np.ma.array(thing.data, mask=(thing.mask|q)), "+")
        ax.set_xlabel("interval")
        ax.set_ylabel("median of the means")
    if to_return == "int_mask":
        return q
    if to_return == "array":
        return np.where(q)[0]
    if to_return == "set":
        return set(np.where(q)[0])
    if to_return == "mask":
        msk = np.zeros(means_data.shape, dtype=bool)
        msk[np.where(q)[0],:] = True
        return msk


# ### iterative +- sigma


# this works but iqrm is better
# and I was using a positive_only version of this which is literally what iqrm is designed for
# nope I take it back, iqrm zaps more than it needs to!

# this is less necessary now I have get_step_chans, but still good!
def reject_pm_sigma_iteration(arr1d, init_mask, thresh=5, plot=False, positive_only=False, iteration=0, prev_hits=np.array([9E9])):
    tmp = np.ma.array(arr1d, mask=init_mask)
    working_mask = copy.deepcopy(init_mask)
    md = np.ma.median(tmp)
    sig = np.ma.std(tmp)

    lo_lim = md - thresh*sig
    hi_lim = md + thresh*sig

    if positive_only:
        condit = (tmp > hi_lim)
    else:
        condit = ((tmp > hi_lim) | (tmp < lo_lim))

    all_hits = np.where(condit)[0]
    new_hits = np.ma.where(condit)[0]
    logging.debug(f"iteration {iteration}: std = {sig}, {condit.sum()} match/es : {np.ma.where(condit)[0]}" )

    if plot:
        plt.plot(np.ma.array(arr1d, mask=working_mask), "x")
        plt.axhline(hi_lim, c='orange')
        if not positive_only:
            plt.axhline(lo_lim, c='orange')
        plt.show()

    if len(new_hits) == 0:
        logging.info(f"channels zapped: {np.where(condit)[0]}")
        return working_mask

    for c in np.where(condit):
        working_mask[c] = True

    return reject_pm_sigma_iteration(arr1d, working_mask, thresh=thresh, plot=plot, positive_only=positive_only, iteration=iteration+1, prev_hits=all_hits)



# ### plotting

plt.rcParams['figure.figsize'] = [12, 8]

def plot_stat_map(stat, axis=None, mask=None, **plot_kwargs):
    nint, nchan = stat.shape
    grids = np.meshgrid(np.arange(nint + 1), np.arange(nchan + 1), indexing='ij')
    # for some WEIRD reason passing in the grids introduces a bunch of artifacts
    # sure not real as when zoom in they disappear/some are still there but much smaller than 1 int/chan
    # tried passing in centers and shading='nearest' and they're still there
    # tried a bunch of other things like snap=True, pcolor, saving the fig, etc but NADA! V confusing!!
    # doesn't show up is use pcolormesh without x and y though
    # BUT some bright single channel stuff you then can't see.
    # So I think he renderer is upping the resolution or something odd and it's producing weirdness
    # could not find a solution so using the thing that means I don't freak out that the data is terrible and lose half a day

    if type(mask) != np.ndarray:
        to_plot = stat
    else:
        to_plot = np.ma.masked_array(data=stat, mask=mask)

    if axis == None:
        #im = plt.pcolormesh(grids[0], grids[1], to_plot, shading='flat', **plot_kwargs)
        im = plt.pcolormesh(to_plot.T, **plot_kwargs)
        plt.xlabel="interval"
        plt.ylabel="channel"
        plt.colorbar(im)
        plt.show()
    else:
        #im = axis.pcolormesh(grids[0], grids[1], to_plot, shading='flat', **plot_kwargs)
        im = axis.pcolormesh(to_plot.T, **plot_kwargs)
        plt.colorbar(im, ax=axis)
        return axis

def plot_stat_v_nchan(stat, axis=None, mask=None, reduction_function=np.ma.median, **plot_kwargs):
    nint, nchan = stat.shape

    if not isinstance(reduction_function, Iterable):
        reduction_function = [reduction_function]

    if type(mask) != np.ndarray:
        to_plot = stat
    else:
        to_plot = np.ma.masked_array(data=stat, mask=mask)

    if axis == None:
        for red_func in reduction_function:
            plt.plot(red_func(to_plot, axis=0), np.arange(nchan), **plot_kwargs)
        plt.xlabel="channel"
        plt.show()
    else:
        for red_func in reduction_function:
            axis.plot(red_func(to_plot, axis=0), np.arange(nchan), **plot_kwargs)

def plot_stat_v_nint(stat, axis=None, mask=None, reduction_function=np.ma.median, **plot_kwargs):
    nint, nchan = stat.shape

    if type(mask) != np.ndarray:
        to_plot = stat
    else:
        to_plot = np.ma.masked_array(data=stat, mask=mask)

    if not isinstance(reduction_function, Iterable):
        reduction_function = [reduction_function]

    x = np.arange(nint)

    if axis == None:
        for red_func in reduction_function:
            plt.plot(x, red_func(to_plot, axis=1), **plot_kwargs)

        plt.xlabel("interval")
        plt.show()
    else:
        for red_func in reduction_function:
            axis.plot(x, red_func(to_plot, axis=1), **plot_kwargs)

def plot_map_plus_sums(stat, mask=None, reduction_function=np.ma.median, returnplt=False, fill=True, **plot_kwargs):
    """returnplt => return fig, ((ax0, ax1), (ax2, ax3))
    fill=True => if mask passed in or stat is a masked array, fill in masked values with the median"""
    nint, nchan = stat.shape
    #grids = np.meshgrid(np.arange(nint + 1), np.arange(nchan + 1), indexing='ij')
    widths = [3,1]
    heights = [1,3]
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2,
                                                 sharex='col', sharey='row',
                                                 gridspec_kw={'width_ratios': widths, 'height_ratios': heights})
    ax1.set_axis_off()
    if mask is not None:
        statt = np.ma.masked_array(data=stat, mask=mask)
    else:
        statt = stat
    if isinstance(statt, np.ma.MaskedArray) and fill:
        plot_stat_map(statt.filled(np.ma.median(statt)), axis=ax2)
    else:
        plot_stat_map(statt, axis=ax2)
    plot_stat_v_nchan(statt, axis=ax3, reduction_function=reduction_function)
    plot_stat_v_nint(statt, axis=ax0, reduction_function=reduction_function)

    # colour bar throws things off, this rescales the v_nint plot
    pos_map = ax2.get_position()
    pos_int = ax0.get_position()
    ax0.set_position([pos_map.x0,pos_int.y0,pos_map.width,pos_int.height])

    # labels
    ax2.set_xlabel("interval")
    ax2.set_ylabel("channel")
    if returnplt:
        return fig, ((ax0, ax1), (ax2, ax3))
    else:
        plt.show()

def plot_masked_channels_of_med(thing, channels, ax=None):  #, sig_lims=[3,3]):
    """
    plt the median of <thing> (shape (nint, nchan)) along axis 0
    and highlight <channels>
    sig_lims is for setting ylimits, limits come from med - sig_lims[0]*std and med + sig_lims[0]*std
        where med and std are calculated ignoring values in <channels>
        (if any non-<channels> values are greater or less than this that will be the limit instead)
    """
    nchan = thing.shape[1]
    med_thing = np.ma.median(thing, axis=0)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.arange(nchan), med_thing, "+")
    ax.plot(np.array(list(channels)), med_thing[np.array(list(channels))], "x")
    #unmasked_channels = set(range(nchan)).difference(set(channels))
    #get_limits_from  = med_thing[np.array(list(unmasked_channels))]
    #md = np.ma.median(get_limits_from)
    #std = np.ma.std(get_limits_from)
    #lo = min([md-sig_lims[0]*std, get_limits_from.min()])
    #hi = max([md+sig_lims[1]*std, get_limits_from.max()])
    #ax.set_ylim(lo,hi)

def plot_mask(mask, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.pcolormesh(mask.T)
    ax.set_ylabel("channel")
    ax.set_xlabel("interval")


def check_mask_and_continue(old_mask, old_mask_exstats, add_mask, add_mask_exstats, threshold, rfimask, means, var, pdf, stage=None):
    """
    Check if adding add_mask to old_mask means the masking fraction goes above the <threshold> 
    If it does:
        plot summary plots which will hopefully show what went wrong
        return the old masks
    If it doesn't:
        return (old_mask | add_mask), (old_mask_exstats | add_mask_exstats)
    
    """
    zap_frac = masked_frac(old_mask | add_mask)
    if  zap_frac >= threshold:
        logging.warning(f"{stage}: zaps {zap_frac} of data, which is over the problem threshold, plotting summary and skipping")
        logging.info(f"{stage}: working maask unchanged")
        make_summary_plots(add_mask, add_mask_exstats, rfimask, means, var, pdf, title_insert=f"ERROR stage {stage}")
        return old_mask, old_mask_exstats
    else:
        logging.info(f"{stage}: zaps {zap_frac} of data")
        logging.info(f"{stage}: updating working mask")
        return (old_mask | add_mask), (old_mask_exstats | add_mask_exstats)


# ## Stuff that needs to be input to the code

#args_in = [
    #"test_data/SURVEYv0_point44_DM34_59831_pow_fdp_rfifind.mask",
    #"test_data/gulp76800/SURVEYv0_point44_DM34_59831_pow_fdp_stats.npz",
#    "test_data/SURVEYv0_point97_DM34_59634_pow_fdp_rfifind.mask",
#    "test_data/SURVEYv0_point97_DM34_59634_pow_fdp_stats.npz",
    #"--option", "0,1,2,3,4,5,6",
#    "--option", "0,1,2,3,4,6",
#    "--outfilename", "pipeline_rfifind.mask",
#    "--show",
#    "--ignorechans",
#    "1023,1000,970,965,957,941,928,917,914,910,909,908,906,905,904,903,902,901,900,899,898,897,896,895,894,893,892,891,890,889,888,887,886,885,882,881,880,879,878,877,876,875,874,873,872,871,870,869,868,866,865,862,861,860,859,858,857,856,855,854,853,852,851,850,849,848,847,846,845,843,840,835,832,826,825,821,813,757,755,750,731,728,712,710,706,704,702,699,696,691,689,688,685,682,672,664,662,661,658,656,645,617,616,611,610,606,600,594,592,586,585,584,579,578,577,576,575,574,573,572,571,570,569,568,567,566,564,563,562,561,560,559,558,556,544,536,533,517,512,509,501,493,485,475,471,469,468,467,466,465,464,462,461,460,459,458,457,456,455,451,448,440,439,438,436,434,433,432,430,429,428,425,424,421,397,385,384,380,379,378,368,363,345,339,333,332,325,323,317,309,302,301,285,282,280,272,269,267,259,258,257,256,253,251,246,241,235,234,232,225,224,222,221,215,206,189,181,175,141,125,109,104,101,85,83,77,69,45,24,18,13,5"
#]


# ## Parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Run some rfi mitigation, will write an rfifind-style mask and inf file. Will also write a .ignorechans file a list of (presto format) ignorechans to pass onto the next stage"
)

def _check_mask_name(string):
    if string[-12:] != "rfifind.mask":
        raise argparse.ArgumentTypeError(f'Invalid argument for maskfile/outfilename ({string}): must end in rfifind.mask')
    return string

def _check_m_frac_threshold(string):
    value = float(string)
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError(f'Invalid argument for m_frac_threshold ({value}): cannot be <0 or >1')
    return value

def _check_rfac(string):
    value = float(string)
    if value <= 0:
        raise argparse.ArgumentTypeError(f'Invalid argument for rfac ({value}): must be a positive number')
    return value

parser.add_argument(
    "maskfile",
    type=_check_mask_name,
    help=".mask file output from rfifind. Must also have a corresponding .stats and .inf file"
)
parser.add_argument(
    "extra_stats_file",
    type=str,
    help="""npz file containing extra stats from the fdp process. Must contain n, num_unmasked_points, s1, s2, gulp.
    n = number of points in each interval, useful as the last one is often shorter
    num_unmasked_points = number of points that went into the s1 ans s2 calculation
    s1 = sum
    s1 = sum of the squares
    gulp = gulp used to make stats"""
)

parser.add_argument(
    "--option",
    type=str,
    default="0,1,2,3,4,6",
    help="""Which masking options to use. Comma separated string, will be run from 0 up regardless of order passed in.
    *Highly* recommend always including 0.
    Default is '0,1,2,3,4,6'
    0 = basic processing: ignorechans, anywhere the std is 0, where the number of unmasked points is < set threshold, the rfifind mask, a high fraction cut
    1 = cutting channels which have steps in them (detected via a cumsum)
    2 = cutting outlier intervals, running iqrm on the per-interval median of the means
    3 = cutting outlier channels: running iqrm on the per-channel median of the generalized spectral kurtosis statistic
    4 = running iqrm in 2D on the means, along the time axes (looking for outlier channels in each interval)
    5 = running iqrm in 2D on the means, along the freq axis (looking for outlier intervals in each channels)
    ~ if run either 4/5/both there will be a high fraction cut here ~
    6 = cut channels where the std of the means in each channel is high
    """
)

group = parser.add_mutually_exclusive_group(required=False)

group.add_argument(
    "--outfilename",
    type=_check_mask_name,
    help="filename to write output mask to"
)

group.add_argument(
    "-o",
    "--overwrite",
    action='store_true',
    help="Overwrite the mask with the new one"
)

parser.add_argument(
    "--m_frac_threshold",
    default=0.5,
    type=_check_m_frac_threshold,
    help="If num_unmasked_points in a block is < this fraction, mask this int,chan block"
)

parser.add_argument(
    "--rfac",
    default=10,
    type=_check_rfac,
    help="fraction of the band to use for iqrm window, e.g. if nchan=1024 and pass in 16, window will be 64 channels"
)

parser.add_argument(
    "--show",
    action='store_true',
    help="Show plots rather than saving them as a pdf"
)

parser.add_argument(
    "--dont_flip_band",
    action='store_true',
    help="""If this flag is passed in the channel order in <extra_stats_file> will NOT be flipped.
    If it came from an fdp script channels will be in sigproc order which is the opposite from presto order and thus most of the time they'll need to be flipped"""
)

parser.add_argument(
    "--ignorechans",
    type=str,
    default="",
    help="string of ignorechans in presto format, aka channel 0 is the lowest freq channel, and it's a string separated by commas"
)

parser.add_argument(
    "--problem_frac",
    type=float,
    default=0.7,
    help="If masking fraction goes above this threshold then there is a problem, skip whatever step did this"
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

# args = parser.parse_args(args_in)
args = parser.parse_args()

if args.log is not None:
    logging.basicConfig(
        filename=args.log,
        filemode="a",
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=args.loglevel,
    )
else:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=args.loglevel,
        stream=sys.stdout,
    )

logging.info("rfi_pipeline initialized with arguments:")
logging.info(args)

maskfile = args.maskfile
extra_stats_fn = args.extra_stats_file
rfac = args.rfac
m_frac_threshold = args.m_frac_threshold

if args.ignorechans == "":
    ignorechans = []
else:
    ignorechans = [int(x) for x in args.ignorechans.split(",")]

optstr="".join(args.option.split(","))
if args.overwrite:
    outfilename = args.maskfile
elif args.outfilename is None:
    outfilename = maskfile[:maskfile.rfind("_rfifind.mask")] + "_" + optstr + "_rfifind.mask"
else:
    outfilename = args.outfilename
logging.info(f"New mask will be written to: {outfilename}")

if args.show:
    p = None
else:

    plotfname = "rfipipeline_plots" + outfilename[:outfilename.rfind("_rfifind.mask")]
    if optstr not in plotfname:
        plotfname += f"_{optstr}"
    plotfname += ".pdf"
    p = PdfPages(plotfname)
    logging.info(f"Plots will be written to {plotfname}")

opts = [int(x) for x in args.option.split(",")]
opt_dict = {
    0: "basic processing: ignorechans, anywhere the std is 0, where the number of unmasked points is < set threshold, the rfifind mask, a high fraction cut",
    1: "cutting channels which have steps in them (detected via a cumsum)",
    2: "cutting outlier intervals, running iqrm on the per-interval median of the means",
    3: "cutting outlier channels: running iqrm on the per-channel median of the generalized spectral kurtosis statistic",
    4: "running iqrm in 2D on the means, along the time axes (looking for outlier channels in each interval)",
    5: "running iqrm in 2D on the means, along the freq axis (looking for outlier intervals in each channels)",
    6: "cut channels where the std of the means in each channel is high"
}
logging.info(f"Options selected:")
for x in opts:
    logging.info(f"\t{x}: {opt_dict[x]}")


# ## Load files

rfimask = rfifind.rfifind(maskfile)
logging.info(f"loaded mask from {maskfile}")

extra_stats = np.load(extra_stats_fn, allow_pickle=True)
if args.dont_flip_band:
    M = extra_stats["num_unmasked_points"][:,:]
    s1 = extra_stats["s1"][:,:]
    s2 = extra_stats["s2"][:,:]
else:
    # flip everything to presto channel convention
    logging.info("Reversing channel order in extra stats to match presto convention")
    M = extra_stats["num_unmasked_points"][:,::-1]
    s1 = extra_stats["s1"][:,::-1]
    s2 = extra_stats["s2"][:,::-1]


N = extra_stats["n"]
extra_stats_gulp = extra_stats["gulp"]

# ## make some other parameters
r = rfimask.nchan/rfac
r_int = rfimask.nint/rfac


# ## Pipeline

# ### Setup initial working mask

working_mask = np.zeros_like(rfimask.mask)
working_mask_exstats = np.zeros_like(M, dtype=bool)



if 0 in opts:
    # ### std_stats==0 mask
    logging.info(f"Ignoring {len(ignorechans)}/{rfimask.nchan} channels")

    logging.info(f"\nGetting zeros mask")
    fig0, ax0 = plt.subplots()
    m0 = get_zeros_mask_alt(rfimask, ignorechans=ignorechans, verbose=True, ax=ax0)
    output_plot(fig0, pdf=p)

    # ### M mask (where num_points_unmasked from fdp is under some threshold)
    # cut off anywhere where <0.5 of the gulp was nonzero
    mmask = (M.T < m_frac_threshold*N).T
    # right the gulp is different for fdp and rfifind. damn. will have to record that in extra_stats

    base_mask = m0|reshape_extra_stats_mask(m0.shape, mmask, extra_stats_gulp, rfimask.ptsperint)
    logging.info(f"ignorechans, std_stats=0 and lots of 0 data alone mask out {masked_frac(base_mask)} of data")

    # ### Add initial rfifind mask and do a hard threshold cut
    base_mask = base_mask | rfimask.mask
    logging.info(f"+rfifind mask  masks out {masked_frac(base_mask)} of data")

    fig00, ax00 = plt.subplots(2,2,sharex='col')
    fig01, ax01 = plt.subplots(1,2)
    mcut = cut_off_high_fraction(base_mask, cumul_threshold_chans=1, cumul_threshold_ints=1, plot_diagnostics=True, ax=ax00, axhard=ax01) #, debug=True)
    base_mask = base_mask | mcut
    output_plot(fig00, pdf=p)
    output_plot(fig01, pdf=p)

    fig1, ax1 = plt.subplots()
    ax1.pcolormesh(base_mask.T)
    fig1.suptitle("base mask: std_stats==0, large fraction of masked data within an interval, rfifind mask")
    output_plot(fig1, pdf=p)

    # update working mask
    base_mask_exstats = reshape_rfifind_mask(M.shape, base_mask, extra_stats_gulp, rfimask.ptsperint)
    logging.info(f"Reshaped base_mask from {base_mask.shape} to {base_mask_exstats.shape} for use with the extra stats from fdp")

    logging.info("0: updating working mask")
    working_mask = working_mask | base_mask
    working_mask_exstats = working_mask_exstats | base_mask_exstats
    logging.info(f"0: working mask zaps {masked_frac(working_mask)} of data")

    if masked_frac(working_mask) >= args.problem_frac:
        logging.info("Something went wrong at stage 0, making summary plots and exiting")
        M = np.ma.array(M, mask=working_mask_exstats)
        s1 = np.ma.array(s1, mask=working_mask_exstats)
        s2 = np.ma.array(s2, mask=working_mask_exstats)

        # try estimating shape parameter from the mean and std
        means = s1/M
        var = (s2 - s1**2/M)/M
        make_summary_plots(working_mask, working_mask_exstats, rfimask, means, var, p, title_insert="ERROR stage 0")
        if p is not None:
            logging.info("Writing pdf")
            p.close()
        logging.error("Something went horribly wrong at stage 0")
        sys.exit(1)


    del base_mask
    del base_mask_exstats



# ### Now have base_mask should be using as minimum input for all other steps

M = np.ma.array(M, mask=working_mask_exstats)
s1 = np.ma.array(s1, mask=working_mask_exstats)
s2 = np.ma.array(s2, mask=working_mask_exstats)

# try estimating shape parameter from the mean and std
means = s1/M
var = (s2 - s1**2/M)/M


# ### Steps mask

if 1 in opts:
    logging.info("\nGetting channels with steps")
    step_chans_to_zap, cs, ms = get_step_chans(means, ignorechans=get_ignorechans_from_mask(working_mask_exstats), thresh=30, return_stats=True, output_pdf=p)
    step_mask_exstats = np.zeros_like(means.mask)
    step_mask = np.zeros_like(rfimask.pow_stats, dtype=bool)
    for cc in step_chans_to_zap:
        step_mask_exstats[:,cc] = True
        step_mask[:,cc] = True
    logging.info(f"channels to zap: {step_chans_to_zap}\n")
    #fig02, ax02 = plt.subplots()
    #ax02.plot(cs, ms, "x-")
    #ax02.set_ylim(0, 10)
    #fig02.suptitle("steps stats")
    #output_plot(fig02, pdf=p)

    working_mask, working_mask_exstats = check_mask_and_continue(working_mask, working_mask_exstats, step_mask, step_mask_exstats, args.problem_frac, rfimask, means, var, p, stage=1)

    del step_mask
    del step_mask_exstats







# ### Look for outlier intervals, via running iqrm on the median of the means in that interval

if 2 in opts:
    logging.info("\n2: Getting outlier intervals: iqrm on the median(axis=1) of the means")
    figtmp, axtmp = plt.subplots()
    zapints_med_means_time_mask = iqrm_of_median_of_means(means.data, working_mask_exstats, r_int, ax=axtmp, to_return="array")
    output_plot(figtmp, pdf=p)
    logging.info(f"zapping intervals: {zapints_med_means_time_mask}")
    med_means_time_mask = np.zeros_like(working_mask)
    med_means_time_mask[zapints_med_means_time_mask,:] = True
    med_means_time_mask_exstats = np.zeros_like(working_mask_exstats)
    med_means_time_mask_exstats[zapints_med_means_time_mask,:] = True

    working_mask, working_mask_exstats = check_mask_and_continue(
        working_mask, working_mask_exstats, 
        med_means_time_mask, med_means_time_mask_exstats, 
        args.problem_frac, rfimask, means, var, p, 
        stage=2,
    )

    del med_means_time_mask
    del med_means_time_mask_exstats


# ## Make gsk

if 3 in opts or 4 in opts or 5 in opts:
    logging.info("\nMaking the generalized spectral kurtosis statistic, estimating d from means**2/var")
    delta = means**2/var
    gsk_d_estimate = ((M * delta + 1) / (M - 1)) * (M * (s2 / s1**2) - 1)
    gsk_d_estimate_masked = np.ma.array(gsk_d_estimate, mask=working_mask_exstats)
    gsk_d_estimate_masked.mask[np.isnan(gsk_d_estimate)] = True

    figgsk, axgsk = plot_map_plus_sums(gsk_d_estimate_masked.data, gsk_d_estimate_masked.mask, returnplt=True)
    figgsk.suptitle("GSK statistic")
    output_plot(figgsk, pdf=p)


# ## channel mask - iqrm on the median of the gsk

# This is an overzap but the best thing I've found, that doesn't require too much tuning.
# (see how it performs on other pointings too!)

if 3 in opts:
    logging.info("\n3: Getting outlier channels, running iqrm on the median of the gsk")
    gsk_med_nomask_chans = get_iqrm_chans(gsk_d_estimate_masked.data, None, rfac=rfac, size_fill=1, out='set',).difference(set(ignorechans))

    gsk_chan_mask = np.zeros_like(working_mask, dtype=bool)
    gsk_chan_mask_exstats = np.zeros_like(working_mask_exstats, dtype=bool)
    for chan in gsk_med_nomask_chans:
        gsk_chan_mask[:,chan] = True
        gsk_chan_mask_exstats[:,chan] = True

    fig2, ax2 = plt.subplots(4,1,sharex="col", figsize=(12,36))
    plot_masked_channels_of_med(gsk_d_estimate_masked, gsk_med_nomask_chans, ax=ax2[0])
    ax2[0].set_title("gsk")

    plot_masked_channels_of_med(np.ma.array(rfimask.pow_stats, mask=working_mask), gsk_med_nomask_chans, ax=ax2[1])
    ax2[1].set_title("pow_stats")

    plot_masked_channels_of_med(np.ma.array(means.data, mask=working_mask_exstats), gsk_med_nomask_chans, ax=ax2[2])
    ax2[2].set_title("means")

    plot_masked_channels_of_med(np.ma.array(var.data, mask=working_mask_exstats), gsk_med_nomask_chans, ax=ax2[3])
    ax2[3].set_title("var")
    ax2[3].set_xlabel("channel")

    fig2.suptitle("iqrm on median of gsk - channel mask")
    output_plot(fig2, pdf=p)

    working_mask, working_mask_exstats = check_mask_and_continue(
        working_mask, working_mask_exstats, 
        gsk_chan_mask, gsk_chan_mask_exstats, 
        args.problem_frac, rfimask, means, var, p, 
        stage=3,
    )

    del gsk_chan_mask
    del gsk_chan_mask_exstats


# ### 2D iqrms

fig03,ax03 = plot_map_plus_sums(means.data, mask=working_mask_exstats, returnplt=True)
fig03.suptitle("means, before 2D iqrm")
output_plot(fig03, pdf=p)

if 4 in opts or 5 in opts:
    working_mask_exstats_pre2d = copy.deepcopy(working_mask_exstats)

if 4 in opts:
    logging.info(f"\n4: Running 2D iqrm on the means, stepping through intervals, looking for bad channels")
    m_iqrm_means_freq_nomask_exstats = run_iqrm_2D(means.data, None, 1, r, size_fill=1)
    m_iqrm_means_freq_nomask = reshape_extra_stats_mask(rfimask.pow_stats.shape, m_iqrm_means_freq_nomask_exstats, extra_stats_gulp, rfimask.ptsperint)


    fig3, ax3 = plt.subplots()
    ax3.pcolormesh((m_iqrm_means_freq_nomask_exstats.astype(int) - working_mask_exstats_pre2d.astype(int)).T)
    fig3.suptitle("2D iqrm means looking for bad channels (referenced against working mask before option 4)")
    output_plot(fig3, pdf=p)

    working_mask, working_mask_exstats = check_mask_and_continue(
        working_mask, working_mask_exstats, 
        m_iqrm_means_freq_nomask, m_iqrm_means_freq_nomask_exstats, 
        args.problem_frac, rfimask, means, var, p, 
        stage=4,
    )
    
    del m_iqrm_means_freq_nomask
    del m_iqrm_means_freq_nomask_exstats

if 5 in opts:
    logging.info(f"\n5: Running 2D iqrm on the means, stepping through channels, looking for bad intervals")
    m_iqrm_means_time_nomask_exstats = run_iqrm_2D(means.data, None, 0, r_int, size_fill=1)
    m_iqrm_means_time_nomask = reshape_extra_stats_mask(rfimask.pow_stats.shape, m_iqrm_means_time_nomask_exstats, extra_stats_gulp, rfimask.ptsperint)

    fig4, ax4 = plt.subplots()
    ax4.pcolormesh((m_iqrm_means_time_nomask_exstats.astype(int) - working_mask_exstats_pre2d.astype(int)).T)
    fig4.suptitle("2D iqrm means looking for bad intervals (referenced against working mask before option 4)")
    output_plot(fig4, pdf=p)

    working_mask, working_mask_exstats = check_mask_and_continue(
        working_mask, working_mask_exstats, 
        m_iqrm_means_time_nomask, m_iqrm_means_time_nomask_exstats, 
        args.problem_frac, rfimask, means, var, p, 
        stage=5,
    )

    del m_iqrm_means_time_nomask
    del m_iqrm_means_time_nomask_exstats

if 4 in opts or 5 in opts:
    logging.info("\nAs ran 4/5 doing a high fraction cut")
    # get high fraction of 2d iqrm which was looking for bad intervals
    fig5, ax5 = plt.subplots(2,2,sharex='col')
    fig6, ax6 = plt.subplots(1,2)
    hf_exstats = cut_off_high_fraction(
        (working_mask_exstats),
        cumul_threshold_chans=0.99,
        cumul_threshold_ints=0.99,
        ax=ax5,
        axhard=ax6,
    )
    hf = reshape_extra_stats_mask(rfimask.pow_stats.shape, hf_exstats, extra_stats_gulp, rfimask.ptsperint)

    fig5.suptitle("high fraction cut of working mask post-4&5")
    output_plot(fig5, pdf=p)
    output_plot(fig6, pdf=p)

    working_mask, working_mask_exstats = check_mask_and_continue(
        working_mask, working_mask_exstats, 
        hf, hf_exstats, 
        args.problem_frac, rfimask, means, var, p, 
        stage="post4/5",
    )

    del hf
    del hf_exstats


# ### eliminate channels where the std of the mean is high
# Operate on the std of the means in each channel. Take the running median, then statistic working with is the original - the running median. (do not take the abs since if the std is lower than the running average that's fine and we don't care)
# iqrm doesn't work great here, had to tune the threshold to 12 to stop it from overzapping.
# I guess that threshold might work well for all obs and be generic but I doubt it
#
# Iterating a cut based of the median + thresh x sigma works better in this instance (with `reject_pm_sigma_iteration`)

if 6 in opts:
    logging.info("\n6:Looking for channels where the means have a high standard deviation")
    tmp_means = np.ma.array(means.data, mask=working_mask_exstats)

    # take std of means in each channel
    std_of_means = np.ma.std(tmp_means, axis=0)
    # take a running median of this
    std_of_means_runningmed = generic_filter(std_of_means.filled(fill_value=np.nan), np.nanmedian, size=(32), mode='nearest')

    # plot to check running median is ok
    fig7, ax7 = plt.subplots(3, 1, figsize=(12,27))  #(4, 1, figsize=(12,36))
    ax7[0].plot(std_of_means)
    ax7[0].plot(std_of_means_runningmed)
    ax7[0].set_title("std of means + running median")

    # do an iterative medain+5*std cut on the residuals
    t = (std_of_means - std_of_means_runningmed)
    std_of_avg_chan_mask = reject_pm_sigma_iteration(t.data, t.mask, plot=False, positive_only=True)
    std_of_avg_mask_exstats = np.tile(std_of_avg_chan_mask, reps=(working_mask_exstats.shape[0],1))
    std_of_avg_mask = np.tile(std_of_avg_chan_mask, reps=(working_mask.shape[0],1))

    ax7[1].plot(t, "x")
    ax7[1].plot(np.ma.array(t.data, mask=(t.mask|std_of_avg_chan_mask)), "+")
    ax7[1].set_title("std of means showing which have been masked")

    # plot the zapped channels
    ax7[2].plot(tmp_means[:,std_of_avg_chan_mask])
    ax7[2].set_title("means of channels zapped in this process")

    # plot unzapped channels
    #msk = copy.deepcopy(working_mask)
    #msk[:,std_of_avg_chan_mask] = True

    #thing = np.ma.array(means.data, mask=msk)
    #ax7[3].plot(thing)
    #ax7[3].set_title("means of all unzapped channels")
    #fig7.suptitle("std of means")
    output_plot(fig7, pdf=p)

# plot individual zapped channels
#for i in np.where(std_of_avg_chan_mask)[0]:
#    if tmp_means[:,i].sum():
#        print(i)
#        plt.plot(tmp_means[:,i])
#        plt.show()

    working_mask, working_mask_exstats = check_mask_and_continue(
        working_mask, working_mask_exstats, 
        std_of_avg_mask, std_of_avg_mask_exstats, 
        args.problem_frac, rfimask, means, var, p, 
        stage=6,
    )

    del std_of_avg_mask
    del std_of_avg_mask_exstats


#print(f"Doing high fraction cut on mask before finalizing")
#pre_final_mask = (working_mask|std_of_avg_mask)
#fig8, ax8 = plt.subplots(2,2,sharex='col')
#fig9, ax9 = plt.subplots(1,2)
#high_frac_fin = cut_off_high_fraction(pre_final_mask, cumul_threshold_chans=0.99, cumul_threshold_ints=0.99, ax=ax8, axhard=ax9)
#fig8.suptitle("Final mask high fraction cut")
#output_plot(fig8, pdf=p)
#output_plot(fig9, pdf=p)

# change shape back!
#print("Changing to correct shape")
#final_mask = reshape_extra_stats_mask(rfimask.pow_stats.shape, (pre_final_mask|high_frac_fin), extra_stats_gulp, rfimask.ptsperint)


# ### Wrapping up

logging.info("\nWrapping up")
wrap_up(working_mask, working_mask_exstats, rfimask, means, var, p, outfilename, infstats_too=(not args.overwrite))

sys.exit(0)
