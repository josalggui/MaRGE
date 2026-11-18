"""Reconstruction module for the MIMO coherence test (RX1 vs RXn, n repetitions)."""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io


def _circ_stats(phase_rad):
    """Circular mean and circular std (degrees) of an array of angles in radians."""
    z = np.mean(np.exp(1j * phase_rad))
    r = np.abs(z)
    mean = np.degrees(np.angle(z))
    std = np.degrees(np.sqrt(-2 * np.log(max(r, 1e-12))))
    return mean, std, r


def _lin_stats(x):
    m = np.mean(x)
    s = np.std(x, ddof=1) if len(x) > 1 else 0.0
    return m, s, (s / abs(m) * 100 if m != 0 else np.nan)


# ======================================================================================================
# Temporal analysis inside the burst (added). It only READS the same `data_decimated` array and never
# changes the stability analysis above. The horizontal axis is the SAMPLE INDEX of each channel's own
# record: differences between channels along this axis are an observed difference in the response versus
# sample index. They are NOT a measurement of ADC skew, jitter or absolute delay between boards.
# ======================================================================================================

_EDGE_LEVELS = (0.1, 0.5, 0.9)
_AXIS_NOTE = "x axis = sample index of each channel's record (not absolute time, not ADC skew)"
_DT = [None]      # sampling period (us) used for the time axis; None -> sample index
_ZOOM_US = [None]  # half width of the edge zooms in us; None -> auto (12 samples, at least 10 us)


def _zoom_half():
    """Half width of the edge zooms in samples (defined in us when the sampling period is known)."""
    if not _DT[0]:
        return 12
    half_us = _ZOOM_US[0] if _ZOOM_US[0] else max(12 * _DT[0], 10.0)
    return max(3, int(np.ceil(half_us / _DT[0])))


def _dt():
    return _DT[0] if _DT[0] else 1.0


def _xlab():
    return 'Time from start of the acquisition window (us)' if _DT[0] else 'Sample index'


def _set_axis_note():
    global _AXIS_NOTE
    if _DT[0]:
        _AXIS_NOTE = ("x axis = time = sample index x %0.3f us, counted from the start of each channel's own record "
                      "(not absolute time, not ADC skew)" % _DT[0])


def _colors(n):
    return plt.cm.turbo(np.linspace(0.05, 0.95, max(n, 1)))


def _plateau(env):
    """
    Amplitude reference of each burst (last axis): median of the samples above 50 % of the 95th percentile.
    Works both when the burst covers the whole window and when it only covers part of it (edges inside).
    """
    p95 = np.percentile(env, 95, axis=-1, keepdims=True)
    top = np.where(env >= 0.5 * p95, env, np.nan)
    return np.nanmedian(top, axis=-1)


def _valid_rf_channels(cdata, snr_min):
    """
    A channel has a valid RF signal if its amplitude is large compared with its repetition-to-repetition scatter:
    mean(|x|) / std(|x|) over repetitions, taken at the 80th percentile over the samples, so it also works when the
    burst only covers part of the window. A channel with noise only gives ~1.9 (Rayleigh); a channel with RF gives
    >> snr_min. With fewer than 3 repetitions the coherent SNR over the record is used instead.
    """
    n_ch, n_rep, n_pts = cdata.shape
    snr = np.zeros(n_ch)
    if n_rep >= 3:
        env = np.abs(cdata)
        m = env.mean(axis=1)
        sd = env.std(axis=1, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = m / sd
        r = np.where(sd == 0, np.where(m > 0, np.inf, 0.0), r)
        snr = np.percentile(r, 80, axis=-1, method='lower')
    else:
        for ic in range(n_ch):
            x = cdata[ic]
            mean = np.mean(x, axis=1)
            m = np.abs(mean)
            sd = np.sqrt(np.mean(np.abs(x - mean[:, None]) ** 2, axis=1))
            with np.errstate(divide='ignore', invalid='ignore'):
                r = m / (sd / np.sqrt(n_pts))
            r = np.where(sd == 0, np.where(m > 0, np.inf, 0.0), r)
            snr[ic] = np.median(r)
    return snr >= snr_min, snr


def _sample_snr(env):
    """Per-sample SNR from the repetition-to-repetition scatter: mean(|x|) / std(|x|) over axis -2 (repetitions)."""
    m = env.mean(axis=-2)
    sd = env.std(axis=-2, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = m / sd
    return np.where(sd > 0, r, np.where(m > 0, np.inf, 0.0))


def _wrap_deg_masked(phase_rad, valid):
    """Original circular phase in (-180, 180] deg, NaN where the sample is not valid."""
    return np.where(valid, np.degrees(phase_rad), np.nan)


def _unwrap_deg_masked(phase_rad, valid):
    """Phase in degrees, unwrapped only inside each contiguous run of valid samples (NaN elsewhere)."""
    out = np.full(np.shape(phase_rad), np.nan)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        v = np.degrees(np.unwrap(phase_rad[seg]))
        out[seg] = v - 360.0 * np.round(np.median(v) / 360.0)   # no arbitrary multiple of 360 from the first sample
    return out


def _drop_short_runs(ok, min_run=3):
    """Removes runs of valid samples shorter than min_run (an isolated noise spike above the threshold is not signal)."""
    ok = np.asarray(ok, dtype=bool).copy()
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return ok
    min_run = min(min_run, ok.size)
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if seg.size < min_run:
            ok[seg] = False
    return ok


def _edge_metrics(env, plateau):
    """
    Observed transitions of one amplitude record (sample index where it crosses 10/50/90 % of its own plateau).
    A rising edge is only reported if the record STARTS below 10 % of the plateau, a falling edge only if it
    ENDS below 10 % (median of the first / last 3 samples). Otherwise there is no edge inside the acquired window.
    '*_50_interp' is the 50 % crossing linearly interpolated between the two samples around it: an observed crossing
    index, not a delay between boards.
    """
    res = {}
    n = env.size
    if not np.isfinite(plateau) or plateau <= 0:
        for lv in _EDGE_LEVELS:
            res['rise_%d' % round(lv * 100)] = np.nan
            res['fall_%d' % round(lv * 100)] = np.nan
        res.update(rise_50_interp=np.nan, fall_50_interp=np.nan, rise_status='no signal', fall_status='no signal',
                   start_dev_percent=np.nan, end_dev_percent=np.nan)
        return res
    r = env / plateau
    m = min(3, n)
    rise_in = bool(np.median(r[:m]) < _EDGE_LEVELS[0])    # median of the first/last 3 samples: robust to one noisy point
    fall_in = bool(np.median(r[-m:]) < _EDGE_LEVELS[0])
    for lv in _EDGE_LEVELS:
        a = r >= lv
        both = a[:-1] & a[1:] if n > 2 else a                # a crossing needs 2 consecutive samples (noise spikes)
        hit = np.flatnonzero(both)
        res['rise_%d' % round(lv * 100)] = float(hit[0]) if (rise_in and hit.size) else np.nan
        res['fall_%d' % round(lv * 100)] = float(hit[-1] + 1) if (fall_in and hit.size and n > 2) else np.nan
    res['rise_50_interp'] = np.nan
    res['fall_50_interp'] = np.nan
    i = res['rise_50']
    if np.isfinite(i) and i >= 1:
        i = int(i)
        e0, e1 = r[i - 1], r[i]
        if e1 != e0:
            res['rise_50_interp'] = (i - 1) + (0.5 - e0) / (e1 - e0)
    j = res['fall_50']
    if np.isfinite(j) and j < n - 1:
        j = int(j)
        e0, e1 = r[j], r[j + 1]
        if e0 != e1:
            res['fall_50_interp'] = j + (e0 - 0.5) / (e0 - e1)
    k = max(1, int(round(0.05 * n)))
    res['start_dev_percent'] = float(np.max(np.abs(r[:k] - 1)) * 100)
    res['end_dev_percent'] = float(np.max(np.abs(r[-k:] - 1)) * 100)
    res['rise_status'] = 'rising edge in window' if rise_in else 'no rising edge in window (>=10% at start)'
    res['fall_status'] = 'falling edge in window' if fall_in else 'no falling edge in window (>=10% at end)'
    return res


def _edge_table(env, plateau, valid, channels, rep, sel_curve=None):
    """Edge table per channel for repetition `rep` (0-based), statistics over all repetitions and difference vs RX1."""
    n_ch, n_rep, n_pts = env.shape
    rows, selected = [], []
    for ic, c in enumerate(channels):
        if not valid[ic]:
            rows.append({'rx_channel': c, 'valid_rf': False})
            selected.append(None)
            continue
        ms = [_edge_metrics(env[ic, r], plateau[ic, r]) for r in range(n_rep)]
        if sel_curve is None:
            sel = ms[rep]
            plat_sel, rep_lab = float(plateau[ic, rep]), rep + 1
        else:                                              # curve shown in the figures = mean over repetitions
            sel = _edge_metrics(sel_curve[0][ic], sel_curve[1][ic])
            plat_sel, rep_lab = float(sel_curve[1][ic]), 'mean'
        row = {'rx_channel': c, 'valid_rf': True, 'repetition': rep_lab, 'plateau_amp': plat_sel,
               'rise_status': sel['rise_status'], 'fall_status': sel['fall_status']}
        for lv in _EDGE_LEVELS:
            row['rise_%d_idx' % round(lv * 100)] = sel['rise_%d' % round(lv * 100)]
        for lv in _EDGE_LEVELS:
            row['fall_%d_idx' % round(lv * 100)] = sel['fall_%d' % round(lv * 100)]
        row['rise_50_interp'] = sel['rise_50_interp']
        row['fall_50_interp'] = sel['fall_50_interp']
        row['start_dev_percent'] = sel['start_dev_percent']
        row['end_dev_percent'] = sel['end_dev_percent']
        for key in ('rise_50', 'fall_50'):
            v = np.array([m[key] for m in ms])
            ok = np.isfinite(v)
            row[key + '_reps_with_edge'] = int(ok.sum())
            row[key + '_idx_median'] = float(np.median(v[ok])) if ok.any() else np.nan
            row[key + '_idx_min'] = float(np.min(v[ok])) if ok.any() else np.nan
            row[key + '_idx_max'] = float(np.max(v[ok])) if ok.any() else np.nan
            vi = np.array([m[key + '_interp'] for m in ms])
            oki = np.isfinite(vi)
            row[key + '_interp_median'] = float(np.median(vi[oki])) if oki.any() else np.nan
        rows.append(row)
        selected.append(sel)
    df = pd.DataFrame(rows)
    # Difference of the observed crossing index with respect to RX1 (index units, not a physical delay)
    if 1 in channels and valid[channels.index(1)]:
        ref = df.loc[df['rx_channel'] == 1].iloc[0]
        for key in ('rise_50_idx_median', 'fall_50_idx_median', 'rise_50_interp_median', 'fall_50_interp_median'):
            df[key + '_minus_RX1'] = df[key] - ref[key]
    return df, selected


def _zoom_ylim(ax, curves, sl):
    vals = [c[sl] for c in curves if np.any(np.isfinite(c[sl]))]
    if not vals:
        return
    lo = min(np.nanmin(v) for v in vals)
    hi = max(np.nanmax(v) for v in vals)
    pad = 0.05 * (hi - lo) if hi > lo else 0.05 * max(abs(hi), 1e-12)
    ax.set_ylim(lo - pad, hi + pad)


def _fig_amplitude(path, x, shown, channels, env_rep, valid, sel_edges, rl, mode_used, sd=None):
    n = x.size
    k = max(5, min(20, n // 2))
    w = _zoom_half()                                           # half width of the edge zooms (samples)
    cols = _colors(len(shown))
    fig = plt.figure(figsize=(11.5, 7.5), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1])
    axm = fig.add_subplot(gs[0, :])
    axs = fig.add_subplot(gs[1, 0])
    axe = fig.add_subplot(gs[1, 1])
    curves = []
    for j, c in enumerate(shown):
        ic = channels.index(c)
        y = env_rep[ic]
        curves.append(y)
        kw = dict(color='k' if c == 1 else cols[j], lw=2.0 if c == 1 else 1.1, ls='-' if valid[ic] else '--')
        axm.plot(x, y, label='RX%i%s' % (c, '' if valid[ic] else ' (no valid RF)'), **kw)
        if sd is not None and valid[ic]:                       # +-1 std across repetitions
            for a in (axm, axs, axe):
                a.fill_between(x, y - sd[ic], y + sd[ic], color=kw['color'], alpha=0.15, lw=0)
        axs.plot(x, y, **kw)
        axe.plot(x, y, **kw)
    rises = [e['rise_50'] for e in sel_edges if e is not None and np.isfinite(e['rise_50'])]
    falls = [e['fall_50'] for e in sel_edges if e is not None and np.isfinite(e['fall_50'])]
    if rises:                                                  # zoom follows the edges when they exist in the window
        cr = int(round(np.median(rises)))
        lo_s, hi_s, t_s = max(0, cr - w), min(n - 1, cr + w), 'Rising edge (zoom, +-%0.0f us)' % (_zoom_half() * _dt()) if _DT[0] else 'Rising edge (zoom)'
    else:
        lo_s, hi_s, t_s = 0, k - 1, 'First %i samples' % k
    if falls:
        cf = int(round(np.median(falls)))
        lo_e, hi_e, t_e = max(0, cf - w), min(n - 1, cf + w), 'Falling edge (zoom, +-%0.0f us)' % (_zoom_half() * _dt()) if _DT[0] else 'Falling edge (zoom)'
    else:
        lo_e, hi_e, t_e = n - k, n - 1, 'Last %i samples' % k
    axs.set_xlim(lo_s * _dt(), hi_s * _dt())
    axe.set_xlim(lo_e * _dt(), hi_e * _dt())
    _zoom_ylim(axs, curves, slice(lo_s, hi_s + 1))
    _zoom_ylim(axe, curves, slice(lo_e, hi_e + 1))
    axm.set(xlabel=_xlab(), ylabel='Amplitude (a.u.)',
            title='Amplitude, %s%s (no normalisation)' % (rl, ', band = +-1 std across repetitions' if sd is not None else ''))
    axs.set(xlabel=_xlab(), ylabel='Amplitude (a.u.)', title=t_s)
    axe.set(xlabel=_xlab(), ylabel='Amplitude (a.u.)', title=t_e)
    axm.legend(ncol=2, fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    for a in (axm, axs, axe):
        a.grid(alpha=0.3)
    fig.suptitle('Mode %s. Bottom: zoom on the rising and falling edges; if the curves overlap there, the channels '
                 'rise/fall at the same sample index. %s' % (mode_used, _AXIS_NOTE), fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_relative_amplitude(path, x, labels, ratio, diff, thr, rl, mode_used, colors):
    """Amplitude of each channel against RX1: ratio where both are valid, and normalised difference everywhere."""
    fig = plt.figure(figsize=(11.5, 7), layout='constrained')
    gs = fig.add_gridspec(2, 1)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for j, lab in enumerate(labels):
        ax1.plot(x, ratio[j], lw=1.2, color=colors[j], label=lab)
        ax2.plot(x, diff[j], lw=1.2, color=colors[j])
    ax2.axhline(0, color='k', lw=0.8)
    ax1.set(ylabel='|Xn| / |X1|', title='Amplitude ratio vs RX1, %s (masked where RX1 or RXn is below '
                                        '%0.2f x its own plateau)' % (rl, thr))
    ax1.set_ylim(0.5, 1.5)                                     # clipped: the ratio is not defined on the edges
    ax2.set(xlabel=_xlab(), ylabel='(|Xn| - |X1|) / plateau of RX1',
            title='Normalised amplitude difference (not masked)')
    ax1.legend(ncol=2, fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    for a in (ax1, ax2):
        a.grid(alpha=0.3)
    fig.suptitle('Mode %s. Top: ratio clipped to 0.5-1.5. Bottom: a level mismatch is an offset; an edge at a different '
                 'sample index is a bump at that edge. %s' % (mode_used, _AXIS_NOTE), fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _zoom_windows(sel_edges, n, w=None):
    w = _zoom_half() if w is None else w
    k = max(5, min(20, n // 2))
    rises = [e['rise_50'] for e in sel_edges if e is not None and np.isfinite(e['rise_50'])]
    falls = [e['fall_50'] for e in sel_edges if e is not None and np.isfinite(e['fall_50'])]
    if rises:
        cr = int(round(np.median(rises)))
        a = (max(0, cr - w), min(n - 1, cr + w))
    else:
        a = (0, k - 1)
    if falls:
        cf = int(round(np.median(falls)))
        b = (max(0, cf - w), min(n - 1, cf + w))
    else:
        b = (n - k, n - 1)
    return a + b


def _fig_phase_generic(path, x, labels, unwrapped, wrapped, rratio, thr, title, ylabel_unw, ylabel_wrap,
                       ratio_label, note, colors, colors_bold=None, zoom=None):
    """Three stacked panels: (a) unwrapped for visualisation, (b) original circular phase, (c) amplitude ratio."""
    fig = plt.figure(figsize=(11.5, 10.5 if zoom else 8.5), layout='constrained')
    if zoom:
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 2, 3, 1.3])
        ax1, ax2, ax3 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[2, :]), fig.add_subplot(gs[3, :])
        axs, axe = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 1.3])
        ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))
    for j, lab in enumerate(labels):
        kw = dict(color=colors[j], lw=1.2, label=lab)
        ax1.plot(x, unwrapped[j], **kw)
        ax2.plot(x, wrapped[j], marker='.', ms=3, ls='none', color=colors[j])
        ax3.plot(x, rratio[j], lw=1.0, color=colors[j])
    ax3.axhline(thr, color='r', ls='--', lw=1)
    if zoom:
        for a, (lo_, hi_), t_ in ((axs, zoom[:2], 'Rising edge (zoom, +-%0.0f us)' % (_zoom_half() * _dt()) if _DT[0] else 'Rising edge (zoom)'), (axe, zoom[2:], 'Falling edge (zoom, +-%0.0f us)' % (_zoom_half() * _dt()) if _DT[0] else 'Falling edge (zoom)')):
            for j in range(len(labels)):
                a.plot(x, unwrapped[j], lw=1.2, color=colors[j], marker='.', ms=4)
            a.set_xlim(lo_ * _dt(), hi_ * _dt())
            _zoom_ylim(a, unwrapped, slice(lo_, hi_ + 1))
            a.set(xlabel=_xlab(), ylabel=ylabel_unw, title=t_)
            a.grid(alpha=0.3)
    ax1.set(ylabel=ylabel_unw, title=title + ' - (a) unwrapped, visualisation only (per contiguous valid run)')
    ax2.set(ylabel=ylabel_wrap, ylim=(-185, 185), title='(b) original circular phase')
    ax3.set(xlabel=_xlab(), ylabel=ratio_label, title='(c) validity: samples below the red line are masked')
    ax1.legend(ncol=2, fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    for a in (ax1, ax2, ax3):
        a.grid(alpha=0.3)
    fig.suptitle(note + '. ' + _AXIS_NOTE, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_message(path, text):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    ax.text(0.5, 0.5, text, ha='center', va='center', wrap=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_repetition(path, x, c, cdata_c, valid_c, thr, mode_used, sample_snr_min=5.0):
    """All repetitions of one channel: amplitude and phase overlays plus (complementary) dispersion."""
    n_rep, n = cdata_c.shape
    env = np.abs(cdata_c)
    fig = plt.figure(figsize=(12, 8), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])
    axa, axad, axp, axpd = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
    # amplitude
    axa.plot(x, env.T, color='C0', alpha=0.2, lw=0.7)
    mean_a = env.mean(axis=0)
    std_a = env.std(axis=0, ddof=1) if n_rep > 1 else np.zeros(n)
    axa.plot(x, mean_a, color='k', lw=1.5, label='mean')
    axa.fill_between(x, mean_a - std_a, mean_a + std_a, color='k', alpha=0.2, label='+/- 1 std')
    axa.set(ylabel='Amplitude (a.u.)', title='RX%i amplitude, all %i repetitions' % (c, n_rep))
    axa.legend(fontsize=7)
    with np.errstate(divide='ignore', invalid='ignore'):
        axad.plot(x, std_a / mean_a * 100, color='C0')
    axad.set(ylabel='Amplitude CV across repetitions (%)', title='Dispersion')
    if valid_c:
        plat = _plateau(env)
        with np.errstate(divide='ignore', invalid='ignore'):
            ok = env >= thr * plat[:, None]
        if n_rep >= 3:
            ok = ok & (_sample_snr(env) >= sample_snr_min)[None, :]
        ok = np.array([_drop_short_runs(r) for r in ok])
        ph = np.angle(cdata_c)
        z = np.where(ok, np.exp(1j * ph), 0)
        cnt = ok.sum(axis=0)
        good = cnt >= 0.5 * n_rep          # statistics across repetitions only where >= 50 % of them are valid
        zs = z.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.clip(np.abs(zs) / cnt, 1e-12, 1.0)
        cmean = np.angle(zs)
        cstd = np.degrees(np.sqrt(-2 * np.log(R)))
        cstd[~good] = np.nan
        mean_unw = _unwrap_deg_masked(cmean, good)
        dev = np.degrees(np.angle(np.exp(1j * (ph - cmean[None, :]))))
        disp = np.where(ok & good[None, :], mean_unw[None, :] + dev, np.nan)
        axp.plot(x, disp.T, color='C1', alpha=0.2, lw=0.7)
        axp.plot(x, mean_unw, color='k', lw=1.5, label='circular mean')
        axp.fill_between(x, mean_unw - cstd, mean_unw + cstd, color='k', alpha=0.2, label='+/- 1 circular std')
        axp.set(ylabel='Phase (deg)', title='RX%i phase, all %i repetitions' % (c, n_rep))
        axp.legend(fontsize=7)
        axpd.plot(x, cstd, color='C1')
        axpd.set(ylabel='Circular std across repetitions (deg)', title='Dispersion')
    else:
        for a in (axp, axpd):
            a.text(0.5, 0.5, 'RX%i has no valid RF signal:\nphase not shown' % c, ha='center', va='center',
                   transform=a.transAxes)
    for a in (axa, axad, axp, axpd):
        a.set_xlabel(_xlab())
        a.grid(alpha=0.3)
    fig.suptitle('Repeatability of the %i bursts, mode %s. Overlays are the individual repetitions, not an average.\n'
                 'Phase: each sample is drawn around the circular mean of that sample (mean unwrapped for '
                 'visualisation), only where >= 50%% of the repetitions are valid.\n%s'
                 % (n_rep, mode_used, _AXIS_NOTE), fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _parabolic_lag(a, b):
    """Lag (samples, sub-sample by parabolic peak interpolation) at which b best matches a (normalised xcorr)."""
    a = a - a.mean()
    b = b - b.mean()
    na = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if na == 0:
        return np.nan
    n = a.size
    lags = np.arange(-(n // 4), n // 4 + 1)
    cc = np.array([np.sum(a[max(0, -l):n - max(0, l)] * b[max(0, l):n - max(0, -l)]) for l in lags]) / na
    i = int(np.argmax(cc))
    if 0 < i < len(cc) - 1 and (cc[i - 1] - 2 * cc[i] + cc[i + 1]) != 0:
        d = 0.5 * (cc[i - 1] - cc[i + 1]) / (cc[i - 1] - 2 * cc[i] + cc[i + 1])
        return float(lags[i] + d)
    return float(lags[i])


def _sync_report(cdata, env, env_rep, plateau, valid, channels, cross, ok, thr_ok=0.25):
    """Console diagnostics of temporal coincidence of every valid channel against RX1 (units: sample index)."""
    n_ch, n_rep, n_pts = env.shape
    i1 = channels.index(1)
    if not valid[i1]:
        print("Sync report not available: RX1 has no valid RF signal")
        return
    m1 = [_edge_metrics(env[i1, r], plateau[i1, r]) for r in range(n_rep)]
    print("")
    print("=== Temporal coincidence vs RX1 (units: samples%s; resolution = 1 sample; mean of %i repetitions) ===" %
          ((' = %0.3f us each' % _DT[0]) if _DT[0] else '', n_rep))
    print("Differences are observed 50 percent crossing indices (interpolated, sub-sample = statistical estimate) and the")
    print("lag of the mean amplitude curve (cross-correlation). They are NOT ADC skew or physical delays.")
    print("%-6s %-22s %-22s %-14s %-10s %s" % ('RXn', 'rise-RX1 (med +- std)', 'fall-RX1 (med +- std)',
                                              'xcorr lag', 'amp/RX1', 'verdict'))
    for ic, c in enumerate(channels):
        if c == 1 or not valid[ic]:
            continue
        mn = [_edge_metrics(env[ic, r], plateau[ic, r]) for r in range(n_rep)]
        txt, worst = [], 0.0
        for key in ('rise_50_interp', 'fall_50_interp'):
            d = np.array([a[key] - b[key] for a, b in zip(mn, m1)])
            d = d[np.isfinite(d)]
            if d.size:
                md, sd_ = np.median(d), (np.std(d, ddof=1) if d.size > 1 else 0.0)
                txt.append(('%+0.3f us +- %0.3f' % (md * _DT[0], sd_ * _DT[0])) if _DT[0] else
                           '%+0.2f +- %0.2f smp' % (md, sd_))
                worst = max(worst, abs(np.median(d)))
            else:
                txt.append('no edge')
        lag = _parabolic_lag(env_rep[i1], env_rep[ic])
        if np.isfinite(lag):
            worst = max(worst, abs(lag))
        p1, pn = plateau[i1].mean(), plateau[ic].mean()
        if not np.isfinite(worst) or (txt[0] == 'no edge' and txt[1] == 'no edge'):
            verdict = 'no edges in window: cannot evaluate'
        elif worst <= thr_ok:
            verdict = 'coincident (within %0.2f us)' % (thr_ok * _DT[0]) if _DT[0] else 'coincident (within %0.2f sample)' % thr_ok
        elif worst <= 1.0:
            verdict = 'offset below 1 sample (%0.2f us)' % _DT[0] if _DT[0] else 'offset below 1 sample'
        else:
            verdict = 'OFFSET of %0.2f us (%0.1f samples)' % (worst * _DT[0], worst) if _DT[0] else 'OFFSET of %0.1f samples' % worst
        print("RX%-4i %-22s %-22s %-14s %-10.4f %s" % (c, txt[0], txt[1],
              ('%+0.3f us' % (lag * _DT[0]) if _DT[0] else '%+0.2f smp' % lag) if np.isfinite(lag) else 'n/a',
              pn / p1, verdict))
        sel = ok[ic] & ok[i1]
        if sel.sum() >= 3:
            pm = np.mean(cross(ic)[sel])
            per = np.array([np.mean((cdata[ic, r] * np.conj(cdata[i1, r]))[sel]) for r in range(n_rep)])
            z = np.mean(per / np.abs(per))
            print("       phase vs RX1 on common valid samples: mean %+0.3f deg, circular std over repetitions %0.4f deg"
                  % (np.degrees(np.angle(pm)), np.degrees(np.sqrt(-2 * np.log(min(max(abs(z), 1e-12), 1.0))))))
    print("Rule of thumb: a difference up to %0.2f sample counts as coincident at this resolution; a larger, stable value over the" % thr_ok)
    print("repetitions (small std) means a systematic offset; a large std means the edge is too noisy to judge.")
    print("Phase on the edges is masked; use the plateau values for the phase offset.")


def _stability_all(data, channels, lo, hi, stem, valid_channels, gui):
    """
    1:12 version of the stability analysis: every channel with valid RF against RX1, per repetition, using the
    same window and definitions as the RX1-vs-RXn analysis (complex mean over samples lo:hi, ratio = |Xn|/|X1|,
    dphi = arg[Xn conj(X1)]). Saves <stem>_stability_all.csv and <stem>_mimo_12ch_stability.png, prints a table.
    """
    i1 = channels.index(1)
    chs = [c for c in channels if c != 1 and c in valid_channels]
    if 1 not in valid_channels or not chs:
        print("1:12 stability not available: RX1 or the other channels have no valid RF signal")
        return []
    n_rep = data.shape[1]
    x1 = np.mean(data[i1][:, lo:hi], axis=1)
    rows, ratios, dphis = [], [], []
    for c in chs:
        xn = np.mean(data[channels.index(c)][:, lo:hi], axis=1)
        ratio = np.abs(xn) / np.abs(x1)
        dphi = np.angle(xn * np.conj(x1))
        m, sd, cv = _lin_stats(ratio)
        cm, cs, _ = _circ_stats(dphi)
        rows.append({'rx_channel': c, 'n_reps': n_rep, 'window_samples': '%i-%i' % (lo, hi - 1),
                     'amp_rx1_mean': float(np.mean(np.abs(x1))), 'amp_rxn_mean': float(np.mean(np.abs(xn))),
                     'ratio_mean': m, 'ratio_std': sd, 'ratio_cv_percent': cv,
                     'dphi_mean_deg': cm, 'dphi_circ_std_deg': cs})
        ratios.append(ratio)
        dphis.append(dphi)
    df = pd.DataFrame(rows)
    df.to_csv(stem + '_stability_all.csv', index=False)

    print("")
    print("=== 1:12 stability vs RX1 (%i repetitions, window samples %i-%i) ===" % (n_rep, lo, hi - 1))
    print("%-6s %-12s %-12s %-12s %-14s %s" % ('RXn', 'ratio mean', 'ratio std', 'CV (%)', 'dphi mean (deg)',
                                               'dphi circ std (deg)'))
    for r in rows:
        print("RX%-4i %-12.5f %-12.2e %-12.2e %-+14.3f %0.2e" % (r['rx_channel'], r['ratio_mean'], r['ratio_std'],
                                                                 r['ratio_cv_percent'], r['dphi_mean_deg'],
                                                                 r['dphi_circ_std_deg']))
    print("dphi mean is a fixed offset between boards (not an error by itself); the circular std is the stability.")

    colors = _colors(len(chs))
    rep_x = np.arange(1, n_rep + 1)
    ratio_dev = [(r / np.mean(r) - 1) * 100 for r in ratios]
    dphi_dev = [np.degrees(np.angle(np.exp(1j * (d - np.radians(row['dphi_mean_deg']))))) for d, row in
                zip(dphis, rows)]
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8), layout='constrained')
    xt = np.arange(len(chs))
    lab = ['RX%i' % c for c in chs]
    ax[0, 0].errorbar(xt, df['ratio_mean'], yerr=df['ratio_std'], fmt='o', capsize=3, color='C0')
    ax[0, 0].axhline(1, color='k', lw=0.8, ls='--')
    ax[0, 0].set(xticks=xt, xticklabels=lab, ylabel='|Xn| / |X1|', title='Amplitude ratio vs RX1 (mean +- std)')
    ax[0, 1].errorbar(xt, df['dphi_mean_deg'], yerr=df['dphi_circ_std_deg'], fmt='o', capsize=3, color='C1')
    ax[0, 1].set(xticks=xt, xticklabels=lab, ylabel='Phase difference (deg)',
                 title='Phase difference vs RX1 (mean +- circular std)')
    for j in range(len(chs)):
        ax[1, 0].plot(rep_x, ratio_dev[j], lw=0.8, color=colors[j], label=lab[j])
        ax[1, 1].plot(rep_x, dphi_dev[j], lw=0.8, color=colors[j])
    ax[1, 0].set(xlabel='Repetition', ylabel='Ratio deviation from mean (%)', title='Amplitude stability per repetition')
    ax[1, 1].set(xlabel='Repetition', ylabel='dphi deviation from mean (deg)', title='Phase stability per repetition')
    ax[1, 0].legend(ncol=2, fontsize=7)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    for a in (ax[0, 0], ax[0, 1]):
        a.tick_params(axis='x', rotation=45)
    fig.suptitle('1:12 stability against RX1 (window samples %i-%i). A fixed phase offset between boards is expected; '
                 'stability is the scatter around it.' % (lo, hi - 1), fontsize=9)
    fig.savefig(stem + '_mimo_12ch_stability.png', dpi=150)
    plt.close(fig)
    print("Saved: %s_stability_all.csv and %s_mimo_12ch_stability.png" % (os.path.basename(stem),
                                                                         os.path.basename(stem)))
    results = []
    if gui:
        results.append({'widget': 'curve', 'xData': rep_x, 'yData': ratio_dev, 'xLabel': 'Repetition',
                        'yLabel': 'Amplitude ratio deviation (%)', 'title': 'Amplitude stability vs RX1, 1:12',
                        'legend': lab, 'row': 0, 'col': 3})
        results.append({'widget': 'curve', 'xData': rep_x, 'yData': dphi_dev, 'xLabel': 'Repetition',
                        'yLabel': 'Phase difference deviation (deg)', 'title': 'Phase stability vs RX1, 1:12',
                        'legend': lab, 'row': 1, 'col': 3})
    return results


def _temporal_analysis(cdata, channels, stem, rx_n, mode, repetition, channel, phase_amp_thr, snr_min, gui,
                       sample_snr_min=5.0, compact=False):
    """Figures A-D (+ edge table). Returns (extra output_dict entries, extra MaRGE widgets)."""
    n_ch, n_rep, n_pts = cdata.shape
    if len(channels) != n_ch:
        raise ValueError("'channels' has %i entries but data has %i channels" % (len(channels), n_ch))
    if 1 not in channels:
        raise ValueError("RX1 (reference) is not in 'channels'")
    if mode not in ('auto', '1:2', '1:12'):
        raise ValueError("mode must be 'auto', '1:2' or '1:12'")
    avg = (isinstance(repetition, str) and repetition.strip().lower() == 'mean') or \
          (not isinstance(repetition, str) and int(repetition) == 0)
    if not avg and not 1 <= int(repetition) <= n_rep:
        raise ValueError("repetition must be 'mean' (or 0) or between 1 and %i" % n_rep)
    rep = 0 if avg else int(repetition) - 1
    rl = ('mean of %i repetitions' % n_rep) if avg else ('repetition %i of %i' % (rep + 1, n_rep))
    if channel is None:
        channel = 1
    if channel not in channels:
        print("WARNING: channel %s is not available for the repeatability figure, using RX1" % channel)
        channel = 1
    i1 = channels.index(1)
    x = np.arange(n_pts) * _dt()

    valid, snr = _valid_rf_channels(cdata, snr_min)
    mode_used = ('1:12' if int(valid.sum()) > 2 else '1:2') if mode == 'auto' else mode
    shown = [c for c in dict.fromkeys([1, rx_n]) if c in channels] if mode_used == '1:2' else list(channels)

    print("Temporal analysis (sample index axis). Mode: %s%s, %s" %
          (mode_used, ' (auto)' if mode == 'auto' else '', rl))
    print("Coherent SNR (valid RF if >= %0.1f): %s" % (snr_min, ', '.join(
        'RX%i %0.0f%s' % (c, snr[i], '' if valid[i] else ' (no RF)') for i, c in enumerate(channels))))
    if mode_used == '1:2' and int(valid.sum()) > 2:
        print("NOTE: more than 2 channels have RF but mode 1:2 was requested: only RX1 and RX%i are shown" % rx_n)
    if mode_used == '1:12' and not valid.all():
        print("NOTE: channels without valid RF are drawn only in the amplitude figure: %s" %
              ', '.join('RX%i' % c for i, c in enumerate(channels) if not valid[i]))

    env = np.abs(cdata)                                   # (channels, repetitions, points)
    plateau = _plateau(env)                               # (channels, repetitions)
    if avg:   # curves shown = mean over all repetitions (amplitude: mean of |x|; phase: angle of the complex mean)
        env_rep = env.mean(axis=1)
        sd_rep = env.std(axis=1, ddof=1) if n_rep > 1 else np.zeros_like(env_rep)
        ph_rep = np.angle(cdata.mean(axis=1))
        plateau_sel = _plateau(env_rep[:, None, :])[:, 0]
    else:
        env_rep = env[:, rep, :]
        sd_rep = None
        ph_rep = np.angle(cdata[:, rep, :])
        plateau_sel = plateau[:, rep]

    def _cross(ic):                                       # X_n * conj(X_1) of the curve shown
        if avg:
            return np.mean(cdata[ic] * np.conj(cdata[i1]), axis=0)
        return cdata[ic, rep, :] * np.conj(cdata[i1, rep, :])

    with np.errstate(divide='ignore', invalid='ignore'):
        rratio = env_rep / plateau_sel[:, None]           # amplitude / own plateau
    ok = (rratio >= phase_amp_thr) & valid[:, None] & np.isfinite(rratio)
    if n_rep >= 3:                                        # noise-only samples have mean/std of ~1.9 (Rayleigh)
        ok = ok & (_sample_snr(env) >= sample_snr_min)
    ok = np.array([_drop_short_runs(ok[i]) for i in range(n_ch)])

    edge_df, sel_edges = _edge_table(env, plateau, valid, channels, rep,
                                     sel_curve=(env_rep, plateau_sel) if avg else None)
    edge_df.to_csv(stem + '_mimo_edges.csv', index=False)

    # Figure A: amplitude
    _fig_amplitude(stem + '_mimo_12ch_amplitude.png', x, shown, channels, env_rep, valid, sel_edges, rl,
                   mode_used, sd_rep)

    # Figure B: phase (valid channels only)
    sv = [c for c in shown if valid[channels.index(c)]]
    gui_results = []
    cols_all = dict(zip(channels, _colors(len(channels))))
    cols_all[1] = np.array([0, 0, 0, 1.0])
    if sv:
        unw = [_unwrap_deg_masked(ph_rep[channels.index(c)], ok[channels.index(c)]) for c in sv]
        wrp = [_wrap_deg_masked(ph_rep[channels.index(c)], ok[channels.index(c)]) for c in sv]
        rr = [rratio[channels.index(c)] for c in sv]
        _fig_phase_generic(stem + '_mimo_12ch_phase.png', x, ['RX%i' % c for c in sv], unw, wrp, rr, phase_amp_thr,
                           'Phase, %s' % rl, 'Phase (deg)', 'Phase (deg)',
                           '|X| / plateau', 'Mode %s, phase masked where |X| < %0.2f x own plateau' %
                           (mode_used, phase_amp_thr), [cols_all[c] for c in sv],
                           zoom=_zoom_windows(sel_edges, n_pts))
    else:
        unw = []
        _fig_message(stem + '_mimo_12ch_phase.png', 'No channel with valid RF signal: phase not shown')

    # Figure C: phase difference relative to RX1
    sc = [c for c in sv if c != 1]
    dphi_unw = []
    dphi_valid = []
    if valid[i1] and sc:
        dunw, dwrp, drr = [], [], []
        for c in sc:
            ic = channels.index(c)
            dphi = np.angle(_cross(ic))
            okc = ok[ic] & ok[i1]
            dunw.append(_unwrap_deg_masked(dphi, okc))
            dwrp.append(_wrap_deg_masked(dphi, okc))
            drr.append(np.minimum(rratio[ic], rratio[i1]))
        dphi_unw = dunw
        dphi_valid = dunw
        _fig_phase_generic(stem + '_mimo_12ch_relative_phase.png', x, ['RX%i-RX1' % c for c in sc], dunw, dwrp, drr,
                           phase_amp_thr, 'Phase difference vs RX1, %s' % rl,
                           'Phase difference (deg)', 'Phase difference (deg)', 'min(|X|/plateau)',
                           'Mode %s, masked where RX1 or RXn is below %0.2f x its own plateau' % (mode_used,
                                                                                                phase_amp_thr),
                           [cols_all[c] for c in sc], zoom=_zoom_windows(sel_edges, n_pts))
    else:
        _fig_message(stem + '_mimo_12ch_relative_phase.png',
                     'Relative phase not shown: RX1 or the other channel has no valid RF signal')

    # Figure E: amplitude relative to RX1 (ratio where both are valid, normalised difference everywhere)
    if valid[i1] and sc:
        rat, dif = [], []
        for c in sc:
            ic = channels.index(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                rat.append(np.where(ok[ic] & ok[i1], env_rep[ic] / env_rep[i1], np.nan))
                dif.append((env_rep[ic] - env_rep[i1]) / plateau_sel[i1])
        _fig_relative_amplitude(stem + '_mimo_12ch_relative_amplitude.png', x, ['RX%i vs RX1' % c for c in sc], rat,
                                dif, phase_amp_thr, rl, mode_used, [cols_all[c] for c in sc])
    else:
        _fig_message(stem + '_mimo_12ch_relative_amplitude.png',
                     'Relative amplitude not shown: RX1 or the other channel has no valid RF signal')

    # Figure D: repeatability of the bursts for one channel
    _fig_repetition(stem + '_mimo_12ch_repetition_stability.png', x, channel, cdata[channels.index(channel)],
                    bool(valid[channels.index(channel)]), phase_amp_thr, mode_used, sample_snr_min)

    _sync_report(cdata, env, env_rep, plateau, valid & np.array([c in shown for c in channels]), channels, _cross, ok)
    print("Saved: %s_mimo_12ch_amplitude/phase/relative_phase/relative_amplitude/repetition_stability.png and "
          "%s_mimo_edges.csv" % (os.path.basename(stem), os.path.basename(stem)))
    for ic, c in enumerate(channels):
        e = sel_edges[ic]
        if c not in shown:
            continue
        if e is not None and (np.isfinite(e['rise_50']) or np.isfinite(e['fall_50'])):
            print("RX%i edges (sample index, 50%%): rise %s, fall %s" % (c, e['rise_50'], e['fall_50']))

    if 'rise_50_interp_median_minus_RX1' in edge_df.columns:
        for _, row_ in edge_df[edge_df['valid_rf'] == True].iterrows():          # noqa: E712
            if row_['rx_channel'] in shown and row_['rx_channel'] != 1 and np.isfinite(row_['rise_50_interp_median_minus_RX1']):
                print("RX%i - RX1, observed 50%% crossing index (median over repetitions, interpolated): rise %+0.2f, "
                      "fall %+0.2f samples" % (row_['rx_channel'], row_['rise_50_interp_median_minus_RX1'],
                                               row_['fall_50_interp_median_minus_RX1']))

    # MaRGE widgets: same 'curve' structure as the existing ones, laid out in a 3 x 3 grid so that nothing is squeezed
    # (column 0 = the two original stability plots):
    #   row 0: amplitude (mean)            | phase (valid samples only)
    #   row 1: amplitude rising edge zoom  | amplitude falling edge zoom
    #   row 2: phase rising edge zoom      | phase falling edge zoom      (columns 1 and 2)
    # Phase curves are never sent with masked (NaN) samples: only the samples valid in every curve are shown.
    if gui and compact:
        # Compact viewer: 4 plots in a 2 x 2 grid. Row 0: amplitude (mean of the repetitions) and phase difference vs RX1
        # (valid samples only). Row 1: zoom on the rising and on the falling edge of the amplitude.
        def put(row, col, xd, yd, ylab, title, legend):
            gui_results.append({'widget': 'curve', 'xData': xd, 'yData': yd, 'xLabel': _xlab(), 'yLabel': ylab,
                                'title': title, 'legend': legend, 'row': row, 'col': col})

        put(0, 0, x, [env_rep[channels.index(c)] for c in shown], 'Amplitude (a.u.)',
            'Amplitude, %s (mode %s)' % (rl, mode_used), ['RX%i' % c for c in shown])
        if sc and dphi_valid:
            good = np.all([np.isfinite(u) for u in dphi_valid], axis=0)
            if good.sum() >= 3:
                put(0, 1, x[good], [u[good] for u in dphi_valid], 'Phase difference (deg)',
                    'Phase difference vs RX1 (valid samples only), %s' % rl, ['RX%i-RX1' % c for c in sc])
        if any(e is not None and (np.isfinite(e['rise_50']) or np.isfinite(e['fall_50'])) for e in sel_edges):
            lo_s, hi_s, lo_e, hi_e = _zoom_windows(sel_edges, n_pts)
            for col_g, name, (l_, h_) in ((0, 'rising edge', (lo_s, hi_s)), (1, 'falling edge', (lo_e, hi_e))):
                sl = slice(l_, h_ + 1)
                put(1, col_g, x[sl], [env_rep[channels.index(c)][sl] for c in shown], 'Amplitude (a.u.)',
                    'Amplitude, %s (zoom)' % name, ['RX%i' % c for c in shown])
    elif gui:
        def add_widget(row, col, xd, yd, ylab, title, legend):
            gui_results.append({'widget': 'curve', 'xData': xd, 'yData': yd, 'xLabel': _xlab(), 'yLabel': ylab,
                                'title': title, 'legend': legend, 'row': row, 'col': col})

        add_widget(0, 1, x, [env_rep[channels.index(c)] for c in shown], 'Amplitude (a.u.)',
                   'Amplitude, %s (mode %s)' % (rl, mode_used), ['RX%i' % c for c in shown])
        if sv:
            good = np.all([np.isfinite(u) for u in unw], axis=0)
            if good.sum() >= 3:
                add_widget(0, 2, x[good], [u[good] for u in unw], 'Phase (deg)',
                           'Phase (unwrapped, valid samples only), %s' % rl, ['RX%i' % c for c in sv])
            else:
                print("NOTE: phase not sent to the MaRGE viewer (no valid samples); see the PNG")
        has_edges = any(e is not None and (np.isfinite(e['rise_50']) or np.isfinite(e['fall_50'])) for e in sel_edges)
        if has_edges:
            lo_s, hi_s, lo_e, hi_e = _zoom_windows(sel_edges, n_pts)
            for col_g, name, (l_, h_) in ((1, 'rising edge', (lo_s, hi_s)), (2, 'falling edge', (lo_e, hi_e))):
                sl = slice(l_, h_ + 1)
                add_widget(1, col_g, x[sl], [env_rep[channels.index(c)][sl] for c in shown], 'Amplitude (a.u.)',
                           'Amplitude, %s (zoom)' % name, ['RX%i' % c for c in shown])
                if sv:
                    good = np.all([np.isfinite(u[sl]) for u in unw], axis=0)
                    if good.sum() >= 3:
                        add_widget(2, col_g, x[sl][good], [u[sl][good] for u in unw], 'Phase (deg)',
                                   'Phase, %s (zoom, valid samples only)' % name, ['RX%i' % c for c in sv])
        if dphi_unw and all(np.all(np.isfinite(u)) for u in dphi_unw):
            add_widget(2, 0, x, dphi_unw, 'Phase difference (deg)',
                       'Phase difference vs RX1 (unwrapped), %s' % rl, ['RX%i-RX1' % c for c in sc])
        else:
            print("NOTE: relative phase not sent to the MaRGE viewer (masked samples or no valid channel); see the PNG")

    extra = {'temporal_mode': mode_used, 'valid_rf_channels': [int(c) for i, c in enumerate(channels) if valid[i]],
             'coherent_snr': snr}
    return extra, gui_results


def MimoCoherence(raw_data_path=None, trim=0.1, mode='auto', repetition='mean', channel=None,
                  phase_amp_thr=0.1, snr_min=10.0, gui_temporal=True, sample_snr_min=5.0, zoom_us=None, gui_plots='compact'):
    """
    Per repetition, estimates amplitude and phase of RX1 and RXn (complex mean over the central part of the
    acquisition window, edges trimmed by `trim` on each side), then computes
        dphi = arg[Xn * conj(X1)],  ratio = |Xn| / |X1|
    and saves: <name>_reps.csv, <name>_summary.csv and <name>_plots.png next to the .mat file.

    Added (does not change the results above): temporal analysis inside the burst, using the same data_decimated
    array. Saves <name>_mimo_12ch_amplitude.png, _phase.png, _relative_phase.png, _repetition_stability.png and
    <name>_mimo_12ch_relative_amplitude.png and <name>_mimo_edges.csv. Horizontal axis = sample index (not absolute
    time, not ADC skew). Edges only exist in the data if the sequence was run with burstTime > 0.
      mode          'auto' (default: follows 'Compare RX1 with' in the sequence: 0 = 1:12, 2-12 = 1:2), '1:2' or '1:12'
      repetition    'mean' (default, or 0): mean over all repetitions (amplitude = mean |x|, band = +-1 std;
                    phase = angle of the complex mean). An integer 1..N shows that single repetition (1-based)
      channel       channel shown in the repeatability figure D (default RX1)
      zoom_us       half width of the rising/falling edge zooms in us (default: 12 samples, at least 10 us)
      phase_amp_thr phase is masked where |X| is below this fraction of the channel's own plateau amplitude
      snr_min       minimum coherent SNR for a channel to count as having a valid RF signal
      (1:12 mode also saves <name>_stability_all.csv and _mimo_12ch_stability.png: every channel vs RX1)
      gui_plots     'compact' (default): 4 plots in the MaRGE viewer (amplitude, phase difference vs RX1, zoom on the
                    rising edge, zoom on the falling edge). 'full': the two original stability plots plus the
                    temporal plots. The CSV files and PNG figures are the same in both cases
      gui_temporal  False to keep only the two original MaRGE widgets
      sample_snr_min phase is also masked at samples whose mean(|x|)/std(|x|) over the repetitions is below this
                    (noise only gives ~1.9)
    """
    if raw_data_path is None:
        return None

    mat_data = sp.io.loadmat(raw_data_path)
    output_dict = {}
    dicom_meta_data = {}

    # Print inputs
    try:
        keys = mat_data['input_keys']
        strings = mat_data['input_strings']
        string = ""
        print("****Inputs****")
        for ii, key in enumerate(keys):
            string = string + f"{str(strings[ii]).strip()}: {np.squeeze(mat_data[str(key).strip()])}, "
        print(string)
    except Exception:
        pass
    print("****Outputs****")

    # Data: (channels, repetitions, points)
    data = np.asarray(mat_data['data_decimated'])
    channels = [int(c) for c in np.atleast_1d(np.squeeze(mat_data['channels']))]
    rx_sel = int(np.squeeze(mat_data['rxSplitter']))
    all_mode = rx_sel == 0                       # 0 = splitter 1:12 (all channels vs RX1), 2-12 = RX1 vs that RX
    rx_n = 2 if all_mode else rx_sel             # the original RX1-vs-RXn outputs use RX2 when 'all' is selected
    if mode == 'auto':
        mode = '1:12' if all_mode else '1:2'     # chosen in the sequence ('Compare RX1 with'); mode= overrides it
    if all_mode:
        print("Selection in the sequence: all channels vs RX1 (1:12). The original RX1-vs-RX2 outputs are also saved.")
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError("data_decimated must have shape (channels, repetitions, points)")
    n_ch, n_rep, n_pts = data.shape

    i1 = channels.index(1)
    i_n = channels.index(rx_n)
    lo = int(trim * n_pts)
    hi = n_pts - lo
    # Sequence run with a short burst (burstTime > 0, edges inside the window): average only over the central part of
    # the burst, never over the edges or the part of the window without RF. Without burstTime the window is unchanged.
    try:
        burst_us = float(np.squeeze(mat_data['burstTime_us']))
    except Exception:
        burst_us = 0.0
    if burst_us > 0:
        try:
            burst_samples = burst_us / float(np.squeeze(mat_data['samplingPeriod']))
        except Exception:
            burst_samples = 0.5 * n_pts                       # sampling period not stored: assume half of the window
        half = 0.3 * burst_samples                            # central 60 % of the burst
        lo = max(0, int(round(n_pts / 2 - half)))
        hi = min(n_pts, int(round(n_pts / 2 + half)))
        print("Short burst (%0.0f us): stability statistics use samples %i-%i (central part of the burst)" %
              (burst_us, lo, hi - 1))

    w1 = data[i1][:, lo:hi]
    wn = data[i_n][:, lo:hi]
    x1 = np.mean(w1, axis=1)  # (repetitions,)
    xn = np.mean(wn, axis=1)
    # Rms noise of the samples around their own mean, inside each window (per repetition)
    noise1 = np.sqrt(np.mean(np.abs(w1 - x1[:, None]) ** 2, axis=1))
    noisen = np.sqrt(np.mean(np.abs(wn - xn[:, None]) ** 2, axis=1))

    amp1, ampn = np.abs(x1), np.abs(xn)
    ph1, phn = np.angle(x1), np.angle(xn)
    dphi = np.angle(xn * np.conj(x1))  # rad
    ratio = ampn / amp1
    reps = np.arange(1, n_rep + 1)

    # Per-repetition CSV
    stem = os.path.splitext(raw_data_path)[0]
    df = pd.DataFrame({
        'rx_channel': rx_n, 'repetition': reps,
        'amp_rx1': amp1, 'amp_rxn': ampn, 'amp_ratio': ratio,
        'phase_rx1_deg': np.degrees(ph1), 'phase_rxn_deg': np.degrees(phn),
        'dphi_deg': np.degrees(dphi),
        'noise_rx1': noise1, 'noise_rxn': noisen,
    })
    df.to_csv(stem + '_reps.csv', index=False)

    # Summary CSV
    rows = []
    for name, x in [('amp_rx1', amp1), ('amp_rxn', ampn), ('amp_ratio', ratio)]:
        m, s, cv = _lin_stats(x)
        rows.append({'rx_channel': rx_n, 'quantity': name, 'mean': m, 'std': s, 'cv_percent': cv,
                     'circ_mean_deg': np.nan, 'circ_std_deg': np.nan})
    for name, x in [('phase_rx1', ph1), ('phase_rxn', phn), ('dphi', dphi)]:
        cm, cs, _ = _circ_stats(x)
        rows.append({'rx_channel': rx_n, 'quantity': name, 'mean': np.nan, 'std': np.nan, 'cv_percent': np.nan,
                     'circ_mean_deg': cm, 'circ_std_deg': cs})
    summary = pd.DataFrame(rows)
    summary.to_csv(stem + '_summary.csv', index=False)

    dphi_mean, dphi_std, dphi_r = _circ_stats(dphi)
    m_r, s_r, cv_r = _lin_stats(ratio)
    print("RX1 vs RX%i (%i repetitions)" % (rx_n, n_rep))
    print("Amplitude ratio: %0.5f +- %0.3e (CV %0.3e %%)" % (m_r, s_r, cv_r))
    print("Phase difference: %0.4f deg, circular std %0.3e deg" % (dphi_mean, dphi_std))
    # Rough expectation of the repetition-to-repetition scatter if the only source were the noise in the window
    n_eff = hi - lo
    snr1 = np.mean(amp1) / max(np.mean(noise1), 1e-300)
    snrn = np.mean(ampn) / max(np.mean(noisen), 1e-300)
    sig_phi = np.degrees(np.sqrt((1 / (2 * n_eff * snr1 ** 2)) + (1 / (2 * n_eff * snrn ** 2))))
    print("In-window SNR: RX1 %0.1f, RX%i %0.1f. Phase std expected from that noise alone (rough): %0.3e deg"
          % (snr1, rx_n, snrn, sig_phi))
    print("Distinct amplitude values in RX1 across repetitions: %i of %i" % (len(np.unique(amp1)), n_rep))

    # Plots
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(reps, amp1, label='RX1')
    ax[0, 0].plot(reps, ampn, label='RX%i' % rx_n)
    ax[0, 0].set(xlabel='Repetition', ylabel='Amplitude (a.u.)', title='Amplitude per repetition')
    ax[0, 0].legend()
    ax[0, 1].plot(reps, ratio)
    ax[0, 1].set(xlabel='Repetition', ylabel='|RX%i| / |RX1|' % rx_n, title='Amplitude ratio')
    ax[1, 0].plot(reps, np.degrees(dphi), '.-')
    ax[1, 0].set(xlabel='Repetition', ylabel='Phase difference (deg)', title='Phase difference per repetition')
    ax[1, 1].hist(np.degrees(dphi), bins=20)
    ax[1, 1].set(xlabel='Phase difference (deg)', ylabel='Counts', title='Phase difference histogram')
    fig.tight_layout()
    fig.savefig(stem + '_plots.png', dpi=150)
    plt.close(fig)

    output_dict['amp_rx1'] = amp1
    output_dict['amp_rxn'] = ampn
    output_dict['amp_ratio'] = ratio
    output_dict['dphi_deg'] = np.degrees(dphi)
    output_dict['summary'] = summary.to_dict('list')

    # GUI plots show deviations from the mean, otherwise a stable signal is just a flat line
    dev1 = (amp1 / np.mean(amp1) - 1) * 100
    devn = (ampn / np.mean(ampn) - 1) * 100
    dphi_dev = np.degrees(np.angle(np.exp(1j * (dphi - np.radians(dphi_mean)))))
    result_1 = {'widget': 'curve', 'xData': reps,
                'yData': [dev1, devn],
                'xLabel': 'Repetition', 'yLabel': 'Amplitude deviation from mean (%)',
                'title': 'Amplitude stability (mean RX1 %0.3g, RX%i %0.3g a.u.)' % (np.mean(amp1), rx_n, np.mean(ampn)),
                'legend': ['RX1', 'RX%i' % rx_n], 'row': 0, 'col': 0}
    result_2 = {'widget': 'curve', 'xData': reps,
                'yData': [dphi_dev],
                'xLabel': 'Repetition', 'yLabel': 'Phase difference deviation (deg)',
                'title': 'RX1 vs RX%i phase stability (mean %0.4f deg)' % (rx_n, dphi_mean),
                'legend': ['dphi - mean'], 'row': 1, 'col': 0}

    # ---- Temporal analysis inside the burst (added). A failure here must never affect the results above ----
    extra_results = []
    try:
        try:
            _DT[0] = float(np.squeeze(mat_data['samplingPeriod']))
            if not _DT[0] > 0:
                _DT[0] = None
        except Exception:
            _DT[0] = None
        if _DT[0] is None:                                    # fallback: bandwidth in kHz -> period in us
            try:
                _DT[0] = 1000.0 / float(np.squeeze(mat_data['bw']))
            except Exception:
                _DT[0] = None
        _ZOOM_US[0] = float(zoom_us) if zoom_us else None
        _set_axis_note()
        extra_dict, extra_results = _temporal_analysis(np.asarray(data, dtype=complex), channels, stem, rx_n, mode,
                                                      repetition, channel, phase_amp_thr, snr_min, gui_temporal,
                                                      sample_snr_min, compact=(gui_plots == 'compact'))
        output_dict.update(extra_dict)
        if extra_dict.get('temporal_mode') == '1:12':         # splitter 1:12: stability of every channel vs RX1
            try:
                extra_results = extra_results + _stability_all(np.asarray(data, dtype=complex), channels, lo, hi, stem,
                                                               extra_dict['valid_rf_channels'],
                                                               gui_temporal and gui_plots != 'compact')
            except Exception as e:
                print("WARNING: 1:12 stability failed (%s). The other results are not affected." % e)
    except Exception as e:
        print("WARNING: temporal analysis failed (%s). The stability results above are not affected." % e)

    if gui_temporal and gui_plots == 'compact' and extra_results:
        # Compact viewer: the temporal plots only. If the burst has no edges (burstTime = 0) there are no zooms:
        # the two original stability plots take the second row.
        widgets = list(extra_results)
        if len(widgets) < 3:
            widgets += [dict(result_1, row=1, col=0), dict(result_2, row=1, col=1)]
        return output_dict, widgets, dicom_meta_data
    return output_dict, [result_1, result_2] + extra_results, dicom_meta_data


def MimoCoherenceAll(folder, out_name='mimo_coherence_all'):
    """
    Combines the *_reps.csv files of a folder (one per RXn) into the summary across channels:
    <out_name>_summary.csv, <out_name>_ratio.png, <out_name>_phase.png.
    """
    files = sorted(glob.glob(os.path.join(folder, '*_reps.csv')))
    if not files:
        raise FileNotFoundError("No *_reps.csv files found in %s" % folder)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    rows, ratio_data, dphi_data, labels = [], [], [], []
    for ch, g in df.groupby('rx_channel'):
        m, s, cv = _lin_stats(g['amp_ratio'].values)
        cm, cs, r = _circ_stats(np.radians(g['dphi_deg'].values))
        rows.append({'rx_channel': ch, 'n_reps': len(g), 'ratio_mean': m, 'ratio_std': s, 'ratio_cv_percent': cv,
                     'dphi_circ_mean_deg': cm, 'dphi_circ_std_deg': cs, 'dphi_R': r})
        ratio_data.append(g['amp_ratio'].values)
        dphi_data.append(g['dphi_deg'].values)
        labels.append('RX%i' % ch)
    pd.DataFrame(rows).to_csv(os.path.join(folder, out_name + '_summary.csv'), index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(ratio_data, tick_labels=labels)
    ax.set(ylabel='|RXn| / |RX1|', title='Amplitude ratio vs RX1')
    fig.tight_layout()
    fig.savefig(os.path.join(folder, out_name + '_ratio.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(dphi_data, tick_labels=labels)
    ax.set(ylabel='Phase difference (deg)', title='Phase difference vs RX1')
    fig.tight_layout()
    fig.savefig(os.path.join(folder, out_name + '_phase.png'), dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)