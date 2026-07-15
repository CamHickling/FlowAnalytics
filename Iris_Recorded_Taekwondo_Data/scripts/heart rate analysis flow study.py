"""
Full-session heart-rate analysis for the poomsae flow study.

What changed:
----------------------------------------
1. The previous 'baseline' was the warm-up MEAN... an already-
   activated ENTRY state, not rest.  We compute TWO anchors and TWO deltas:
       * resting baseline (recovered 'finish' state)
             -> delta_fitness = peak - rest            [PRIMARY cardiac DV]  (HR reserve / fitness)
       * warm-up entry level (pre-performance mean)
             -> delta_reactivity = peak - warm-up entry (reactivity conditioned on the warm-up)

2. I've added vertical lines + labels demarcate SELF-NARRATION (review) and
   OBJECTIVE-SCORING (scoring), in addition to performance start/end.

3. The PERFORMANCE ROI now shows RAW and NORMALISED (as per Baruch).  Two panels side by side:
       (a) RAW BPM on a seconds axis (absolute values preserved), and
       (b) % CHANGE from the WARM-UP ENDPOINT (mean of the last ENTRY_ONSET_SEC of
           warm-up) so every athlete starts at ~0% and the SHAPES are comparable
           across participants with different baselines.
   A third panel shows the early-recovery ROI (HRR60 + optional tau).

4. The crawl also emits a single figure overlaying every
   participant's warm-up-endpoint-normalised performance curve on one axis... the
   direct cross-participant pattern comparison the normalisation unlocks.

5. The crawl writes an analysis-ready master CSV of per-session
   scalar features (both deltas, peak, time-to-peak, HRR60, tau, AUCg/AUCi,
   artifact/QC counts, resting/recovery RMSSD & SDNN).

Notes
-----
* HR envelope smoothed with a 5s CENTRED rolling mean for MACRO dynamics only.
  The RR series is NEVER smoothed before HRV.
* Peak / HRR / recovery use the smoothed series (robust to single-sample spikes);
  raw peak is also stored.
* scipy is optional (recovery-tau fit only).
"""

import os
import fnmatch
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# numpy>=2.0 renamed trapz -> trapezoid; support both (lab conda envs often ship numpy 1.x)
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

# ----------------------------------------------------------------------------- #
#  CONFIG
# ----------------------------------------------------------------------------- #
SMOOTHING_WINDOW   = '5s'      # centred rolling mean on the BPM envelope
REST_PHASE         = 'finish'  # recovered/rested state used for the resting baseline
REST_FALLBACK_MIN  = 5.0       # if no 'finish' phase, use the last N minutes as rest
ENTRY_ONSET_SEC    = 15        # warm-up ENDPOINT window (s): mean of last N s of warm-up
HRR_SECONDS        = 60        # HRR window (s) after peak
RECOVERY_FIT_MIN   = 5.0       # window (min) after peak for the mono-exponential fit
MALIK_PCT          = 0.20      # NN artifact rule: drop beats differing >20% from the prior
PERF_ZOOM_PRE_S    = 5         # pre-onset context (s) shown in the performance panels
PERF_ZOOM_POST_S   = 5         # post-end context (s) shown in the performance panels
REC_ZOOM_S         = 180       # length (s) of the recovery zoom panel

C_WARM, C_PERF, C_REVIEW, C_SCORE, C_FINISH = '#1f77b4', '#d62728', '#7d5ba6', '#dd8a1f', '#2ca02c'


# ----------------------------------------------------------------------------- #
#  SMALL HELPERS
# ----------------------------------------------------------------------------- #
def _phase(df, name):
    return df[df['phase'].astype(str).str.lower() == name]


def _phase_bounds_min(df, name):
    s = _phase(df, name)
    if s.empty:
        return None, None
    return s['rel_time_mins'].min(), s['rel_time_mins'].max()


def _resting_baseline(df):
    """Robust resting HR: lowest 30 s rolling mean of the recovered 'finish' phase."""
    rec = _phase(df, REST_PHASE)
    if rec.empty:
        tmax = df['rel_time_mins'].max()
        rec = df[df['rel_time_mins'] >= tmax - REST_FALLBACK_MIN]
    if rec.empty:
        return float(df['bpm'].min())
    roll = rec.set_index('datetime')['bpm'].rolling('30s', min_periods=10, center=True).mean()
    val = roll.min()
    return float(val) if np.isfinite(val) else float(rec['bpm'].quantile(0.05))


def _hrv_timedomain(df, phase_name):
    if 'rr_intervals_ms' not in df.columns:
        return {}
    sub = _phase(df, phase_name)
    rr = []
    for v in sub['rr_intervals_ms'].dropna():
        rr.extend(float(x) for x in str(v).split(';') if x not in ('', 'nan'))
    rr = np.asarray(rr, dtype=float)
    rr = rr[(rr >= 300) & (rr <= 2000)]
    if rr.size < 5:
        return {}
    keep = np.ones(rr.size, bool)
    keep[1:] = (np.abs(np.diff(rr)) / rr[:-1]) <= MALIK_PCT
    nn = rr[keep]
    d = np.diff(nn)
    return {'rmssd_ms': float(np.sqrt(np.mean(d ** 2))), 'sdnn_ms': float(np.std(nn, ddof=1)),
            'mean_hr': float(60000.0 / nn.mean()), 'n_beats': int(rr.size),
            'pct_flagged': float(100.0 * (~keep).mean())}


def _mono(t, hr_rest, amp, tau):
    return hr_rest + amp * np.exp(-t / tau)


def _perf_norm_curves(df, m, pre_pad_s=0.0, post_pad_s=0.0):
    """Performance-window HR relative to the warm-up endpoint (entry_onset), as BOTH
    % change and absolute Δ-bpm. Returns sec, pct, delta, ref."""
    ref = m['entry_onset_bpm']
    perf_dur_min = m['perf_duration_s'] / 60.0
    lo = -pre_pad_s / 60.0
    hi = perf_dur_min + post_pad_s / 60.0
    seg = df[(df['rel_time_mins'] >= lo) & (df['rel_time_mins'] <= hi)].copy()
    sec = seg['rel_time_mins'].values * 60.0
    delta = seg['bpm_smoothed'].values - ref
    pct = delta / ref * 100.0
    return sec, pct, delta, ref


# ----------------------------------------------------------------------------- #
#  METRIC EXTRACTION
# ----------------------------------------------------------------------------- #
def compute_session_metrics(df, perf_duration_min):
    perf_dur_s = perf_duration_min * 60.0

    baseline_rest = _resting_baseline(df)

    pre = df[df['rel_time_mins'] < 0]
    entry_warmup = float(pre['bpm'].mean()) if not pre.empty else np.nan
    onset_win = pre[pre['rel_time_mins'] >= -(ENTRY_ONSET_SEC / 60.0)]
    entry_onset = float(onset_win['bpm'].mean()) if not onset_win.empty else entry_warmup

    perf_win = df[(df['rel_time_mins'] >= 0) & (df['rel_time_mins'] <= perf_duration_min)]
    if perf_win.empty:
        perf_win = df
    peak_hr = float(perf_win['bpm_smoothed'].max())
    peak_raw = float(perf_win['bpm'].max())
    peak_time_min = float(perf_win.loc[perf_win['bpm_smoothed'].idxmax(), 'rel_time_mins'])
    time_to_peak_s = peak_time_min * 60.0

    delta_fitness = peak_hr - baseline_rest
    delta_reactivity = peak_hr - entry_warmup

    t60 = peak_time_min + HRR_SECONDS / 60.0
    hrr = np.nan
    if df['rel_time_mins'].max() >= t60:
        idx = (df['rel_time_mins'] - t60).abs().idxmin()
        hrr = peak_hr - float(df.loc[idx, 'bpm_smoothed'])

    tau_s, tau_r2, tau_fit = np.nan, np.nan, None
    if _HAVE_SCIPY:
        rec = df[(df['rel_time_mins'] >= peak_time_min) &
                 (df['rel_time_mins'] <= peak_time_min + RECOVERY_FIT_MIN)].copy()
        if len(rec) >= 10:
            tr = (rec['rel_time_mins'].values - rec['rel_time_mins'].values[0]) * 60.0
            yr = rec['bpm_smoothed'].values
            try:
                popt, _ = curve_fit(_mono, tr, yr, p0=[baseline_rest, max(peak_hr - baseline_rest, 1), 60],
                                    maxfev=30000, bounds=([40, 0, 3], [120, 200, 900]))
                tau_s = float(popt[2])
                ss_res = np.sum((yr - _mono(tr, *popt)) ** 2)
                ss_tot = np.sum((yr - yr.mean()) ** 2)
                tau_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
                tau_fit = (float(popt[0]), float(popt[1]), float(popt[2]))
            except Exception:
                pass

    aucg = auci = np.nan
    if len(perf_win) >= 2:
        tt = (perf_win['datetime'] - perf_win['datetime'].iloc[0]).dt.total_seconds().values
        yy = perf_win['bpm'].values
        aucg = float(_trapz(yy, tt))
        auci = float(_trapz(yy - baseline_rest, tt))

    metrics = {
        'baseline_rest_bpm': round(baseline_rest, 1),
        'entry_warmup_bpm': round(entry_warmup, 1) if np.isfinite(entry_warmup) else np.nan,
        'entry_onset_bpm': round(entry_onset, 1) if np.isfinite(entry_onset) else np.nan,
        'peak_smoothed_bpm': round(peak_hr, 1),
        'peak_raw_bpm': round(peak_raw, 1),
        'delta_fitness_bpm': round(delta_fitness, 1),
        'delta_reactivity_bpm': round(delta_reactivity, 1) if np.isfinite(delta_reactivity) else np.nan,
        'peak_pct_change_from_entry': round((peak_hr - entry_onset) / entry_onset * 100, 1) if np.isfinite(entry_onset) else np.nan,
        'time_to_peak_s': round(time_to_peak_s, 1),
        'hrr60_bpm': round(hrr, 1) if np.isfinite(hrr) else np.nan,
        'recovery_tau_s': round(tau_s, 1) if np.isfinite(tau_s) else np.nan,
        'recovery_tau_r2': round(tau_r2, 3) if np.isfinite(tau_r2) else np.nan,
        'perf_aucg_bpm_s': round(aucg, 0) if np.isfinite(aucg) else np.nan,
        'perf_auci_bpm_s': round(auci, 0) if np.isfinite(auci) else np.nan,
        'perf_duration_s': round(perf_dur_s, 1),
    }
    for tag, ph in [('warmup', 'warmup_calibration'), ('finish', REST_PHASE)]:
        h = _hrv_timedomain(df, ph)
        if h:
            metrics[f'rmssd_{tag}_ms'] = round(h['rmssd_ms'], 1)
            metrics[f'sdnn_{tag}_ms'] = round(h['sdnn_ms'], 1)
            metrics[f'pct_rr_flagged_{tag}'] = round(h['pct_flagged'], 2)

    metrics['_peak_time_min'] = peak_time_min
    if tau_fit is not None:
        metrics['_tau_fit'] = tau_fit
    return metrics


# ----------------------------------------------------------------------------- #
#  PLOTTING
# ----------------------------------------------------------------------------- #
def _shade_regions(ax, df, xcol='rel_time_mins', xscale=1.0):
    regions = [('warmup_calibration', C_WARM, 'Warm-up'),
               ('performance', C_PERF, 'Performance'),
               ('review', C_REVIEW, 'Self-narration\n(review)'),
               ('scoring', C_SCORE, 'Objective scoring'),
               ('finish', C_FINISH, 'Recovery\n(finish)')]
    labels = []
    for ph, col, lab in regions:
        s = _phase(df, ph)
        if s.empty:
            continue
        x0, x1 = s[xcol].min() * xscale, s[xcol].max() * xscale
        ax.axvspan(x0, x1, color=col, alpha=0.06, zorder=0)
        labels.append(((x0 + x1) / 2.0, lab, col))
    return labels


def _plot_full_session(ax, df, m):
    baseline_rest, entry_warmup, peak_hr = m['baseline_rest_bpm'], m['entry_warmup_bpm'], m['peak_smoothed_bpm']
    perf_dur_min = m['perf_duration_s'] / 60.0

    ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.6)
    region_labels = _shade_regions(ax, df)
    ax.scatter(df['rel_time_mins'], df['bpm'], color='gray', s=7, alpha=0.22, label='Raw telemetry', zorder=1)
    for lo, hi, col, lab in [(-np.inf, 0, C_WARM, 'Pre-performance'),
                             (0, perf_dur_min, C_PERF, 'Performance'),
                             (perf_dur_min, np.inf, C_FINISH, 'Post-performance')]:
        seg = df[(df['rel_time_mins'] >= lo) & (df['rel_time_mins'] <= hi)]
        ax.plot(seg['rel_time_mins'], seg['bpm_smoothed'], color=col, linewidth=3.0, label=lab, zorder=2)

    ax.axhline(baseline_rest, color=C_FINISH, ls='--', lw=1.6, zorder=3, label=f'Resting baseline {baseline_rest:.0f}  (recovered)')
    ax.axhline(entry_warmup, color=C_SCORE, ls='--', lw=1.6, zorder=3, label=f'Warm-up entry {entry_warmup:.0f}  (pre-perf mean)')
    ax.axhline(peak_hr, color='darkred', ls='--', lw=1.4, zorder=3, label=f'Peak {peak_hr:.0f}')

    xm = perf_dur_min / 2.0
    ax.annotate('', xy=(xm + 0.06, peak_hr), xytext=(xm + 0.06, baseline_rest),
                arrowprops=dict(arrowstyle='<->', color=C_FINISH, lw=2), zorder=4)
    ax.text(xm + 0.14, (peak_hr + baseline_rest) / 2, f'Δ fitness\n(peak−rest)\n+{m["delta_fitness_bpm"]:.0f}',
            color=C_FINISH, fontsize=9, fontweight='bold', va='center', zorder=4,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor=C_FINISH, boxstyle='round,pad=0.2'))
    ax.annotate('', xy=(xm - 0.06, peak_hr), xytext=(xm - 0.06, entry_warmup),
                arrowprops=dict(arrowstyle='<->', color=C_SCORE, lw=2), zorder=4)
    ax.text(xm - 0.14, (peak_hr + entry_warmup) / 2, f'Δ reactivity\n(peak−warm-up)\n+{m["delta_reactivity_bpm"]:.0f}',
            color=C_SCORE, fontsize=9, fontweight='bold', va='center', ha='right', zorder=4,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor=C_SCORE, boxstyle='round,pad=0.2'))

    ymin, ymax = ax.get_ylim()
    markers = [(0.0, 'PERFORMANCE START'), (perf_dur_min, 'PERF END / SELF-NARRATION'),
               (_phase_bounds_min(df, 'scoring')[0], 'OBJECTIVE SCORING'),
               (_phase_bounds_min(df, 'finish')[0], 'RECOVERY (FINISH)')]
    for x, lab in markers:
        if x is None:
            continue
        ax.axvline(x=x, color='black', ls='-', lw=1.6, zorder=3)
        ax.text(x + 0.06, ymin + 6, lab, fontsize=8.5, fontweight='bold', rotation=90, va='bottom')
    for xc, lab, col in region_labels:
        ax.text(xc, ymax - 4, lab, ha='center', va='top', fontsize=8.5, color=col, style='italic', fontweight='bold')

    ax.set_title('Full-session heart-rate architecture — dual baseline (fitness vs. reactivity)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Time from performance onset (minutes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Heart rate (BPM)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8.5, ncol=2)


def _perf_xrange_s(m):
    return -PERF_ZOOM_PRE_S, m['perf_duration_s'] + PERF_ZOOM_POST_S


def _plot_performance_raw(ax, df, m):
    """ROI (a): performance on a SECONDS axis, RAW absolute BPM."""
    perf_dur_s = m['perf_duration_s']
    x0, x1 = _perf_xrange_s(m)
    seg = df[(df['rel_time_mins'] * 60 >= x0) & (df['rel_time_mins'] * 60 <= x1)].copy()
    seg['sec'] = seg['rel_time_mins'] * 60.0

    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.axvspan(0, perf_dur_s, color=C_PERF, alpha=0.06)
    ax.scatter(seg['sec'], seg['bpm'], color='gray', s=12, alpha=0.35, zorder=1, label='raw')
    ax.plot(seg['sec'], seg['bpm_smoothed'], color=C_PERF, lw=2.6, zorder=2, label='5 s smoothed')
    ax.axhline(m['baseline_rest_bpm'], color=C_FINISH, ls='--', lw=1.2)
    ax.axhline(m['entry_onset_bpm'], color=C_SCORE, ls='--', lw=1.2)
    ttp = m['time_to_peak_s']
    ax.scatter([ttp], [m['peak_smoothed_bpm']], s=70, color='darkred', zorder=5, edgecolor='white', lw=1.1)
    ax.annotate(f'peak {m["peak_smoothed_bpm"]:.0f} @ {ttp:.0f}s', xy=(ttp, m['peak_smoothed_bpm']),
                xytext=(ttp + 4, m['peak_smoothed_bpm'] + 1.5), fontsize=9, color='darkred', fontweight='bold')
    ax.axvline(0, color='black', lw=1.2)
    ax.axvline(perf_dur_s, color='black', lw=1.2)
    ax.set_xlim(x0, x1)
    ax.set_title(f'ROI ①a  Performance — RAW ({perf_dur_s:.0f} s)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Seconds from performance onset', fontsize=10, fontweight='bold')
    ax.set_ylabel('Heart rate (BPM)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')


def _plot_performance_pctchange(ax, df, m):
    """ROI (b): performance normalised to the warm-up endpoint (starts at ~0%).
    Left axis = % change; secondary right axis reads the same curve as absolute Δ-bpm."""
    perf_dur_s = m['perf_duration_s']
    ref = m['entry_onset_bpm']
    sec, pct, delta, _ = _perf_norm_curves(df, m, pre_pad_s=PERF_ZOOM_PRE_S, post_pad_s=PERF_ZOOM_POST_S)

    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.axvspan(0, perf_dur_s, color=C_PERF, alpha=0.06)
    ax.axhline(0, color='black', lw=1.3, zorder=3)  # warm-up-endpoint reference (0%)
    ax.plot(sec, pct, color=C_PERF, lw=2.6, zorder=2)
    if pct.size:
        ipk = int(np.nanargmax(pct))
        ax.scatter([sec[ipk]], [pct[ipk]], s=70, color='darkred', zorder=5, edgecolor='white', lw=1.1)
        ax.annotate(f'+{pct[ipk]:.0f}%  (+{delta[ipk]:.0f} bpm)', xy=(sec[ipk], pct[ipk]),
                    xytext=(sec[ipk] + 4, pct[ipk] + 1), fontsize=9, color='darkred', fontweight='bold')
    ax.axvline(0, color='black', lw=1.2)
    ax.axvline(perf_dur_s, color='black', lw=1.2)
    ax.set_xlim(*_perf_xrange_s(m))
    ax.set_title(f'ROI ①b  Performance — normalised to warm-up endpoint ({ref:.0f} bpm)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Seconds from performance onset', fontsize=10, fontweight='bold')
    ax.set_ylabel('% change from warm-up endpoint', fontsize=10, fontweight='bold')
    secax = ax.secondary_yaxis('right', functions=(lambda p: p * ref / 100.0, lambda d: d / ref * 100.0))
    secax.set_ylabel('Δ HR from endpoint (bpm)', fontsize=9)


def _plot_recovery_zoom(ax, df, m):
    peak_t = m['_peak_time_min']
    seg = df[(df['rel_time_mins'] >= peak_t) & (df['rel_time_mins'] <= peak_t + REC_ZOOM_S / 60.0)].copy()
    seg['sec'] = (seg['rel_time_mins'] - peak_t) * 60.0
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.scatter(seg['sec'], seg['bpm'], color='gray', s=10, alpha=0.3, zorder=1)
    ax.plot(seg['sec'], seg['bpm_smoothed'], color=C_FINISH, lw=2.6, zorder=2, label='HR (smoothed)')
    if '_tau_fit' in m:
        tt = np.linspace(0, REC_ZOOM_S, 200)
        ax.plot(tt, _mono(tt, *m['_tau_fit']), color='#d69e2e', lw=2.4, zorder=3,
                label=f'mono-exp fit (τ={m["recovery_tau_s"]:.0f}s, R²={m["recovery_tau_r2"]:.2f})')
    if np.isfinite(m['hrr60_bpm']):
        hr60 = m['peak_smoothed_bpm'] - m['hrr60_bpm']
        ax.annotate('', xy=(HRR_SECONDS, hr60), xytext=(HRR_SECONDS, m['peak_smoothed_bpm']),
                    arrowprops=dict(arrowstyle='<->', color='purple', lw=2), zorder=4)
        ax.text(HRR_SECONDS + 4, (hr60 + m['peak_smoothed_bpm']) / 2, f'HRR₆₀\n−{m["hrr60_bpm"]:.0f}',
                color='purple', fontsize=9, fontweight='bold', va='center')
        ax.axvline(HRR_SECONDS, color='purple', ls=':', lw=1)
    ax.axhline(m['baseline_rest_bpm'], color=C_FINISH, ls='--', lw=1.3)
    ax.set_title(f'ROI ②  Early recovery (0–{REC_ZOOM_S} s from peak)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Seconds from peak', fontsize=10, fontweight='bold')
    ax.set_ylabel('Heart rate (BPM)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)


def render_group_overlay(curves, desktop_folder, root_directory, fname='group_performance_normalised.png',
                         grid_points=201):
    """Overlay every participant's performance curve, TIME-NORMALISED to % of poomsae
    time (so different-length forms align), in two flavours: % change from the warm-up
    endpoint, and absolute Δ-bpm from the same endpoint. Adds the group-mean trajectory."""
    curves = [c for c in curves if c['sec'].size > 1]
    if not curves:
        return
    grid = np.linspace(0, 100, grid_points)  # % of performance time

    fig, (ax_pct, ax_delta) = plt.subplots(1, 2, figsize=(18, 7.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(curves)))
    # thin/fade the individual traces as N grows so the group mean stays readable
    n = len(curves)
    _alpha = 0.65 if n <= 8 else (0.45 if n <= 20 else 0.28)
    _lw = 1.4 if n <= 8 else (1.1 if n <= 20 else 0.8)
    pct_stack, delta_stack = [], []
    for c, col in zip(curves, colors):
        nt = c['sec'] / c['dur_s'] * 100.0            # actual time -> % of performance time
        order = np.argsort(nt)
        nt, pct_o, delta_o = nt[order], c['pct'][order], c['delta'][order]
        pct_g = np.interp(grid, nt, pct_o)             # resample onto the common 0-100% grid
        delta_g = np.interp(grid, nt, delta_o)
        pct_stack.append(pct_g)
        delta_stack.append(delta_g)
        ax_pct.plot(grid, pct_g, lw=_lw, alpha=_alpha, color=col, label=str(c['id'])[:24], zorder=3)
        ax_delta.plot(grid, delta_g, lw=_lw, alpha=_alpha, color=col, zorder=3)

    for ax, stack, ylab, sub in [
        (ax_pct, np.array(pct_stack), '% change from warm-up endpoint', '% change'),
        (ax_delta, np.array(delta_stack), 'Δ HR from warm-up endpoint (bpm)', 'absolute Δ-bpm')]:
        mu = stack.mean(axis=0)
        if len(curves) >= 3:
            sd = stack.std(axis=0, ddof=1)
            ax.fill_between(grid, mu - sd, mu + sd, color='gray', alpha=0.18, zorder=1, label='± 1 SD')
        ax.plot(grid, mu, color='black', lw=2.8, zorder=5, label='group mean')
        ax.axhline(0, color='black', lw=1.0, ls=':', zorder=2)
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel('% of performance (poomsae) time', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylab, fontsize=12, fontweight='bold')
        ax.set_title(f'Normalised: {sub}', fontsize=12, fontweight='bold')
    if len(curves) <= 18:
        ax_pct.legend(fontsize=7.5, ncol=2, loc='upper left', framealpha=0.9)

    fig.suptitle('All participants — performance HR, time-normalised to % of poomsae time',
                 fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.06, right=0.985, wspace=0.20)
    for dest in (os.path.join(desktop_folder, fname), os.path.join(root_directory, fname)):
        try:
            fig.savefig(dest, dpi=200)
            print(f"  Group overlay -> {dest}")
        except Exception as e:
            print(f"  [WARN] could not write group overlay to {dest}: {e}")
    plt.close(fig)


# ----------------------------------------------------------------------------- #
#  SESSION PROCESSING
# ----------------------------------------------------------------------------- #
def _parse_ids(file_path, root_directory):
    """Session folder = first path component under the data root, e.g.
    <root>/P05_20260225_121858/heart_rate/P05_hr_full_session.csv -> 'P05_20260225_121858'.
    Participant id = leading token, e.g. 'P05' (also handles 'P02B', 'P04__...')."""
    try:
        rel = os.path.relpath(file_path, root_directory)
        experiment_id = rel.split(os.sep)[0]
    except Exception:
        parts = file_path.split(os.sep)
        experiment_id = parts[-3] if len(parts) >= 3 else "Unknown_Session"
    participant_id = experiment_id.split('_')[0] if experiment_id else "Unknown"
    return experiment_id, participant_id


def process_single_session(file_path, desktop_folder, root_directory=None, curve_accumulator=None,
                           output_image_name='session_hr_analysis.png'):
    if root_directory is None:
        root_directory = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    experiment_id, participant_id = _parse_ids(file_path, root_directory)
    print(f"  Processing Session: {experiment_id}  (participant {participant_id})")

    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('datetime').reset_index(drop=True)

    perf_data = _phase(df, 'performance')
    if perf_data.empty:
        print(f"    [Skipped] No 'performance' phase found for {experiment_id}")
        return None
    perf_start, perf_end = perf_data['datetime'].min(), perf_data['datetime'].max()
    df['rel_time_mins'] = (df['datetime'] - perf_start).dt.total_seconds() / 60.0
    perf_duration_min = (perf_end - perf_start).total_seconds() / 60.0

    df = df.set_index('datetime')
    df['bpm_smoothed'] = df['bpm'].rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean()
    df = df.reset_index().dropna(subset=['bpm_smoothed'])
    if df.empty:
        return None

    m = compute_session_metrics(df, perf_duration_min)

    # figure: full session on top; performance RAW | performance %CHANGE | recovery below
    fig = plt.figure(figsize=(18, 10.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.3, 1.0])
    ax_full = fig.add_subplot(gs[0, :])
    ax_raw = fig.add_subplot(gs[1, 0])
    ax_pct = fig.add_subplot(gs[1, 1])
    ax_rec = fig.add_subplot(gs[1, 2])

    _plot_full_session(ax_full, df, m)
    _plot_performance_raw(ax_raw, df, m)
    _plot_performance_pctchange(ax_pct, df, m)
    _plot_recovery_zoom(ax_rec, df, m)

    fig.suptitle(f'Session ID: {experiment_id}', fontsize=13, fontweight='bold', y=0.995)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.05, right=0.99, hspace=0.34, wspace=0.22)

    local_output_path = os.path.join(os.path.dirname(file_path), output_image_name)
    plt.savefig(local_output_path, dpi=300)
    plt.close(fig)
    shutil.copy2(local_output_path, os.path.join(desktop_folder, f"{experiment_id}_responsive_analysis.png"))
    # NOTE: console output is deliberately plain ASCII (Windows consoles are often cp1252
    # and raise UnicodeEncodeError on characters like the delta sign when redirected).
    print(f"    -> Exported figure. d_fit=+{m['delta_fitness_bpm']:.0f} | d_react=+{m['delta_reactivity_bpm']:.0f} | "
          f"peak%={m['peak_pct_change_from_entry']:.0f}% | t-to-peak={m['time_to_peak_s']:.0f}s | HRR60={m['hrr60_bpm']}")

    # accumulate the normalised performance curve (0..perf_dur_s) for the group overlay
    if curve_accumulator is not None:
        sec, pct, delta, _ = _perf_norm_curves(df, m, pre_pad_s=0.0, post_pad_s=0.0)
        curve_accumulator.append({'id': participant_id, 'sec': sec, 'pct': pct,
                                  'delta': delta, 'dur_s': m['perf_duration_s']})

    m_public = {k: v for k, v in m.items() if not k.startswith('_')}
    return {'participant_id': participant_id,
            'is_repeat_session': participant_id.upper().endswith('B'),
            'experiment_id': experiment_id,
            'file_path': file_path, **m_public}


def crawl_and_render_all(root_directory, target_pattern,
                         desktop_folder_name='Taekwondo_HR_Plots_Full',
                         metrics_csv_name='taekwondo_hr_metrics_master.csv'):
    if not os.path.exists(root_directory):
        print(f"Error: Target directory '{root_directory}' does not exist.")
        return
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', desktop_folder_name)
    os.makedirs(desktop_path, exist_ok=True)
    print(f"Output folder: {desktop_path}")

    print(f"\nSweeping: {root_directory}\n" + "=" * 60)
    processed, rows, curves = 0, [], []
    for dirpath, _, filenames in os.walk(root_directory):
        # pattern match so renamed files (e.g. 'P05_hr_full_session.csv') are still found
        for fname in sorted(fnmatch.filter(filenames, target_pattern)):
            full_csv_path = os.path.join(dirpath, fname)
            try:
                m = process_single_session(full_csv_path, desktop_path,
                                           root_directory=root_directory, curve_accumulator=curves)
                if m is not None:
                    rows.append(m)
                    processed += 1
            except Exception as e:
                print(f"    [ERROR] {full_csv_path}: {e}")

    if rows:
        metrics_df = pd.DataFrame(rows).sort_values('participant_id').reset_index(drop=True)
        for dest in (os.path.join(desktop_path, metrics_csv_name), os.path.join(root_directory, metrics_csv_name)):
            try:
                metrics_df.to_csv(dest, index=False)
                print(f"  Master metrics -> {dest}")
            except Exception as e:
                print(f"  [WARN] could not write metrics to {dest}: {e}")

    render_group_overlay(curves, desktop_path, root_directory)
    print("=" * 60 + f"\nSweep complete! {processed} sessions rendered.")

    if processed == 0:
        _diagnose_empty_sweep(root_directory, target_pattern)


def _diagnose_empty_sweep(root_directory, target_pattern, max_show=15):
    """Explain WHY nothing was found: wrong folder, or a filename mismatch?"""
    print("\n" + "-" * 60)
    print(f"DIAGNOSTIC: nothing matching '{target_pattern}' found under:\n  {root_directory}")

    subdirs, csvs = [], []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        if dirpath == root_directory:
            subdirs = sorted(dirnames)
        for f in filenames:
            if f.lower().endswith('.csv'):
                csvs.append(os.path.join(dirpath, f))

    print(f"\nTop-level folders here ({len(subdirs)}): "
          f"{', '.join(subdirs[:max_show]) if subdirs else '(none)'}")

    if not csvs:
        print("\nNo .csv files found anywhere under this root.")
        print("=> The session DATA is probably not in this folder tree (only the scripts?).")
        print("   The original script pointed at F:\\Iris_Recorded_Taekwondo_Data - check that drive,")
        print("   then set TARGET_ROOT_DIR at the bottom of this script to the real data folder.")
    else:
        print(f"\nFound {len(csvs)} .csv file(s). First few:")
        for p in csvs[:max_show]:
            print(f"   {p}")
        names = sorted({os.path.basename(p) for p in csvs})
        print(f"\nDistinct CSV filenames: {', '.join(names[:max_show])}")
        near = [n for n in names if 'hr' in n.lower() or 'heart' in n.lower()]
        if near:
            print(f"\n=> Possible FILENAME MISMATCH. Candidates: {', '.join(near)}")
            print(f"   Set TARGET_FILE_NAME at the bottom of this script to the correct name.")
        else:
            print("\n=> CSVs exist but none look like heart-rate session files.")
            print("   Check that this is the right data root.")
    print("-" * 60)


def _default_root():
    """Data root = the folder ABOVE this script (works when the script sits in
    <data root>/scripts/). Falls back to the current working directory."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(here) if os.path.basename(here).lower() == 'scripts' else here
    except NameError:
        return os.getcwd()


if __name__ == "__main__":
    # Leave TARGET_ROOT_DIR = None to auto-detect (script lives in <data root>/scripts/),
    # or hard-code a path, e.g. r"C:\Users\BarlabPRIME\Desktop\FlowAnalytics\Iris_Recorded_Taekwondo_Data"
    TARGET_ROOT_DIR = None
    # glob pattern, so both 'hr_full_session.csv' and 'P05_hr_full_session.csv' match
    TARGET_FILE_PATTERN = "*hr_full_session.csv"

    root = TARGET_ROOT_DIR or _default_root()
    print(f"Data root: {root}")
    print(f"Matching:  {TARGET_FILE_PATTERN}")
    crawl_and_render_all(root, TARGET_FILE_PATTERN)