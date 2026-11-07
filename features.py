import numpy as np, pandas as pd, sys, time
import scipy.ndimage as nd
from scipy import stats
from astropy.timeseries import BoxLeastSquares

import os
DATA = os.environ.get('DATA', '.')
DT = 0.0208333
N = 1315
PERIODS = np.exp(np.linspace(np.log(0.45), np.log(13.0), 900))
DURS = np.array([0.05, 0.1, 0.2])


def ccd_trend(f, stars):
    """per-CCD median curve (computed within the dataset itself)."""
    tr = np.zeros_like(f)
    key = (stars.camera.values * 10 + stars.ccd.values)
    for k in np.unique(key):
        m = key == k
        tr[m] = np.nanmedian(f[m], axis=0)
    return tr


def prep(flux, cen, stars):
    f = np.asarray(flux, np.float64)
    f = f / np.nanmedian(f, 1, keepdims=True) - 1
    f_c = f - ccd_trend(f, stars)
    c = np.asarray(cen, np.float64)  # (n,2,N)
    cc = np.empty_like(c)
    key = (stars.camera.values * 10 + stars.ccd.values)
    for ax in range(2):
        x = c[:, ax] - np.nanmedian(c[:, ax], 1, keepdims=True)
        tr = np.zeros_like(x)
        for k in np.unique(key):
            m = key == k
            tr[m] = np.nanmedian(x[m], axis=0)
        cc[:, ax] = x - tr
    return f_c, cc


def fill(a):
    """NaN->0 after detrend, return mask."""
    m = np.isfinite(a)
    return np.where(m, a, 0.0), m


def robust_sigma(x, axis=1):
    med = np.nanmedian(x, axis, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(x - med), axis) + 1e-12


def spectral(f0, mask):
    """FFT power on uniform grid (gaps zero-filled)."""
    n = f0.shape[0]
    nf = 4096
    w = np.hanning(N)
    F = np.fft.rfft((f0 - (f0.sum(1, keepdims=True) / mask.sum(1, keepdims=True)) * mask) * w, nf, axis=1)
    P = (np.abs(F) ** 2)[:, 1:]
    freq = np.fft.rfftfreq(nf, DT)[1:]  # cycles/day
    P = P / (P.sum(1, keepdims=True) + 1e-30)
    out = {}
    edges = [0.04, 0.15, 0.3, 0.6, 1, 2, 4, 8, 14, 24.1]
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (freq >= a) & (freq < b)
        out[f'sp_{a}_{b}'] = np.log10(P[:, sel].sum(1) + 1e-9)
    # top peaks
    Pm = P.copy()
    for k in range(3):
        i = Pm.argmax(1)
        out[f'pk{k}_f'] = np.log10(freq[i])
        out[f'pk{k}_p'] = np.log10(Pm[np.arange(n), i] + 1e-12)
        for d in range(-25, 26):
            j = np.clip(i + d, 0, Pm.shape[1] - 1)
            Pm[np.arange(n), j] = 0
    # peak sharpness vs median
    out['pk_over_med'] = out['pk0_p'] - np.log10(np.median(P, 1) + 1e-12)
    # harmonic ratio
    i0 = P.argmax(1)
    j2 = np.clip(2 * i0 + 1, 0, P.shape[1] - 1)
    out['harm2'] = np.log10(P[np.arange(n), j2] + 1e-12) - out['pk0_p']
    # spectral slope for noise-like
    out['sp_hi_lo'] = out['sp_8_14'] - out['sp_0.04_0.15']
    return out


def acf_feats(f0, mask):
    n = f0.shape[0]
    x = f0 - f0.sum(1, keepdims=True) / mask.sum(1, keepdims=True) * mask
    F = np.fft.rfft(x, 4096, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :700]
    ac = ac / (ac[:, :1] + 1e-30)
    out = {}
    out['ac1'] = ac[:, 1]
    out['ac10'] = ac[:, 10]
    out['ac48'] = ac[:, 48]
    # first local max after first zero crossing / dip below 0.2
    pk = np.zeros(n); ph = np.zeros(n)
    for i in range(n):
        a = ac[i]
        below = np.where(a < 0.1)[0]
        if len(below) == 0:
            pk[i] = 700; ph[i] = a[-1]; continue
        s = below[0]
        seg = a[s:]
        j = np.argmax(seg)
        pk[i] = (s + j) * DT; ph[i] = seg[j]
    out['acf_lag'] = pk; out['acf_h'] = ph
    return out


def orbit_split():
    return None


def bls_block(t, r, rmask, cx, cy, dump_idx, orbit_edge):
    """r: (N,) detrended relative flux with NaN. returns dict."""
    out = {}
    m = np.isfinite(r)
    tt = t[m]; y = r[m]
    sig = 1.4826 * np.median(np.abs(y - np.median(y))) + 1e-9
    bls = BoxLeastSquares(tt, y, dy=np.full(len(y), sig))
    res = bls.power(PERIODS, DURS, objective='snr')
    k = int(np.argmax(res.power))
    P = res.period[k]; t0 = res.transit_time[k]; dur = res.duration[k]
    depth = res.depth[k]
    out['bls_snr'] = res.depth_snr[k]
    out['bls_pow'] = res.power[k]
    out['bls_logP'] = np.log10(P)
    out['bls_dur'] = dur
    out['bls_dur_over_P'] = dur / P
    out['bls_depth'] = depth
    out['bls_depth_sigma'] = depth / sig
    # power distribution: how peaked
    pw = res.power
    out['bls_pow_med'] = np.median(pw)
    out['bls_pow_ratio'] = pw[k] / (np.percentile(pw, 90) + 1e-9)
    # second best period away from harmonics of best
    ratio = PERIODS / P
    far = ~((np.abs(ratio - np.round(ratio)) < 0.03) | (np.abs(1 / ratio - np.round(1 / ratio)) < 0.03) | (np.abs(ratio - 0.5) < 0.03) | (np.abs(ratio - 1.5) < 0.03))
    out['bls_pow2'] = pw[far].max() if far.any() else 0.0

    ph = ((tt - t0 + 0.5 * P) % P) - 0.5 * P   # centered on transit
    ep = np.round((tt - t0) / P).astype(int)
    intr = np.abs(ph) < dur / 2
    inner = np.abs(ph) < dur / 6
    shoulder = (np.abs(ph) >= dur / 2 * 0.6) & (np.abs(ph) < dur / 2)
    oot = np.abs(ph) > dur
    base = np.median(y[oot]) if oot.sum() > 5 else 0.0
    d_in = base - (np.mean(y[intr]) if intr.any() else base)
    out['n_in'] = intr.sum()
    out['n_epochs_in'] = len(np.unique(ep[intr])) if intr.any() else 0
    # shape U vs V: ratio of central depth to shoulder depth
    if inner.sum() > 1 and shoulder.sum() > 1:
        dc = base - y[inner].mean(); ds = base - y[shoulder].mean()
        out['shape_c_over_s'] = dc / (ds + 1e-9 * np.sign(ds + 1e-12))
        out['shape_c_s_diff'] = (dc - ds) / sig
    else:
        out['shape_c_over_s'] = np.nan; out['shape_c_s_diff'] = np.nan
    # odd / even
    od = intr & (ep % 2 == 1); ev = intr & (ep % 2 == 0)
    if od.sum() > 2 and ev.sum() > 2:
        a, b = base - y[od].mean(), base - y[ev].mean()
        se = sig * np.sqrt(1 / od.sum() + 1 / ev.sum())
        out['oe_diff_sig'] = abs(a - b) / se
        out['oe_ratio'] = min(a, b) / (max(a, b) + 1e-9) if max(a, b) > 0 else np.nan
    else:
        out['oe_diff_sig'] = np.nan; out['oe_ratio'] = np.nan
    # secondary at phase 0.5
    ph2 = ((tt - t0) % P) / P
    sec = (np.abs(ph2 - 0.5) * P < dur / 2)
    if sec.sum() > 2:
        ds = base - y[sec].mean()
        out['sec_depth_sig'] = ds / (sig / np.sqrt(sec.sum()))
        out['sec_over_pri'] = ds / (d_in + 1e-9 * np.sign(d_in + 1e-12))
    else:
        out['sec_depth_sig'] = np.nan; out['sec_over_pri'] = np.nan
    # per-epoch depth consistency
    deps = []
    for e in np.unique(ep[intr]):
        mm = intr & (ep == e)
        if mm.sum() >= 2:
            deps.append(base - y[mm].mean())
    deps = np.array(deps)
    out['ep_depth_cv'] = deps.std() / (abs(deps.mean()) + 1e-9) if len(deps) >= 2 else np.nan
    out['ep_depth_min_sig'] = (deps.min() / sig) if len(deps) else np.nan
    # fold scatter: scatter of in-transit points relative to noise
    out['in_scatter_ratio'] = (y[intr].std() / sig) if intr.sum() > 3 else np.nan
    # alignment with dumps: fraction of in-transit epochs near a dump
    if len(dump_idx):
        dt_ = t[dump_idx]
        tcs = t0 + P * np.arange(-5, int(30 / P) + 6)
        tcs = tcs[(tcs >= t.min()) & (tcs <= t.max())]
        if len(tcs):
            dmin = np.abs(tcs[:, None] - dt_[None]).min(1)
            out['frac_tr_near_dump'] = np.mean(dmin < 0.15)
        else:
            out['frac_tr_near_dump'] = np.nan
    else:
        out['frac_tr_near_dump'] = np.nan
    # transit-time spacing regularity relative to orbit edges: fraction of in-transit pts near orbit edges
    out['frac_in_near_edge'] = np.mean(np.min(np.abs(tt[intr, None] - orbit_edge[None]), 1) < 0.5) if intr.any() else np.nan
    # centroid shift in transit
    for name, c in (('cx', cx), ('cy', cy)):
        cm = np.isfinite(c)
        cc = c[cm]; tc_ = t[cm]
        sg = 1.4826 * np.median(np.abs(cc - np.median(cc))) + 1e-9
        phc = ((tc_ - t0 + 0.5 * P) % P) - 0.5 * P
        i_in = np.abs(phc) < dur / 2
        i_out = np.abs(phc) > dur
        if i_in.sum() > 2 and i_out.sum() > 5:
            out[f'{name}_shift_sig'] = (cc[i_in].mean() - cc[i_out].mean()) / (sg / np.sqrt(i_in.sum()))
            out[f'{name}_shift_over_noise'] = (cc[i_in].mean() - cc[i_out].mean()) / sg
        else:
            out[f'{name}_shift_sig'] = np.nan; out[f'{name}_shift_over_noise'] = np.nan
    out['cen_shift_abs'] = np.hypot(out['cx_shift_over_noise'], out['cy_shift_over_noise']) if np.isfinite(out['cx_shift_over_noise']) else np.nan
    out['cen_shift_sig_abs'] = np.hypot(out['cx_shift_sig'], out['cy_shift_sig']) if np.isfinite(out['cx_shift_sig']) else np.nan
    # centroid shift per unit flux depth (background-source signature)
    out['cen_per_depth'] = out['cen_shift_abs'] / (abs(depth) / sig + 1e-3) if np.isfinite(out['cen_shift_abs']) else np.nan
    return out


def extract(flux, cen, stars, cad, batch=500, verbose=True):
    t = cad.time_btjd.values
    q = cad.quality.values
    dump_idx = np.where((q & 1) > 0)[0]
    gaps = np.where(np.diff(t) > 0.5)[0]
    orbit_edge = np.concatenate([[t[0]], t[gaps], t[gaps + 1], [t[-1]]])
    # dump-window mask
    dmask = np.zeros(N, bool)
    for d in dump_idx:
        dmask[d:d + 4] = True
    n = len(stars)
    # CCD trends need the full set; compute prep for whole dataset once in chunks per column group
    f_all = np.asarray(flux, np.float64)
    f_all = f_all / np.nanmedian(f_all, 1, keepdims=True) - 1
    f_all = f_all - ccd_trend(f_all, stars)
    c_all = np.asarray(cen, np.float64)
    key = (stars.camera.values * 10 + stars.ccd.values)
    for ax in range(2):
        x = c_all[:, ax] - np.nanmedian(c_all[:, ax], 1, keepdims=True)
        tr = np.zeros_like(x)
        for k in np.unique(key):
            m = key == k
            tr[m] = np.nanmedian(x[m], axis=0)
        c_all[:, ax] = x - tr
    rows = []
    t_start = time.time()
    for s in range(0, n, batch):
        f = f_all[s:s + batch]; c = c_all[s:s + batch]
        nb = len(f)
        # detrend: remove slow variations (2 day median)
        fz = np.where(np.isfinite(f), f, np.nan)
        # fill NaN with local interpolation for filtering
        ff = fz.copy()
        for i in range(nb):
            m = np.isfinite(ff[i])
            if m.sum() > 10 and (~m).any():
                ff[i] = np.interp(np.arange(N), np.where(m)[0], ff[i][m])
        slow = nd.median_filter(ff, size=(1, 97), mode='nearest')
        fast = nd.median_filter(ff, size=(1, 9), mode='nearest')
        r = ff - slow
        r[~np.isfinite(fz)] = np.nan
        mask = np.isfinite(fz)
        f0 = np.where(mask, fz, 0.0)
        feat = {}
        sg = robust_sigma(r)
        feat['sigma_hp'] = np.log10(sg)
        feat['std_raw'] = np.log10(np.nanstd(fz, 1) + 1e-9)
        feat['mad_raw'] = np.log10(robust_sigma(fz))
        feat['mad_ratio'] = np.log10(np.nanstd(fz, 1) / robust_sigma(fz))
        feat['skew'] = stats.skew(fz, 1, nan_policy='omit')
        feat['kurt'] = stats.kurtosis(fz, 1, nan_policy='omit')
        feat['skew_hp'] = stats.skew(r, 1, nan_policy='omit')
        feat['kurt_hp'] = stats.kurtosis(r, 1, nan_policy='omit')
        for qq in (0.5, 2, 10, 90, 98, 99.5):
            feat[f'pz{qq}'] = np.nanpercentile(fz, qq, 1) / sg[:] if False else np.nanpercentile(r, qq, 1) / sg
        rz = r / sg[:, None]
        feat['frac_below3'] = np.nanmean(rz < -3, 1)
        feat['frac_above3'] = np.nanmean(rz > 3, 1)
        feat['frac_below5'] = np.nanmean(rz < -5, 1)
        feat['frac_above5'] = np.nanmean(rz > 5, 1)
        feat['min_z'] = np.nanmin(rz, 1); feat['max_z'] = np.nanmax(rz, 1)
        # slow variability amplitude (spots / pulsation)
        feat['slow_std'] = np.log10(np.std(slow, 1) + 1e-9)
        feat['slow_ptp'] = np.log10(np.ptp(slow, 1) + 1e-9)
        feat['slow_over_noise'] = np.log10(np.std(slow, 1) / sg + 1e-9)
        feat['raw_ptp_over_noise'] = np.log10(np.nanmax(fast, 1) - np.nanmin(fast, 1) + 1e-9) - np.log10(sg)
        # diffs / jumps (glitches)
        d = np.diff(ff, axis=1)
        dsig = robust_sigma(d)
        dz = d / dsig[:, None]
        feat['max_jump_z'] = np.abs(dz).max(1)
        feat['neg_jump_z'] = dz.min(1); feat['pos_jump_z'] = dz.max(1)
        sl_d = np.diff(slow, axis=1)
        feat['slow_jump_max'] = np.log10(np.abs(sl_d).max(1) + 1e-9) - np.log10(sg)
        feat['n_big_jumps'] = (np.abs(dz) > 6).sum(1)
        # orbit-wise stats (stripes in one orbit only)
        mid = np.argmax(np.diff(t)) + 1
        s1 = np.nanstd(r[:, :mid], 1); s2 = np.nanstd(r[:, mid:], 1)
        feat['orbit_std_ratio'] = np.log10((np.maximum(s1, s2) + 1e-9) / (np.minimum(s1, s2) + 1e-9))
        m1 = np.nanmedian(fz[:, :mid], 1); m2 = np.nanmedian(fz[:, mid:], 1)
        feat['orbit_med_diff'] = np.abs(m1 - m2) / sg
        # dump-aligned response
        feat['dump_resp'] = (np.nanmean(rz[:, dmask], 1) - np.nanmean(rz[:, ~dmask], 1))
        feat['dump_absresp'] = np.nanmean(np.abs(rz[:, dmask]), 1) / (np.nanmean(np.abs(rz[:, ~dmask]), 1) + 1e-9)
        # flares: positive runs
        pos = (rz > 3) & mask
        runs_n = np.zeros(nb); runs_max = np.zeros(nb); single = np.zeros(nb); decay = np.zeros(nb)
        for i in range(nb):
            p = pos[i]
            if not p.any():
                continue
            idx = np.where(p)[0]
            splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            nf = 0; sg1 = 0; dec = []
            for sp in splits:
                if len(sp) >= 2:
                    nf += 1
                    pk = sp[np.argmax(rz[i, sp])]
                    # decay: mean of 3 points after peak vs 3 before
                    a = rz[i, max(pk - 3, 0):pk]; b = rz[i, pk + 1:pk + 4]
                    if len(a) and len(b):
                        dec.append(np.nanmean(b) - np.nanmean(a))
                else:
                    sg1 += 1
            runs_n[i] = nf; single[i] = sg1
            runs_max[i] = max(rz[i, sp].max() for sp in splits if len(sp) >= 2) if nf else 0
            decay[i] = np.mean(dec) if dec else 0
        feat['n_flare_runs'] = runs_n; feat['n_single_pos'] = single
        feat['flare_max_z'] = runs_max; feat['flare_decay'] = decay
        feat['pos_neg_tail'] = np.nanmean(rz > 3, 1) / (np.nanmean(rz < -3, 1) + 1e-3)
        # spectral on CCD-corrected raw (keep slow + fast)
        feat.update(spectral(f0, mask))
        feat.update(acf_feats(f0, mask))
        # smoothness / alias
        feat['hf_ratio'] = np.log10(np.nanstd(ff - fast, 1) / sg + 1e-9)
        # centroid global stats
        cx = c[:, 0]; cy = c[:, 1]
        for name, cc in (('cx', cx), ('cy', cy)):
            cf = cc.copy()
            ccm = np.isfinite(cf)
            cs = nd.median_filter(np.where(ccm, cf, 0), size=(1, 97), mode='nearest')
            cr = cf - cs
            feat[f'{name}_std'] = np.log10(np.nanstd(cf, 1) + 1e-6)
            feat[f'{name}_sig_hp'] = np.log10(robust_sigma(cr) + 1e-6)
            feat[f'{name}_kurt'] = stats.kurtosis(cr, 1, nan_policy='omit')
            feat[f'{name}_maxz'] = np.nanmax(np.abs(cr), 1) / robust_sigma(cr)
            # correlation of centroid residual with flux residual
            fm = np.where(mask & ccm, r, np.nan)
            a = fm - np.nanmean(fm, 1, keepdims=True); b = np.where(mask & ccm, cr, np.nan); b = b - np.nanmean(b, 1, keepdims=True)
            feat[f'{name}_corr'] = np.nansum(a * b, 1) / (np.sqrt(np.nansum(a * a, 1) * np.nansum(b * b, 1)) + 1e-12)
            feat[f'{name}_jump'] = np.nanmax(np.abs(np.diff(cs, axis=1)), 1) / (robust_sigma(cr) + 1e-9)
        # BLS per star
        bl = []
        cxr = c[:, 0] - nd.median_filter(np.where(np.isfinite(c[:, 0]), c[:, 0], 0), size=(1, 97), mode='nearest')
        cyr = c[:, 1] - nd.median_filter(np.where(np.isfinite(c[:, 1]), c[:, 1], 0), size=(1, 97), mode='nearest')
        cxr[~np.isfinite(c[:, 0])] = np.nan; cyr[~np.isfinite(c[:, 1])] = np.nan
        for i in range(nb):
            bl.append(bls_block(t, r[i], mask[i], cxr[i], cyr[i], dump_idx, orbit_edge))
        bl = pd.DataFrame(bl)
        df = pd.DataFrame(feat)
        df = pd.concat([df, bl], axis=1)
        rows.append(df)
        if verbose:
            print(f'{s + nb}/{n}  {time.time() - t_start:.0f}s', flush=True)
    X = pd.concat(rows, ignore_index=True)
    star_cols = stars[['tmag', 'teff', 'logg', 'radius', 'contamination']].reset_index(drop=True)
    return pd.concat([X, star_cols], axis=1)


if __name__ == '__main__':
    split = sys.argv[1]
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    stars = pd.read_csv(f'{DATA}/{split}_stars.csv')
    cad = pd.read_csv(f'{DATA}/{split}_cadences.csv')
    flux = np.load(f'{DATA}/{split}_flux.npy')
    cen = np.load(f'{DATA}/{split}_centroid.npy')
    if lim:
        # pick stratified-ish subset: ccd trends need many stars so take first lim but keep all for trend via full arrays
        stars = stars.iloc[:lim]; flux = flux[:lim]; cen = cen[:lim]
    X = extract(flux, cen, stars, cad)
    out = f'{DATA}/feat_{split}' + (f'_{lim}' if lim else '') + '.parquet'
    X.to_parquet(out)
    print('saved', out, X.shape)
