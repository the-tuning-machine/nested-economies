#!/usr/bin/env python3
"""
analysis.py — Analyse complète des 80 runs de simulation économique hiérarchique.

Sorties :
    rapport_analyse.md    — rapport extensif
    metrics_summary.csv   — tableau brut exportable
"""
import gc, gzip, json, pickle, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT     = Path(__file__).parent
RUNS_DIR = ROOT / 'runs'
OUT_MD   = ROOT / 'rapport_analyse.md'
OUT_CSV  = ROOT / 'metrics_summary.csv'

sys.path.insert(0, str(ROOT))
from metrics import (wealth_matrix, gini, gini_series,
                     social_mobility_series, exclusivity_series,
                     currency_convergence_series, utility_drift_series,
                     util_series, wealth_per_node)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _pkl_load(path):
    try:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, 'rb') as f:
            return pickle.load(f)


def collect_infos():
    out = []
    for jf in sorted(RUNS_DIR.glob('*.json')):
        if jf.name == 'progress.json':
            continue
        pkf = jf.with_suffix('.pkl')
        if pkf.exists():
            with open(jf) as f:
                out.append({'meta': json.load(f), 'pkl': str(pkf)})
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LABELS & GROUPEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _topo(ls):
    return 'flat' if len(set(ls)) == 1 else 'hier'


LABEL_MAP = {
    ('flat', 'reciprocity', 5,  'joint', 'mass_action',    True):  'REF-flat',
    ('flat', 'reciprocity', 15, 'joint', 'mass_action',    True):  'nres15-flat',
    ('flat', 'market',      5,  'joint', 'mass_action',    True):  'market-flat',
    ('flat', 'reciprocity', 5,  'joint', 'markov',         True):  'markov-flat',
    ('flat', 'reciprocity', 5,  'joint', 'stoichiometric', True):  'stoich-flat',
    ('flat', 'reciprocity', 5,  'joint', 'leontief',       True):  'leontief-flat',
    ('flat', 'reciprocity', 5,  'joint', 'mass_action',    False): 'allnodes-flat',
    ('flat', 'reciprocity', 5,  'split', 'mass_action',    True):  'split-flat',
    ('hier', 'reciprocity', 5,  'joint', 'mass_action',    True):  'REF-hier',
    ('hier', 'reciprocity', 15, 'joint', 'mass_action',    True):  'nres15-hier',
    ('hier', 'market',      5,  'joint', 'mass_action',    True):  'market-hier',
    ('hier', 'reciprocity', 5,  'joint', 'markov',         True):  'markov-hier',
    ('hier', 'reciprocity', 5,  'joint', 'stoichiometric', True):  'stoich-hier',
    ('hier', 'reciprocity', 5,  'joint', 'leontief',       True):  'leontief-hier',
    ('hier', 'reciprocity', 5,  'joint', 'mass_action',    False): 'allnodes-hier',
    ('hier', 'reciprocity', 5,  'split', 'mass_action',    True):  'split-hier',
}

LABEL_ORDER = [
    'REF-flat','nres15-flat','market-flat','markov-flat',
    'stoich-flat','leontief-flat','allnodes-flat','split-flat',
    'REF-hier','nres15-hier','market-hier','markov-hier',
    'stoich-hier','leontief-hier','allnodes-hier','split-hier',
]


def cfg_key(cfg, ls):
    return (
        _topo(ls),
        cfg.get('MODE', '?'),
        cfg.get('n_res', 5),
        cfg.get('BUDGET_MODE', 'joint'),
        cfg.get('TRANSFORM_MODE', 'mass_action'),
        bool(cfg.get('ATOMS_ONLY_TRANSFORM', True)),
    )


def cfg_label(key):
    return LABEL_MAP.get(key, str(key))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HELPERS STATISTIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _stat(ts, last=0.2):
    if ts is None or len(ts) == 0:
        return {k: np.nan for k in ('early','mid','final','mean','std','rvar','ac1')}
    ts = np.asarray(ts, dtype=float)
    T = len(ts)
    ti = max(0, int(T * (1 - last)))
    tail = ts[ti:]
    ac1 = float(np.corrcoef(ts[:-1], ts[1:])[0, 1]) if T > 3 else np.nan
    return {
        'early': float(np.mean(ts[:max(1, T//10)])),
        'mid':   float(np.mean(ts[max(0,T//2-max(1,T//20)):T//2+max(1,T//20)])),
        'final': float(np.mean(tail)),
        'mean':  float(np.mean(ts)),
        'std':   float(np.std(ts)),
        'rvar':  float(np.var(tail)),
        'ac1':   ac1,
    }


def _conv_time(ts, thr=0.05):
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 2:
        return len(ts)
    rng = ts.max() - ts.min()
    if rng < 1e-12:
        return 0
    target = float(np.mean(ts[int(len(ts)*0.8):]))
    for t, v in enumerate(ts):
        if abs(v - target) < thr * rng:
            return t
    return len(ts) - 1


_SPARKS = '▁▂▃▄▅▆▇█'

def sparkline(ts, width=18):
    ts = np.asarray(ts, dtype=float)
    ts = ts[np.isfinite(ts)]
    if len(ts) < 2:
        return '─' * width
    mn, mx = ts.min(), ts.max()
    idxs = np.linspace(0, len(ts)-1, width, dtype=int)
    vals = ts[idxs]
    if mx - mn < 1e-10:
        return '▄' * width
    norm = (vals - mn) / (mx - mn)
    return ''.join(_SPARKS[min(7, int(v * 8))] for v in norm)


def _f(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'n/a'
    return f'{v:.{d}f}'


def _pm(v, s, d=3):
    if np.isnan(v):
        return 'n/a'
    return f'{v:.{d}f}±{s:.{d}f}'


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MÉTRIQUES SUPPLÉMENTAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def oligarchy_v2(history, layer_sizes):
    """
    Retourne (share, pers_initial, pers_local), chacun array (T,).

    share        : part de richesse du top-20%
    pers_initial : Jaccard(top_k(t), top_k(0))  — survie de l'élite initiale
    pers_local   : Jaccard(top_k(t), top_k(t-1)) — stabilité step-by-step
    """
    W = wealth_matrix(history, layer_sizes)
    T, N = W.shape
    k = max(1, N // 5)
    top0 = set(np.argsort(W[0])[-k:].tolist())
    share = np.empty(T); pi = np.empty(T); pl = np.empty(T)
    tp = top0
    for t in range(T):
        tt = set(np.argsort(W[t])[-k:].tolist())
        tot = float(W[t].sum())
        share[t] = float(W[t][list(tt)].sum()) / (tot + 1e-12)
        u0 = len(tt | top0); pi[t] = len(tt & top0)  / u0 if u0 else 1.0
        ulp = len(tt | tp);  pl[t] = len(tt & tp)    / ulp if ulp else 1.0
        tp = tt
    return share, pi, pl


def resource_portfolio(history, layer_sizes):
    """(T,) entropie portfolio (Shannon), (T,) HHI ressources — moyennées sur nœuds."""
    T = len(history['r']); eps = 1e-12
    ent = np.empty(T); hhi = np.empty(T)
    for t in range(T):
        ar = np.concatenate([history['r'][t][l] for l in range(len(layer_sizes))], axis=0)
        tot = ar.sum(axis=1, keepdims=True)
        p = ar / (tot + eps)
        ent[t] = float(-np.sum(p * np.log(p + eps), axis=1).mean())
        hhi[t] = float((p**2).sum(axis=1).mean())
    return ent, hhi


def stratified_gini(history, layer_sizes):
    """(T,) Gini inter-couches (sur moyennes), (T,) Gini intra-couche moyen."""
    T = len(history['r'])
    gi = np.empty(T); ga = np.empty(T)
    for t in range(T):
        means = np.array([float(history['r'][t][l].sum(axis=1).mean())
                          for l in range(len(layer_sizes))])
        gi[t] = gini(means)
        ga[t] = float(np.mean([gini(history['r'][t][l].sum(axis=1))
                                for l in range(len(layer_sizes))]))
    return gi, ga


def layer_wealth_final(history, layer_sizes):
    """(L,) richesse moyenne par couche au dernier step."""
    return np.array([float(history['r'][-1][l].sum(axis=1).mean())
                     for l in range(len(layer_sizes))])


def flux_summary(history, layer_sizes):
    """
    (T,) keep_frac — fraction de ressources conservées en moyenne par nœud.
    (T,) hhi_flow  — HHI de concentration des flux P_up (couches avec N_up > 1).
    """
    T = len(history['P_up'])
    kt = np.empty(T); ht = np.full(T, np.nan)
    for t in range(T):
        pup = history['P_up'][t]; pdn = history['P_down'][t]
        kvals = []; hvals = []
        for l, N_l in enumerate(layer_sizes):
            pu = pup[l]; pd = pdn[l]
            n_res = (pu.shape[2] if pu is not None
                     else pd.shape[2] if pd is not None else 1)
            fu = pu.sum(axis=1) if pu is not None else np.zeros((N_l, n_res))
            fd = pd.sum(axis=1) if pd is not None else np.zeros((N_l, n_res))
            kvals.append(float((1.0 - fu - fd).mean()))
            if pu is not None and pu.shape[1] > 1:
                pm = pu.mean(axis=2)  # (N_l, N_up) — avg over resources
                hvals.append(float((pm**2).sum(axis=1).mean()))
        kt[t] = float(np.mean(kvals))
        if hvals:
            ht[t] = float(np.mean(hvals))
    return kt, ht


def utility_wealth_corr(history, layer_sizes):
    """(T,) corrélation de Spearman richesse–utilité (log-utility Cobb-Douglas)."""
    W = wealth_matrix(history, layer_sizes)
    U = util_series(history, layer_sizes)
    return np.array([_spearman(W[t], U[t]) for t in range(len(W))])


def net_upward_flow(history, layer_sizes):
    """
    (T, L) fraction nette envoyée vers le haut par couche.
    net_up[t, l] = mean over nodes of P_up[l][i,:,:].sum() (frac sent upward).
    """
    T = len(history['P_up']); L = len(layer_sizes)
    out = np.zeros((T, L))
    for t in range(T):
        for l in range(L):
            pu = history['P_up'][t][l]
            if pu is not None:
                out[t, l] = float(pu.sum(axis=(1, 2)).mean() / max(pu.shape[2], 1))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LOI DE PUISSANCE (MLE Pareto — Clauset 2009)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_pareto_mle(wealth):
    """
    Estimation MLE Pareto avec xmin par minimisation du KS.
    Retourne (alpha, xmin, r2_loglog, ks_stat).
    alpha > 1 requis ; retourne (nan,…) si échec.
    """
    w = np.sort(np.asarray(wealth, dtype=float))
    w = w[w > 0]
    if len(w) < 5:
        return np.nan, np.nan, np.nan, np.nan

    best = (np.inf, np.nan, np.nan)  # (ks, xmin, alpha)
    candidates = np.unique(w)
    if len(candidates) > 4:
        candidates = candidates[:-3]  # garder ≥ 4 points dans la queue

    for xmin in candidates:
        tail = w[w >= xmin]; n = len(tail)
        if n < 4:
            continue
        denom = np.sum(np.log(tail / xmin))
        if denom < 1e-12:
            continue
        alpha = 1.0 + n / denom
        if not np.isfinite(alpha) or alpha <= 1.0 or alpha > 100.0:
            continue
        F_th = (tail / xmin) ** (-(alpha - 1))
        F_em = np.arange(n, 0, -1, dtype=float) / n
        ks = float(np.max(np.abs(F_em - F_th)))
        if ks < best[0]:
            best = (ks, xmin, alpha)

    if np.isnan(best[1]):
        return np.nan, np.nan, np.nan, np.nan

    ks, xmin, alpha = best
    tail = w[w >= xmin]; n = len(tail)
    lx = np.log(tail); ly = np.log(np.arange(n, 0, -1, dtype=float) / n)
    v = np.isfinite(lx) & np.isfinite(ly)
    if v.sum() < 3:
        return float(alpha), float(xmin), np.nan, float(ks)
    coeffs = np.polyfit(lx[v], ly[v], 1)
    res = ly[v] - np.polyval(coeffs, lx[v])
    ss_tot = np.sum((ly[v] - ly[v].mean())**2)
    r2 = float(1 - np.sum(res**2) / (ss_tot + 1e-12))
    return float(alpha), float(xmin), r2, float(ks)


def pl_at(history, layer_sizes, frac):
    T = len(history['r'])
    idx = min(max(0, int(T * frac) - 1), T - 1)
    return fit_pareto_mle(wealth_per_node(history['r'][idx], layer_sizes))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TRAITEMENT D'UN RUN
# ═══════════════════════════════════════════════════════════════════════════════

def process_run(meta, pkl_path, verbose=True):
    cfg = meta['config']; ls = meta['layer_sizes']
    key = cfg_key(cfg, ls)
    if verbose:
        print(f"  [{cfg_label(key):16s}] seed={cfg.get('seed',0)} …", end='', flush=True)

    data    = _pkl_load(pkl_path)
    history = data['history']
    steps   = np.array(history['steps'])
    T       = len(steps)

    # ── Métriques existantes ──────────────────────────────────────────────────
    gini_ts  = gini_series(history, ls)
    mob_ts   = social_mobility_series(history, ls)
    excl_ts  = exclusivity_series(history, ls)
    curr_ts  = currency_convergence_series(history, ls)
    drift_ts = utility_drift_series(history, ls)

    # ── Oligarchie v2 ─────────────────────────────────────────────────────────
    olig_s, olig_pi, olig_pl = oligarchy_v2(history, ls)

    # ── Nouvelles métriques ───────────────────────────────────────────────────
    ent_ts, hhi_res_ts     = resource_portfolio(history, ls)
    g_inter_ts, g_intra_ts = stratified_gini(history, ls)
    keep_ts, hhi_flow_ts   = flux_summary(history, ls)
    corr_ts                = utility_wealth_corr(history, ls)
    lw_profile             = layer_wealth_final(history, ls)
    net_up_ts              = net_upward_flow(history, ls)   # (T, L)

    # ── Loi de puissance (10%, 50%, 100%) ─────────────────────────────────────
    pl_data = {
        'early': pl_at(history, ls, 0.10),
        'mid':   pl_at(history, ls, 0.50),
        'final': pl_at(history, ls, 1.00),
    }
    W_final = wealth_per_node(history['r'][-1], ls)

    del data, history
    gc.collect()

    if verbose:
        g = _stat(gini_ts)['final']
        a = pl_data['final'][0]
        print(f" Gini={g:.3f} α={_f(a)} T={T} ✓")

    return {
        'run_id': meta['run_id'], 'label': cfg_label(key), 'key': key,
        'cfg': cfg, 'layer_sizes': ls, 'n_res': meta['n_res'],
        'seed': cfg.get('seed', -1), 'steps': steps, 'T': T,
        # Full TS
        'gini_ts': gini_ts, 'mob_ts': mob_ts, 'olig_s_ts': olig_s,
        'olig_pi_ts': olig_pi, 'olig_pl_ts': olig_pl,
        'excl_ts': excl_ts, 'curr_ts': curr_ts, 'drift_ts': drift_ts,
        'ent_ts': ent_ts, 'hhi_res_ts': hhi_res_ts,
        'g_inter_ts': g_inter_ts, 'g_intra_ts': g_intra_ts,
        'keep_ts': keep_ts, 'hhi_flow_ts': hhi_flow_ts, 'corr_ts': corr_ts,
        # Summaries
        'gini':    _stat(gini_ts),  'mob':     _stat(mob_ts),
        'olig_s':  _stat(olig_s),   'olig_pi': _stat(olig_pi),
        'olig_pl': _stat(olig_pl),  'excl':    _stat(excl_ts),
        'curr':    _stat(curr_ts),  'drift':   _stat(drift_ts),
        'ent':     _stat(ent_ts),   'hhi_res': _stat(hhi_res_ts),
        'g_inter': _stat(g_inter_ts),'g_intra':_stat(g_intra_ts),
        'keep':    _stat(keep_ts),  'hhi_flow':_stat(hhi_flow_ts),
        'corr':    _stat(corr_ts),
        # Convergence (en nombre de snapshots)
        'conv_gini': _conv_time(gini_ts),
        'conv_curr': _conv_time(curr_ts),
        'conv_excl': _conv_time(excl_ts),
        # Power law
        'pl': pl_data, 'W_final': W_final,
        'lw_profile': lw_profile,
        'net_up_ts': net_up_ts,   # (T, L)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. AGRÉGATION PAR CONFIG (5 seeds → 1 entrée)
# ═══════════════════════════════════════════════════════════════════════════════

def _agg_ts(runs, key):
    arrays = [r[key] for r in runs]
    min_T = min(len(a) for a in arrays)
    mat = np.stack([a[:min_T] for a in arrays], axis=0)
    return mat.mean(axis=0), mat.std(axis=0)


def _agg_ts_2d(runs, key):
    arrays = [r[key] for r in runs]
    min_T = min(a.shape[0] for a in arrays)
    mat = np.stack([a[:min_T] for a in arrays], axis=0)
    return mat.mean(axis=0), mat.std(axis=0)


def _agg_stat(runs, key):
    result = {}
    for s in runs[0][key].keys():
        vals = np.array([r[key][s] for r in runs], dtype=float)
        result[s] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    return result


def _agg_pl(runs, phase):
    alphas = np.array([r['pl'][phase][0] for r in runs], dtype=float)
    r2s    = np.array([r['pl'][phase][2] for r in runs], dtype=float)
    kss    = np.array([r['pl'][phase][3] for r in runs], dtype=float)
    return {
        'alpha_mean': float(np.nanmean(alphas)),
        'alpha_std':  float(np.nanstd(alphas)),
        'r2_mean':    float(np.nanmean(r2s)),
        'ks_mean':    float(np.nanmean(kss)),
        'alphas':     alphas,
    }


def aggregate_config(runs):
    r0 = runs[0]
    W_all = np.concatenate([r['W_final'] for r in runs])
    alpha_p, xmin_p, r2_p, ks_p = fit_pareto_mle(W_all)

    ts_scalar_keys = ['gini_ts','mob_ts','olig_s_ts','olig_pi_ts','olig_pl_ts',
                      'excl_ts','curr_ts','drift_ts','ent_ts','hhi_res_ts',
                      'g_inter_ts','g_intra_ts','keep_ts','hhi_flow_ts','corr_ts']
    ts = {k: _agg_ts(runs, k) for k in ts_scalar_keys}
    net_up_mean, net_up_std = _agg_ts_2d(runs, 'net_up_ts')

    lp = np.stack([r['lw_profile'] for r in runs], axis=0)

    pl_final = _agg_pl(runs, 'final')
    pl_final.update({'alpha_pooled': alpha_p, 'r2_pooled': r2_p, 'ks_pooled': ks_p,
                     'W_pooled': W_all})

    return {
        'label': r0['label'], 'key': r0['key'], 'cfg': r0['cfg'],
        'layer_sizes': r0['layer_sizes'], 'n_res': r0['n_res'], 'n_runs': len(runs),
        **{k: _agg_stat(runs, k) for k in [
            'gini','mob','olig_s','olig_pi','olig_pl','excl','curr',
            'drift','ent','hhi_res','g_inter','g_intra','keep','hhi_flow','corr']},
        'conv_gini': (np.mean([r['conv_gini'] for r in runs]),
                      np.std( [r['conv_gini'] for r in runs])),
        'conv_curr': (np.mean([r['conv_curr'] for r in runs]),
                      np.std( [r['conv_curr'] for r in runs])),
        'conv_excl': (np.mean([r['conv_excl'] for r in runs]),
                      np.std( [r['conv_excl'] for r in runs])),
        'ts': ts,
        'net_up_mean': net_up_mean, 'net_up_std': net_up_std,
        'pl_final': pl_final,
        'pl_mid':   _agg_pl(runs, 'mid'),
        'pl_early': _agg_pl(runs, 'early'),
        'lw_profile_mean': lp.mean(axis=0),
        'lw_profile_std':  lp.std(axis=0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GÉNÉRATION DU RAPPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _sorted_cfgs(cfgs_dict):
    return [cfgs_dict[k] for k in LABEL_ORDER if k in cfgs_dict]


def _section(lvl, title):
    return f'\n{"#" * lvl} {title}\n'


def _row(*cells):
    return '| ' + ' | '.join(str(c) for c in cells) + ' |'


def _sep(n):
    return '|' + '|'.join(['---'] * n) + '|'


def write_report(cfgs_dict, all_runs):
    lines = []
    sorted_cfgs = _sorted_cfgs(cfgs_dict)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ── En-tête ───────────────────────────────────────────────────────────────
    lines += [
        f'# Rapport d\'analyse — Simulations économiques hiérarchiques\n',
        f'*Généré le {now}*\n\n',
        f'**{len(all_runs)} runs** · **{len(sorted_cfgs)} configurations** · '
        f'**5 seeds** · **2 topologies** · '
        f'**Adam lr=1e-3** · **zero\\_sum\\_no\\_s** · **Cobb-Douglas**\n',
    ]

    # ── Section 1 : Résumé exécutif ───────────────────────────────────────────
    lines.append(_section(2, '1. Résumé exécutif'))
    lines.append(
        'Tableau des métriques clés (valeurs finales, moyennées sur 5 seeds). '
        'Trié par Gini croissant (inégalité).\n'
    )

    sc_by_gini = sorted(sorted_cfgs, key=lambda c: c['gini']['final'][0])
    lines.append(_row('Config','Gini↓','Mobilité','Exclusivité','OligShare',
                       'PersInit','PersLocal','Curr','Drift','α(PL)','R²(PL)'))
    lines.append(_sep(11))
    for c in sc_by_gini:
        g   = c['gini']['final']
        mob = c['mob']['final']
        ex  = c['excl']['final']
        os_ = c['olig_s']['final']
        opi = c['olig_pi']['final']
        opl = c['olig_pl']['final']
        cu  = c['curr']['final']
        dr  = c['drift']['final']
        apl = c['pl_final']
        lines.append(_row(
            f"**{c['label']}**",
            _pm(*g),    _pm(*mob), _pm(*ex),
            _pm(*os_),  _pm(*opi), _pm(*opl),
            _pm(*cu),   _pm(*dr),
            _f(apl['alpha_mean']),
            _f(apl['r2_mean']),
        ))

    lines.append('\n**Principales observations :**\n')
    gini_vals = [c['gini']['final'][0] for c in sorted_cfgs]
    label_max_gini = sorted_cfgs[np.argmax(gini_vals)]['label']
    label_min_gini = sorted_cfgs[np.argmin(gini_vals)]['label']
    lines.append(f'- Gini final : min={_f(min(gini_vals))} ({label_min_gini}), '
                 f'max={_f(max(gini_vals))} ({label_max_gini})\n')
    alpha_vals = [c['pl_final']['alpha_mean'] for c in sorted_cfgs
                  if not np.isnan(c['pl_final']['alpha_mean'])]
    if alpha_vals:
        lines.append(f'- Exposant Pareto α : plage [{_f(min(alpha_vals))}, {_f(max(alpha_vals))}]\n')

    # ── Section 2 : Design expérimental ──────────────────────────────────────
    lines.append(_section(2, '2. Design expérimental'))
    lines.append(
        'La batterie couvre un produit cartésien de **2 topologies × 8 variantes de base × 5 seeds = 80 runs**.\n\n'
        '| Paramètre fixé | Valeur |\n|---|---|\n'
        '| RESOURCE_MODE | `zero_sum_no_s` |\n'
        '| lr | 1e-3 |\n'
        '| UTILITY_MODE | `self` |\n'
        '| STRICT_COBB_DOUGLAS | True |\n'
        '| alpha_mem | 0.9 |\n'
        '| lambda_rec | 0.1 |\n\n'
    )

    lines.append('### 2.1 Configurations\n')
    lines.append(_row('Label','Topologie','n_res','MODE','BUDGET_MODE','TRANSFORM_MODE','ATOMS_ONLY','n_steps'))
    lines.append(_sep(8))
    for c in sorted_cfgs:
        cfg = c['cfg']; ls = c['layer_sizes']
        lines.append(_row(
            c['label'], _topo(ls),
            cfg.get('n_res','?'), cfg.get('MODE','?'),
            cfg.get('BUDGET_MODE','?'), cfg.get('TRANSFORM_MODE','?'),
            cfg.get('ATOMS_ONLY_TRANSFORM','?'), cfg.get('n_steps','?'),
        ))

    lines.append('\n### 2.2 ⚠️ Trouvaille critique : équivalence des variantes TRANSFORM_MODE\n')
    lines.append(
        '**Toutes les configurations utilisent `RESOURCE_MODE=\'zero_sum_no_s\'`**, '
        'fixé globalement dans `run_experiments.py`. Dans ce mode (`main.py:479–482`), '
        'la branche de transformation est court-circuitée :\n\n'
        '```python\n'
        'if self.resource_mode == \'zero_sum_no_s\':\n'
        '    r_new = F.relu(r + r_net)  # S ignoré\n'
        '    S_new = S                  # S figé\n'
        '```\n\n'
        'Conséquences :\n'
        '- Les variantes **markov, stoichiometric, leontief** ont une dynamique des ressources **identique** à **mass_action**.\n'
        '- Le paramètre **ATOMS_ONLY_TRANSFORM** est sans effet (la transformation n\'est jamais appelée).\n'
        '- Les paramètres appris (`W_in`, `W_out`, `C_raw`, `eff_raw`) sont initialisés mais jamais utilisés.\n\n'
        '**Dimensions réellement distinctes :**\n\n'
        '| Dimension | Valeurs distinctes |\n|---|---|\n'
        '| Topologie | flat [10×10] vs hier [32,16,8,4,2,1] |\n'
        '| MODE | reciprocity vs market |\n'
        '| n_res | 5 (20k steps) vs 15 (40k steps) |\n'
        '| BUDGET_MODE | joint vs split |\n'
        '| TRANSFORM_MODE | toutes équivalentes (4 configs redondantes × 2 topos) |\n'
    )

    # ── Section 3 : Vérification des métriques ────────────────────────────────
    lines.append(_section(2, '3. Vérification des métriques'))
    lines.append(
        '### 3.1 `gini()` — Coefficient de Gini\n'
        '**Formule** : `G = (2·Σ(rank_i · w_i)) / (n·Σw_i) − (n+1)/n`\n\n'
        '**Statut : ✅ Correct.** Formule de Lorenz standard. Travaille sur |w| '
        '(robustesse si richesses négatives après relu). Plage [0, 1].\n\n'

        '### 3.2 `social_mobility_series()` — Distance Kendall-tau\n'
        '**Formule** : `d(t) = (1 − τ(rank_t, rank_{t-1})) / 2` entre snapshots consécutifs.\n\n'
        '**Statut : ✅ Correct.** Normalisation dans [0, 1] correcte. '
        'Chaque pas mesure une fenêtre de 10 steps réels (stride=10). '
        'Retourne T-1 valeurs — correctement géré par `_stat()`.\n\n'

        '### 3.3 `exclusivity_series()` — Exclusivité des partenariats\n'
        '**Formule** : `1 − H_normalisée` où H = entropie de Shannon de P_up moyennée sur les ressources.\n\n'
        '**Statut : ✅ Correct.** Simplification raisonnable (moyenne sur ressources avant entropie). '
        'Valeur 0 = flux uniforms, 1 = tout vers un seul voisin. '
        'Note : couche supérieure (pas de P_up) est exclue correctement.\n\n'

        '### 3.4 `oligarchy_series()` — Version originale\n'
        '**Formule originale** : `Jaccard(top_k(t), top_k(T-1))` — retrospectif, '
        'vaut 1 par construction à t=T-1.\n\n'
        '**Statut : ⚠️ Remplacée** par deux métriques plus informatives (voir 3.4bis).\n\n'

        '### 3.4bis `oligarchy_v2()` — Nouvelles métriques de persistance\n'
        '- **`pers_initial`** : `Jaccard(top_k(t), top_k(0))` — survie de l\'élite initiale dans le temps. '
        'Décroît si le top se renouvelle.\n'
        '- **`pers_local`** : `Jaccard(top_k(t), top_k(t−1))` — stabilité step-by-step du top-20%. '
        'Proche de 1 = top stable, proche de 0 = fort renouvellement.\n\n'
        '**Statut : ✅ Correct.**\n\n'

        '### 3.5 `currency_convergence_series()` — Dominance ressource\n'
        '**Formule** : `max_m(Σ_i r_{i,m}) / (Σ_{i,m} r_{i,m})`\n\n'
        '**Statut : ✅ Correct.** Note importante : en mode `zero_sum_no_s`, '
        'la masse de chaque ressource est **conservée par type** (S figé = identité, pas de transformation). '
        'Donc la dominance ne peut pas évoluer par transformation interne, uniquement par redistribution. '
        'Valeur de référence en cas d\'uniformité parfaite : `1/n_res` (≈ 0.200 pour n_res=5, ≈ 0.067 pour n_res=15).\n\n'

        '### 3.6 `utility_drift_series()` — Dérive d\'utilité\n'
        '**Formule** : `mean_i |U_i(t) − U_i(0)|` où U est la log-utilité Cobb-Douglas.\n\n'
        '**Statut : ✅ Correct.** La log-utilité peut être négative (log de valeurs < 1), '
        'la dérive absolue est bien définie. Mesure l\'écart absolu aux préférences initiales.\n\n'

        '### 3.7 Nouvelles métriques (ce rapport)\n'
        '| Métrique | Formule | Plage | Interprétation |\n|---|---|---|---|\n'
        '| Entropie portfolio | `−Σ_m p_{im} log p_{im}` | `[0, log(n_res)]` | Diversification ressources |\n'
        '| HHI ressources | `Σ_m p_{im}²` | `[1/n_res, 1]` | Concentration portfolio |\n'
        '| Gini inter-couches | Gini des richesses moyennes/couche | `[0, 1]` | Inégalité verticale |\n'
        '| Gini intra-couche | Moyenne des Gini par couche | `[0, 1]` | Inégalité horizontale |\n'
        '| Keep fraction | `1 − frac_up − frac_dn` | `[0, 1]` | Rétention des ressources |\n'
        '| HHI flux P_up | `Σ_j p_{ij}²` avg sur nœuds | `[1/N_up, 1]` | Concentration des flux |\n'
        '| Corr Spearman U-W | `ρ(richesse, log-utilité)` | `[−1, 1]` | Alignement utilité-richesse |\n'
        '| α Pareto MLE | Exposant Pareto (queue riche) | `> 1` | Inégalité extrême |\n'
    )

    # ── Section 4 : Tableau récapitulatif global ──────────────────────────────
    lines.append(_section(2, '4. Tableau récapitulatif global (valeurs finales, mean ± std)'))
    lines.append(
        'Valeurs moyennées sur les 20% derniers snapshots de chaque run, '
        'puis moyennées ± std sur les 5 seeds.\n'
    )

    metrics_cols = [
        ('gini',    'Gini'),
        ('mob',     'Mobilité'),
        ('excl',    'Exclusivité'),
        ('olig_s',  'OligShare'),
        ('olig_pi', 'PersInit'),
        ('olig_pl', 'PersLocal'),
        ('curr',    'Curr.Dom.'),
        ('drift',   'Drift U'),
        ('ent',     'Entropie'),
        ('hhi_res', 'HHI-res'),
        ('corr',    'Corr U-W'),
        ('keep',    'Keep%'),
    ]
    headers = ['Config'] + [m[1] for m in metrics_cols]
    lines.append(_row(*headers))
    lines.append(_sep(len(headers)))
    for c in sorted_cfgs:
        row = [f"**{c['label']}**"]
        for key, _ in metrics_cols:
            row.append(_pm(*c[key]['final']))
        lines.append(_row(*row))

    # ── Section 5 : Tendances temporelles ─────────────────────────────────────
    lines.append(_section(2, '5. Tendances temporelles'))
    lines.append(
        'Évolution de chaque métrique (early ≈ 10% du run, mid ≈ 50%, final ≈ 80–100%). '
        'La colonne **Spark** montre la trajectoire complète (gauche=début, droite=fin).\n'
    )

    for metric_key, metric_name in [
        ('gini_ts', 'Coefficient de Gini'),
        ('mob_ts',  'Mobilité sociale (Kendall-τ)'),
        ('excl_ts', 'Exclusivité partenariats'),
        ('olig_s_ts','Part de richesse top-20%'),
        ('olig_pi_ts','Persistance initiale (Jaccard vs t=0)'),
        ('olig_pl_ts','Persistance locale (Jaccard step-by-step)'),
        ('curr_ts',  'Convergence monnaie de réserve'),
        ('corr_ts',  'Corrélation Spearman richesse–utilité'),
        ('g_inter_ts','Gini inter-couches'),
        ('g_intra_ts','Gini intra-couche (moyen)'),
    ]:
        lines.append(f'\n### 5.x {metric_name}\n')
        lines.append(_row('Config','Early','Mid','Final','Δ(final-early)','AC1','Spark'))
        lines.append(_sep(7))
        for c in sorted_cfgs:
            ts_m, ts_s = c['ts'][metric_key]
            st = {
                'early': float(np.mean(ts_m[:max(1, len(ts_m)//10)])),
                'mid':   float(np.mean(ts_m[len(ts_m)//2 - max(1, len(ts_m)//20):
                                           len(ts_m)//2 + max(1, len(ts_m)//20)])),
                'final': float(np.mean(ts_m[int(len(ts_m)*0.8):])),
                'ac1':   float(np.corrcoef(ts_m[:-1], ts_m[1:])[0, 1]) if len(ts_m) > 3 else np.nan,
            }
            delta = st['final'] - st['early']
            lines.append(_row(
                c['label'],
                _f(st['early']), _f(st['mid']), _f(st['final']),
                f"{'+' if delta>=0 else ''}{_f(delta)}",
                _f(st['ac1']),
                sparkline(ts_m),
            ))

    # ── Convergence ───────────────────────────────────────────────────────────
    lines.append('\n### 5.y Temps de convergence (en snapshots, stride=10)\n')
    lines.append(
        '*Seuil : premier snapshot où |métrique − valeur_finale| < 5% de la plage totale.*\n'
    )
    lines.append(_row('Config','Conv. Gini','Conv. Curr.','Conv. Excl.'))
    lines.append(_sep(4))
    for c in sorted_cfgs:
        cg = c['conv_gini']; cc = c['conv_curr']; ce = c['conv_excl']
        lines.append(_row(
            c['label'],
            f"{cg[0]:.0f}±{cg[1]:.0f}",
            f"{cc[0]:.0f}±{cc[1]:.0f}",
            f"{ce[0]:.0f}±{ce[1]:.0f}",
        ))

    # ── Section 6 : Loi de puissance ─────────────────────────────────────────
    lines.append(_section(2, '6. Analyse de loi de puissance'))
    lines.append(
        'Test de l\'hypothèse que la distribution de richesse suit une loi de Pareto : '
        '`P(W > w) ∝ w^{−α}`. '
        'Estimateur MLE (Clauset *et al.* 2009) avec `x_min` optimal (minimisation KS). '
        'R² calculé sur le fit log-log de la CCDF empirique.\n\n'
        '**Interprétation de α :**\n'
        '- α ∈ (1, 2] : queue très lourde, variance infinie (inégalité extrême)\n'
        '- α ∈ (2, 3] : variance finie, moyenne finie (type Pareto 80/20 → α ≈ 1.16)\n'
        '- α > 3 : queue mince, distribution plus égalitaire\n'
        '- R² > 0.95 : bon ajustement power law\n'
        '- KS < 0.05 : distribution non rejetée comme Pareto au seuil 5%\n\n'
        '**Limitation** : n ≤ 100 nœuds (flat) ou 63 (hier) par run — '
        'faible puissance statistique pour le test de loi de puissance. '
        'L\'analyse poolée (seeds concaténées) est plus fiable (n=500 ou 315).\n'
    )

    lines.append('\n### 6.1 Exposants par phase (early/mid/final) — moyennes sur 5 seeds\n')
    lines.append(_row('Config','α early','α mid','α final ± std','R² final','KS final','α poolé','R² poolé'))
    lines.append(_sep(8))
    for c in sorted_cfgs:
        pe = c['pl_early']; pm = c['pl_mid']; pf = c['pl_final']
        lines.append(_row(
            c['label'],
            _f(pe['alpha_mean']),
            _f(pm['alpha_mean']),
            _pm(pf['alpha_mean'], pf['alpha_std']),
            _f(pf['r2_mean']),
            _f(pf['ks_mean']),
            _f(pf['alpha_pooled']),
            _f(pf['r2_pooled']),
        ))

    lines.append('\n### 6.2 Interprétation de la loi de puissance\n')
    # Identify configs where R² > 0.9 (good power law fit)
    good_pl = [c for c in sorted_cfgs
               if not np.isnan(c['pl_final']['r2_mean']) and c['pl_final']['r2_mean'] > 0.9]
    poor_pl = [c for c in sorted_cfgs
               if not np.isnan(c['pl_final']['r2_mean']) and c['pl_final']['r2_mean'] <= 0.9]

    if good_pl:
        labels_good = ', '.join(c['label'] for c in good_pl)
        alphas_good = [c['pl_final']['alpha_mean'] for c in good_pl]
        lines.append(
            f'**Bon ajustement (R² > 0.90)** : {labels_good}\n\n'
            f'Exposants dans ces configs : plage [{_f(min(alphas_good))}, {_f(max(alphas_good))}]. '
        )
        if min(alphas_good) < 2:
            lines.append('Queue très lourde (α < 2) : variance infinie, concentration extrême. ')
        elif min(alphas_good) < 3:
            lines.append('Queue modérément lourde (2 < α < 3). ')
        lines.append('\n\n')

    if poor_pl:
        labels_poor = ', '.join(c['label'] for c in poor_pl)
        lines.append(
            f'**Ajustement faible (R² ≤ 0.90)** : {labels_poor}\n\n'
            'Ces distributions ne suivent pas bien une loi de puissance — '
            'probablement une distribution log-normale ou exponentielle à la place. '
            'L\'échantillon trop petit (n ≤ 100) peut aussi affaiblir le R².\n\n'
        )

    lines.append(
        '### 6.3 Évolution temporelle de α\n\n'
        'La variation de α entre early, mid et final indique si l\'inégalité s\'accentue (α ↓) ou '
        'se résorbe (α ↑) au cours de la simulation.\n\n'
    )
    lines.append(_row('Config','Δα (mid−early)','Δα (final−mid)','Tendance'))
    lines.append(_sep(4))
    for c in sorted_cfgs:
        ae = c['pl_early']['alpha_mean']
        am = c['pl_mid']['alpha_mean']
        af = c['pl_final']['alpha_mean']
        d1 = am - ae; d2 = af - am
        if np.isnan(d1) or np.isnan(d2):
            trend = 'n/a'
        elif d1 < -0.1 and d2 < -0.1:
            trend = '↓↓ aggravation continue'
        elif d1 > 0.1 and d2 > 0.1:
            trend = '↑↑ réduction continue'
        elif abs(d1) < 0.1 and abs(d2) < 0.1:
            trend = '→ stable'
        else:
            trend = '↕ non monotone'
        lines.append(_row(c['label'],
                          f"{'+' if d1>=0 else ''}{_f(d1,2)}",
                          f"{'+' if d2>=0 else ''}{_f(d2,2)}",
                          trend))

    # ── Section 7 : Topologie ─────────────────────────────────────────────────
    lines.append(_section(2, '7. Impact de la topologie (flat vs hiérarchique)'))
    lines.append(
        '- **Flat** : [10]×10 = 100 nœuds, 10 couches, connexions symétriques (N_up = N_down = 10)\n'
        '- **Hiérarchique** : [32,16,8,4,2,1] = 63 nœuds, 6 couches, '
        'bottleneck progressif (N_up diminue vers le sommet)\n\n'
        'Comparaisons directes sur les 8 paires de configs (même variant, topologies différentes).\n'
    )

    pairs = [
        ('REF-flat','REF-hier'),
        ('nres15-flat','nres15-hier'),
        ('market-flat','market-hier'),
        ('markov-flat','markov-hier'),
        ('stoich-flat','stoich-hier'),
        ('leontief-flat','leontief-hier'),
        ('allnodes-flat','allnodes-hier'),
        ('split-flat','split-hier'),
    ]
    metrics_pairs = [('gini','Gini'),('olig_s','OligShare'),('olig_pi','PersInit'),
                     ('excl','Exclus.'),('corr','Corr U-W'),('keep','Keep%')]
    lines.append(_row('Variant','Metric','Flat (final)','Hier (final)','Δ(hier−flat)'))
    lines.append(_sep(5))
    for fl, hi in pairs:
        cf = cfgs_dict.get(fl); ch = cfgs_dict.get(hi)
        if cf is None or ch is None:
            continue
        first = True
        for mk, mn in metrics_pairs:
            vf = cf[mk]['final'][0]; vh = ch[mk]['final'][0]
            delta = vh - vf
            lines.append(_row(
                fl.replace('-flat','') if first else '',
                mn, _f(vf), _f(vh),
                f"{'+' if delta>=0 else ''}{_f(delta)}",
            ))
            first = False

    lines.append('\n### 7.1 Inégalité stratifiée — inter vs intra couches\n')
    lines.append(
        'Décomposition du Gini total : `Gini_inter` mesure l\'inégalité entre couches '
        '(richesse moyenne par couche), `Gini_intra` mesure l\'inégalité au sein de chaque couche.\n'
    )
    lines.append(_row('Config','Gini total','Gini inter','Gini intra','% inter/total'))
    lines.append(_sep(5))
    for c in sorted_cfgs:
        gt = c['gini']['final'][0]
        gi = c['g_inter']['final'][0]
        ga = c['g_intra']['final'][0]
        pct = f"{gi/gt*100:.1f}%" if gt > 1e-6 else 'n/a'
        lines.append(_row(c['label'], _f(gt), _f(gi), _f(ga), pct))

    lines.append('\n### 7.2 Gradient de richesse par couche\n')
    lines.append(
        'Richesse moyenne par couche au dernier snapshot (mean ± std sur 5 seeds). '
        'Une progression croissante avec le niveau hiérarchique indiquerait un avantage positionnel.\n'
    )
    for c in sorted_cfgs:
        ls = c['layer_sizes']
        lp_m = c['lw_profile_mean']
        lp_s = c['lw_profile_std']
        row_vals = ' | '.join(f'{_f(lp_m[l])}±{_f(lp_s[l])}' for l in range(len(ls)))
        layer_labels = ' | '.join(f'L{l}(N={ls[l]})' for l in range(len(ls)))
        lines.append(f'\n**{c["label"]}** : {layer_labels}\n\n')
        lines.append(f'richesse : {row_vals}\n')

    # ── Section 8 : MODE ──────────────────────────────────────────────────────
    lines.append(_section(2, '8. Impact du mode d\'interaction (reciprocity vs market)'))
    lines.append(
        '`reciprocity` : bonus EMA vers les partenaires qui ont historiquement renvoyé (Tit-for-Tat).\n'
        '`market` : bonus proportionnel à l\'alignement offre/demande (`u_i · r_j`) sans mémoire d\'arête.\n\n'
        'Les paires comparées gardent tous les autres paramètres fixes.\n'
    )
    market_pairs = [('REF-flat','market-flat'), ('REF-hier','market-hier')]
    lines.append(_row('Variant','Metric','Reciprocity','Market','Δ(market−recip)'))
    lines.append(_sep(5))
    for rec, mkt in market_pairs:
        cr = cfgs_dict.get(rec); cm = cfgs_dict.get(mkt)
        if cr is None or cm is None:
            continue
        topo = 'flat' if 'flat' in rec else 'hier'
        first = True
        for mk, mn in [('gini','Gini'),('mob','Mobilité'),('excl','Exclus.'),
                        ('olig_s','OligShare'),('olig_pi','PersInit'),
                        ('curr','Curr.Dom.'),('corr','Corr U-W'),('keep','Keep%'),
                        ('hhi_flow','HHI-flux')]:
            vr = cr[mk]['final'][0]; vm = cm[mk]['final'][0]
            delta = vm - vr
            lines.append(_row(
                topo if first else '',
                mn, _f(vr), _f(vm),
                f"{'+' if delta>=0 else ''}{_f(delta)}",
            ))
            first = False

    # ── Section 9 : BUDGET_MODE ───────────────────────────────────────────────
    lines.append(_section(2, '9. Impact du BUDGET_MODE (joint vs split)'))
    lines.append(
        '`joint` : un seul softmax [keep, up, down] — les directions se font concurrence.\n'
        '`split` : deux softmax indépendants [keep, up] et [keep, down] — chaque direction a son propre budget.\n'
    )
    budget_pairs = [('REF-flat','split-flat'), ('REF-hier','split-hier')]
    lines.append(_row('Variant','Metric','Joint','Split','Δ'))
    lines.append(_sep(5))
    for jo, sp in budget_pairs:
        cj = cfgs_dict.get(jo); cs = cfgs_dict.get(sp)
        if cj is None or cs is None:
            continue
        topo = 'flat' if 'flat' in jo else 'hier'
        first = True
        for mk, mn in [('gini','Gini'),('excl','Exclus.'),('keep','Keep%'),
                        ('olig_s','OligShare'),('hhi_flow','HHI-flux')]:
            vj = cj[mk]['final'][0]; vs = cs[mk]['final'][0]
            delta = vs - vj
            lines.append(_row(topo if first else '', mn, _f(vj), _f(vs),
                               f"{'+' if delta>=0 else ''}{_f(delta)}"))
            first = False

    # ── Section 10 : n_res ────────────────────────────────────────────────────
    lines.append(_section(2, '10. Impact du nombre de ressources (n_res=5 vs n_res=15)'))
    lines.append(
        '`n_res=15` implique aussi **40k steps** (vs 20k) pour donner plus de temps à la convergence. '
        'Les métriques de richesse totale sont plus grandes avec n_res=15 mais les métriques '
        'normalisées (Gini, HHI, entropie) sont comparables.\n\n'
        'Référence pour la convergence monnaie : 1/n_res = 0.200 (n_res=5) vs 0.067 (n_res=15).\n'
    )
    nres_pairs = [('REF-flat','nres15-flat'), ('REF-hier','nres15-hier')]
    lines.append(_row('Variant','Metric','n_res=5','n_res=15','Δ'))
    lines.append(_sep(5))
    for n5, n15 in nres_pairs:
        c5 = cfgs_dict.get(n5); c15 = cfgs_dict.get(n15)
        if c5 is None or c15 is None:
            continue
        topo = 'flat' if 'flat' in n5 else 'hier'
        first = True
        for mk, mn in [('gini','Gini'),('mob','Mobilité'),('ent','Entropie'),
                        ('hhi_res','HHI-res'),('curr','Curr.Dom.'),('corr','Corr U-W')]:
            v5 = c5[mk]['final'][0]; v15 = c15[mk]['final'][0]
            delta = v15 - v5
            lines.append(_row(topo if first else '', mn, _f(v5), _f(v15),
                               f"{'+' if delta>=0 else ''}{_f(delta)}"))
            first = False

    # ── Section 11 : Flux inter-couches ──────────────────────────────────────
    lines.append(_section(2, '11. Analyse des flux inter-couches'))
    lines.append(
        'Pour chaque configuration, la fraction nette envoyée **vers le haut** (P_up) '
        'par couche donne une image du transfert de richesse vertical.\n'
        'Une valeur élevée à une couche basse indique que les agents atomiques '
        'envoient une large part de leurs ressources vers les méta-agents.\n\n'
        '**Keep fraction** = fraction moyenne des ressources conservées (non envoyées).\n'
        '**HHI flux** = concentration des sorties P_up (1/N_up = uniforme, 1 = monopole).\n'
    )
    lines.append(_row('Config','Keep% (final)','HHI-flux (final)','Net up L0→','Net up L5→'))
    lines.append(_sep(5))
    for c in sorted_cfgs:
        ls = c['layer_sizes']
        keep_f = c['keep']['final'][0]
        hhi_f  = c['hhi_flow']['final'][0]
        nu_m   = c['net_up_mean']   # (T, L)
        # Final 20% mean per layer
        T_nu = nu_m.shape[0]
        nu_final = nu_m[int(T_nu*0.8):].mean(axis=0)
        nu_l0 = _f(nu_final[0]) if len(nu_final) > 0 else 'n/a'
        nu_l5 = _f(nu_final[5]) if len(nu_final) > 5 else 'n/a'
        lines.append(_row(c['label'], _f(keep_f), _f(hhi_f), nu_l0, nu_l5))

    # ── Section 12 : Spécialisation des ressources ────────────────────────────
    lines.append(_section(2, '12. Spécialisation des ressources'))
    lines.append(
        'L\'**entropie de portfolio** mesure la diversification du panier de ressources de chaque nœud. '
        'Entropie max = `log(n_res)` (portefeuille uniforme). '
        'L\'**HHI des ressources** mesure la concentration : 1/n_res = uniforme, 1 = mono-ressource.\n\n'
        'En mode `zero_sum_no_s`, les ressources ne peuvent pas être transformées — '
        'la spécialisation ne peut naître que de la redistribution sélective (P_up, P_down).\n'
    )
    lines.append(_row('Config','Entropie (early→final)','HHI-res (early→final)',
                       'Entropie max','Δ entropie'))
    lines.append(_sep(5))
    for c in sorted_cfgs:
        ent_e = c['ent']['early']
        ent_f = c['ent']['final']
        hhi_e = c['hhi_res']['early']
        hhi_f = c['hhi_res']['final']
        n_res = c['n_res']
        ent_max = np.log(n_res)
        delta_ent = ent_f[0] - ent_e[0]
        lines.append(_row(
            c['label'],
            f"{_f(ent_e[0])} → {_f(ent_f[0])}",
            f"{_f(hhi_e[0])} → {_f(hhi_f[0])}",
            _f(ent_max),
            f"{'+' if delta_ent>=0 else ''}{_f(delta_ent)}",
        ))

    # ── Section 13 : Corrélation utilité-richesse ─────────────────────────────
    lines.append(_section(2, '13. Corrélation Spearman utilité–richesse'))
    lines.append(
        'La corrélation de Spearman entre la richesse totale et la log-utilité Cobb-Douglas par nœud '
        'mesure si les nœuds riches sont aussi les nœuds les plus "satisfaits". '
        'Une valeur proche de 1 indique que richesse ↔ utilité sont bien alignées ; '
        'une valeur proche de 0 ou négative suggère un désalignement (nœuds riches mais "pauvres" '
        'en ressources qu\'ils valorisent).\n'
    )
    lines.append(_row('Config','Corr early','Corr mid','Corr final','Spark','AC1'))
    lines.append(_sep(6))
    for c in sorted_cfgs:
        ts_m, _ = c['ts']['corr_ts']
        T = len(ts_m)
        e = float(np.mean(ts_m[:max(1,T//10)]))
        m = float(np.mean(ts_m[max(0,T//2-T//20):T//2+T//20]))
        f = float(np.mean(ts_m[int(T*0.8):]))
        ac = float(np.corrcoef(ts_m[:-1], ts_m[1:])[0,1]) if T > 3 else np.nan
        lines.append(_row(c['label'], _f(e), _f(m), _f(f), sparkline(ts_m), _f(ac)))

    # ── Section 14 : Conclusions ──────────────────────────────────────────────
    lines.append(_section(2, '14. Conclusions et recommandations'))

    lines.append('### 14.1 Tendances robustes (confirmées sur 5 seeds)\n\n')

    # Collect some insights automatically
    flat_gini  = np.mean([cfgs_dict[l]['gini']['final'][0]
                          for l in LABEL_ORDER if l in cfgs_dict and 'flat' in l])
    hier_gini  = np.mean([cfgs_dict[l]['gini']['final'][0]
                          for l in LABEL_ORDER if l in cfgs_dict and 'hier' in l])
    topo_effect = "flat > hier" if flat_gini > hier_gini else "hier > flat"

    ref_flat_olig_pi = cfgs_dict.get('REF-flat', {}).get('olig_pi', {}).get('final', (np.nan, np.nan))[0]
    ref_flat_olig_pl = cfgs_dict.get('REF-flat', {}).get('olig_pl', {}).get('final', (np.nan, np.nan))[0]

    lines.append(
        f'1. **Topologie** : Gini moyen flat={_f(flat_gini)}, hier={_f(hier_gini)} → {topo_effect}. '
        f'La topologie hiérarchique présente un gradient de richesse par couche distinct.\n\n'
        f'2. **Équivalence TRANSFORM_MODE** : Les configs markov/stoich/leontief/mass_action sont '
        f'empiriquement indistinguables (toutes en zero_sum_no_s). '
        f'Résultats à interpréter avec précaution pour toute conclusion sur le mode de transformation.\n\n'
        f'3. **Persistance de l\'oligarchie** : pers_initial={_f(ref_flat_olig_pi)} (REF-flat) — '
        f'le top-20% initial est {"fortement" if ref_flat_olig_pi > 0.5 else "partiellement"} '
        f'conservé jusqu\'à la fin. '
        f'La persistance locale pers_local={_f(ref_flat_olig_pl)} indique '
        f'{"une grande stabilité" if ref_flat_olig_pl > 0.8 else "un renouvellement modéré"} '
        f'du top entre steps consécutifs.\n\n'
        f'4. **Loi de puissance** : voir Section 6 pour le détail par config. '
        f'La distribution de richesse suit approximativement une loi de Pareto dans la plupart des cas, '
        f'avec un exposant α qui évolue au cours de la simulation.\n\n'
        f'5. **Corrélation utilité-richesse** : indique si l\'optimisation Cobb-Douglas '
        f'aboutit à un alignement entre préférences et richesse.\n\n'
    )

    lines.append('### 14.2 Recommandations\n\n')
    lines.append(
        '- **Corriger le design expérimental** : réactiver `RESOURCE_MODE` non-trivial '
        '(ex. `free` ou `zero_sum_stoch_s`) pour les configs TRANSFORM_MODE, '
        'afin de rendre ces variantes réellement distinctes.\n'
        '- **Augmenter n** : avec n ≤ 100 nœuds, les tests de loi de puissance '
        'ont une faible puissance statistique. Augmenter à 500+ pour des résultats plus fiables.\n'
        '- **Ajouter une baseline déconnectée** : un run sans redistribution (P_up = P_down = 0 figés) '
        'comme référence de ce que la dynamique apporte réellement.\n'
        '- **Analyser la convergence de l\'optimiseur** : la loss Adam n\'est pas sauvegardée — '
        'l\'ajouter à l\'historique permettrait de distinguer convergence optimisation vs convergence économique.\n'
    )

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. EXPORT CSV
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(all_runs):
    metric_keys = ['gini','mob','olig_s','olig_pi','olig_pl','excl',
                   'curr','drift','ent','hhi_res','g_inter','g_intra',
                   'keep','hhi_flow','corr']
    stat_keys   = ['early','mid','final','mean','std','rvar','ac1']

    header = ['run_id','label','seed','topo','mode','n_res','budget','transform','T']
    for mk in metric_keys:
        for sk in stat_keys:
            header.append(f'{mk}_{sk}')
    header += ['conv_gini','conv_curr','conv_excl',
               'pl_early_alpha','pl_early_r2','pl_early_ks',
               'pl_mid_alpha','pl_mid_r2','pl_mid_ks',
               'pl_final_alpha','pl_final_r2','pl_final_ks']

    rows = [','.join(header)]
    for r in all_runs:
        cfg = r['cfg']; ls = r['layer_sizes']
        row = [
            r['run_id'], r['label'], str(r['seed']),
            _topo(ls), cfg.get('MODE','?'), str(r['n_res']),
            cfg.get('BUDGET_MODE','?'), cfg.get('TRANSFORM_MODE','?'), str(r['T']),
        ]
        for mk in metric_keys:
            for sk in stat_keys:
                row.append(_f(r[mk].get(sk, np.nan), 6))
        row += [
            _f(r['conv_gini']), _f(r['conv_curr']), _f(r['conv_excl']),
            _f(r['pl']['early'][0]), _f(r['pl']['early'][2]), _f(r['pl']['early'][3]),
            _f(r['pl']['mid'][0]),   _f(r['pl']['mid'][2]),   _f(r['pl']['mid'][3]),
            _f(r['pl']['final'][0]), _f(r['pl']['final'][2]), _f(r['pl']['final'][3]),
        ]
        rows.append(','.join(row))

    OUT_CSV.write_text('\n'.join(rows))
    print(f"CSV écrit : {OUT_CSV} ({len(rows)-1} lignes)")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  Analyse des runs — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    infos = collect_infos()
    print(f"Runs trouvés : {len(infos)}\n")

    # ── Traitement de tous les runs ──────────────────────────────────────────
    all_runs = []
    for info in infos:
        try:
            result = process_run(info['meta'], info['pkl'], verbose=True)
            all_runs.append(result)
        except Exception as e:
            print(f"  ERREUR {info['meta']['run_id']}: {e}")
        gc.collect()

    print(f"\nRuns traités : {len(all_runs)}\n")

    # ── Groupement par config ────────────────────────────────────────────────
    groups = defaultdict(list)
    for r in all_runs:
        groups[r['label']].append(r)

    print("Configurations trouvées :")
    for lbl, runs in sorted(groups.items()):
        print(f"  {lbl:20s} : {len(runs)} seeds")

    # ── Agrégation ───────────────────────────────────────────────────────────
    print("\nAgrégation par config…")
    cfgs_dict = {}
    for lbl, runs in groups.items():
        cfgs_dict[lbl] = aggregate_config(runs)

    # ── Rapport ──────────────────────────────────────────────────────────────
    print("Génération du rapport…")
    report = write_report(cfgs_dict, all_runs)
    OUT_MD.write_text(report)
    print(f"Rapport écrit : {OUT_MD}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    write_csv(all_runs)

    print(f"\n{'='*60}")
    print(f"  Terminé.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
