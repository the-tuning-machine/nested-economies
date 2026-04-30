"""
metrics.py — Calcul des métriques économiques sur l'historique de simulation.

Toutes les fonctions prennent `history` (dict r/util/P_up/P_down) et `layer_sizes`,
et retournent des arrays numpy ou des figures Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy.stats import kendalltau as _scipy_kendalltau
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ─── Primitives ───────────────────────────────────────────────────────────────

def wealth_per_node(r_snap, layer_sizes):
    """Richesse totale de chaque nœud = Σ_m r_{i,m}. Retourne array (N_total,)."""
    return np.concatenate([r_snap[l].sum(axis=1) for l in range(len(layer_sizes))])


def gini(wealth):
    """Coefficient de Gini (0 = égalité parfaite, 1 = concentration totale)."""
    w = np.sort(np.abs(np.asarray(wealth, dtype=float)))
    n = len(w)
    if n == 0 or w.sum() < 1e-12:
        return 0.0
    return float((2 * (np.arange(1, n + 1) * w).sum()) / (n * w.sum()) - (n + 1) / n)


def ecdf(data):
    """CDF empirique. Retourne (x_sorted, cumulative_prob)."""
    x = np.sort(np.asarray(data, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


# ─── Séries temporelles ───────────────────────────────────────────────────────

def wealth_matrix(history, layer_sizes):
    """Matrice (T, N_total) de richesse totale par nœud à chaque step."""
    return np.array([
        wealth_per_node(history['r'][t], layer_sizes)
        for t in range(len(history['r']))
    ])


def gini_series(history, layer_sizes):
    """Array (T,) du coefficient de Gini à chaque step."""
    W = wealth_matrix(history, layer_sizes)
    return np.array([gini(W[t]) for t in range(len(W))])


def util_series(history, layer_sizes):
    """Matrice (T, N_total) d'utilité par nœud à chaque step."""
    return np.array([
        np.concatenate([history['util'][t][l] for l in range(len(layer_sizes))])
        for t in range(len(history['util']))
    ])


# ─── Figures ──────────────────────────────────────────────────────────────────

_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def _member_runs(run_data):
    """Retourne les runs membres d'un groupe agrégé, ou le run lui-même."""
    return run_data.get('members', [run_data])


def _mean_series(series_list):
    """Moyenne élément par élément de séries numériques de même longueur."""
    return np.mean(np.stack([np.asarray(s, dtype=float) for s in series_list], axis=0), axis=0)


def fig_wealth_distribution(runs_data, scope='last'):
    """
    Distribution de richesse (histogramme + ECDF).

    runs_data : list de {'run_id': str, 'history': dict, 'layer_sizes': list}
    scope     : 'last' (dernier état) | 'all' (tous les états empilés)
    """
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Histogramme', 'CDF empirique'),
                        horizontal_spacing=0.12)

    for k, rd in enumerate(runs_data):
        color = _PALETTE[k % len(_PALETTE)]
        label = rd['run_id']
        samples = []
        for member in _member_runs(rd):
            W = wealth_matrix(member['history'], member['layer_sizes'])
            samples.append(W[-1] if scope == 'last' else W.flatten())

        if scope == 'last':
            scope_lbl = 'dernier état'
        else:
            scope_lbl = 'tous les états'
        w = np.concatenate(samples) if samples else np.array([], dtype=float)

        # Histogramme
        fig.add_trace(go.Histogram(
            x=w, name=label, opacity=0.55, nbinsx=40,
            marker_color=color, showlegend=True,
        ), row=1, col=1)

        # ECDF
        xs, ys = ecdf(w)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines', name=label,
            line=dict(color=color, width=2), showlegend=False,
        ), row=1, col=2)

    fig.update_layout(
        title=f'<b>Distribution des richesses</b> — {scope_lbl}',
        barmode='overlay',
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(orientation='h', y=-0.15, font_size=11),
    )
    fig.update_xaxes(title_text='Richesse (Σ ressources)', row=1, col=1)
    fig.update_yaxes(title_text='Fréquence', row=1, col=1)
    fig.update_xaxes(title_text='Richesse', row=1, col=2)
    fig.update_yaxes(title_text='P(W ≤ w)', row=1, col=2, range=[0, 1])
    fig.update_xaxes(showticklabels=False, ticks='')
    fig.update_yaxes(showticklabels=False, ticks='')
    return fig


def fig_gini(runs_data, scope='last'):
    """
    Coefficient de Gini.

    scope='last'  → valeur scalaire par run (bar chart)
    scope='all'   → évolution temporelle (ligne par run)
    """
    fig = go.Figure()

    for k, rd in enumerate(runs_data):
        color = _PALETTE[k % len(_PALETTE)]
        label = rd['run_id']
        gs = _mean_series([
            gini_series(member['history'], member['layer_sizes'])
            for member in _member_runs(rd)
        ])

        if scope == 'last':
            fig.add_trace(go.Bar(
                x=[label], y=[float(gs[-1])],
                name=label, marker_color=color,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=list(range(len(gs))), y=gs.tolist(),
                mode='lines', name=label,
                line=dict(color=color, width=2),
            ))

    title = ('<b>Coefficient de Gini</b> — dernier état'
             if scope == 'last' else '<b>Coefficient de Gini</b> au fil du temps')
    fig.update_layout(
        title=title,
        xaxis_title='Run' if scope == 'last' else 'Step',
        yaxis=dict(title='Gini', range=[0, 1]),
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=55, r=30, t=55, b=50),
        legend=dict(orientation='h', y=-0.2, font_size=11),
    )
    fig.update_xaxes(showticklabels=False, ticks='')
    fig.update_yaxes(showticklabels=False, ticks='')
    return fig


# ─── Mobilité sociale ─────────────────────────────────────────────────────────

def _kendall_tau_distance(rk_a, rk_b):
    """Kendall-tau entre deux vecteurs de rangs, normalisé dans [0,1]."""
    if _HAS_SCIPY:
        tau = _scipy_kendalltau(rk_a, rk_b).statistic
        return float((1.0 - tau) / 2.0)
    # Fallback O(n²) sans scipy
    n = len(rk_a)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s = int(np.sign(rk_a[i] - rk_a[j])) * int(np.sign(rk_b[i] - rk_b[j]))
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = concordant + discordant
    return float(discordant / total) if total > 0 else 0.0


def social_mobility_series(history, layer_sizes):
    """Array (T-1,) de mobilité entre chaque paire de steps consécutifs."""
    W = wealth_matrix(history, layer_sizes)
    T = len(W)
    if T < 2:
        return np.array([0.0])
    out = []
    rk_prev = np.argsort(np.argsort(W[0]))
    for t in range(1, T):
        rk_t = np.argsort(np.argsort(W[t]))
        out.append(_kendall_tau_distance(rk_prev, rk_t))
        rk_prev = rk_t
    return np.array(out)


def fig_social_mobility(runs_data, scope='last'):
    """
    Mobilité sociale (volatilité des rangs de richesse).

    scope='last'  → mobilité moyenne par run (bar chart)
    scope='all'   → mobilité step-by-step (ligne par run)
    """
    fig = go.Figure()

    for k, rd in enumerate(runs_data):
        color = _PALETTE[k % len(_PALETTE)]
        label = rd['run_id']
        ms = _mean_series([
            social_mobility_series(member['history'], member['layer_sizes'])
            for member in _member_runs(rd)
        ])

        if scope == 'last':
            fig.add_trace(go.Bar(
                x=[label], y=[float(ms.mean())],
                name=label, marker_color=color,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(ms) + 1)), y=ms.tolist(),
                mode='lines', name=label,
                line=dict(color=color, width=2),
            ))

    title = ('<b>Mobilité sociale</b> — moyenne'
             if scope == 'last' else '<b>Mobilité sociale</b> au fil du temps')
    fig.update_layout(
        title=title,
        xaxis_title='Run' if scope == 'last' else 'Step',
        yaxis=dict(title='Distance Kendall-tau (0=stable, 1=chaos)',
                   rangemode='nonnegative'),
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=55, r=30, t=55, b=50),
        legend=dict(orientation='h', y=-0.2, font_size=11),
    )
    fig.update_xaxes(showticklabels=False, ticks='')
    fig.update_yaxes(showticklabels=False, ticks='')
    return fig


# ─── Exclusivité des partenariats ─────────────────────────────────────────────

def _exclusivity_at_step(p_up_t, layer_sizes):
    """Exclusivité moyenne à un step donné. p_up_t = liste de matrices ou None."""
    entropies = []
    for l in range(len(layer_sizes) - 1):
        mat = p_up_t[l]
        if mat is None:
            continue
        p = mat.mean(axis=2)          # (N_l, N_{l+1})
        N_up = p.shape[1]
        if N_up < 2:
            continue
        log_norm = np.log(N_up)
        for i in range(p.shape[0]):
            row = p[i]
            row = row / (row.sum() + 1e-12)
            h = -float(np.sum(row * np.log(row + 1e-12)))
            entropies.append(h / log_norm)
    if not entropies:
        return 0.0
    return float(1.0 - np.mean(entropies))


def exclusivity_series(history, layer_sizes):
    """Array (T,) d'exclusivité de partenariat à chaque step."""
    T = len(history['P_up'])
    return np.array([_exclusivity_at_step(history['P_up'][t], layer_sizes)
                     for t in range(T)])


def fig_exclusivity(runs_data, scope='last'):
    """
    Exclusivité des partenariats (concentration des flux sortants).

    0 = flux uniformes, 1 = tout vers un seul voisin.
    scope='last'  → scalaire par run (bar chart)
    scope='all'   → série temporelle
    """
    fig = go.Figure()

    for k, rd in enumerate(runs_data):
        color = _PALETTE[k % len(_PALETTE)]
        label = rd['run_id']
        es = _mean_series([
            exclusivity_series(member['history'], member['layer_sizes'])
            for member in _member_runs(rd)
        ])

        if scope == 'last':
            fig.add_trace(go.Bar(
                x=[label], y=[float(es[-1])],
                name=label, marker_color=color,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=list(range(len(es))), y=es.tolist(),
                mode='lines', name=label,
                line=dict(color=color, width=2),
            ))

    title = ('<b>Exclusivité des partenariats</b> — dernier état'
             if scope == 'last' else '<b>Exclusivité des partenariats</b> au fil du temps')
    fig.update_layout(
        title=title,
        xaxis_title='Run' if scope == 'last' else 'Step',
        yaxis=dict(title='Exclusivité (0=uniforme, 1=monopole)', range=[0, 1]),
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=55, r=30, t=55, b=50),
        legend=dict(orientation='h', y=-0.2, font_size=11),
    )
    fig.update_xaxes(showticklabels=False, ticks='')
    fig.update_yaxes(showticklabels=False, ticks='')
    return fig


# ─── Oligarchie ───────────────────────────────────────────────────────────────

def oligarchy_series(history, layer_sizes):
    """
    Retourne (share_series, persistence_series), chacun array (T,).

    share       : part de richesse des top-20% à chaque step
    persistence : Jaccard entre top-k(t) et top-k(T-1)
    """
    W = wealth_matrix(history, layer_sizes)
    T, N = W.shape
    k = max(1, N // 5)
    top_final = set(np.argsort(W[-1])[-k:].tolist())

    share = np.zeros(T)
    persistence = np.zeros(T)
    for t in range(T):
        top_t = set(np.argsort(W[t])[-k:].tolist())
        total = float(W[t].sum())
        share[t] = float(W[t][list(top_t)].sum()) / (total + 1e-12)
        inter = len(top_t & top_final)
        union = len(top_t | top_final)
        persistence[t] = inter / union if union > 0 else 1.0
    return share, persistence


def fig_oligarchy(runs_data, scope='last'):
    """
    Oligarchie : part de richesse et persistance du top-20%.

    scope='last'  → deux bar charts (subplots)
    scope='all'   → deux séries temporelles
    """
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Part de richesse (top 20%)',
                                        'Persistance du top 20%'),
                        horizontal_spacing=0.14)

    for k, rd in enumerate(runs_data):
        color = _PALETTE[k % len(_PALETTE)]
        label = rd['run_id']
        metrics = [
            oligarchy_series(member['history'], member['layer_sizes'])
            for member in _member_runs(rd)
        ]
        share = _mean_series([m[0] for m in metrics])
        persist = _mean_series([m[1] for m in metrics])

        if scope == 'last':
            fig.add_trace(go.Bar(x=[label], y=[float(share[-1])],
                                 name=label, marker_color=color, showlegend=True),
                          row=1, col=1)
            fig.add_trace(go.Bar(x=[label], y=[float(persist[-1])],
                                 name=label, marker_color=color, showlegend=False),
                          row=1, col=2)
        else:
            ts = list(range(len(share)))
            fig.add_trace(go.Scatter(x=ts, y=share.tolist(), mode='lines',
                                     name=label, line=dict(color=color, width=2),
                                     showlegend=True),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=ts, y=persist.tolist(), mode='lines',
                                     name=label, line=dict(color=color, width=2),
                                     showlegend=False),
                          row=1, col=2)

    scope_lbl = 'dernier état' if scope == 'last' else 'au fil du temps'
    fig.update_layout(
        title=f'<b>Oligarchie</b> — {scope_lbl}',
        barmode='group',
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=55, r=30, t=60, b=50),
        legend=dict(orientation='h', y=-0.2, font_size=11),
    )
    fig.update_yaxes(title_text='Part de richesse', range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text='Jaccard vs top final', range=[0, 1], row=1, col=2)
    if scope == 'all':
        fig.update_xaxes(title_text='Step', row=1, col=1)
        fig.update_xaxes(title_text='Step', row=1, col=2)
    fig.update_xaxes(showticklabels=False, ticks='')
    fig.update_yaxes(showticklabels=False, ticks='')
    return fig


# ─── Convergence vers une monnaie de réserve ─────────────────────────────────

def currency_convergence_series(history, layer_sizes):
    """
    Array (T,) — dominance de la ressource la plus répandue dans la circulation.

    dominance(t) = max_m(Σ_i r_{i,m}) / (Σ_{i,m} r_{i,m} + ε)
    """
    T = len(history['r'])
    out = np.zeros(T)
    for t in range(T):
        r_snap = history['r'][t]
        stacked = np.concatenate([r_snap[l] for l in range(len(layer_sizes))], axis=0)
        total_per_res = stacked.sum(axis=0)
        grand_total = total_per_res.sum()
        out[t] = float(total_per_res.max()) / float(grand_total + 1e-12)
    return out


def fig_currency_convergence(runs_data, scope='last'):
    """
    Convergence vers une monnaie de réserve.

    scope='last'  → dominance finale par run (bar chart)
    scope='all'   → évolution temporelle
    """
    fig = go.Figure()

    for k, rd in enumerate(runs_data):
        color = _PALETTE[k % len(_PALETTE)]
        label = rd['run_id']
        cs = _mean_series([
            currency_convergence_series(member['history'], member['layer_sizes'])
            for member in _member_runs(rd)
        ])

        if scope == 'last':
            fig.add_trace(go.Bar(
                x=[label], y=[float(cs[-1])],
                name=label, marker_color=color,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=list(range(len(cs))), y=cs.tolist(),
                mode='lines', name=label,
                line=dict(color=color, width=2),
            ))

    title = ('<b>Convergence monnaie de réserve</b> — dernier état'
             if scope == 'last' else '<b>Convergence monnaie de réserve</b> au fil du temps')
    fig.update_layout(
        title=title,
        xaxis_title='Run' if scope == 'last' else 'Step',
        yaxis=dict(title='Dominance (1/n_res=uniforme, 1=monopole)',
                   range=[0, 1]),
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=55, r=30, t=55, b=50),
        legend=dict(orientation='h', y=-0.2, font_size=11),
    )
    fig.update_xaxes(showticklabels=False, ticks='')
    fig.update_yaxes(showticklabels=False, ticks='')
    return fig


# ─── Drift des utilités ───────────────────────────────────────────────────────

def utility_drift_series(history, layer_sizes):
    """
    Array (T,) — dérive moyenne des valeurs d'utilité par rapport à l'état initial.

    drift(t) = (1/N) Σ_i |U_i(t) - U_i(0)|
    """
    U = util_series(history, layer_sizes)   # (T, N)
    T = len(U)
    if T == 0:
        return np.array([0.0])
    u0 = U[0]
    return np.array([float(np.abs(U[t] - u0).mean()) for t in range(T)])


def fig_utility_drift(runs_data, scope='last'):
    """
    Utility drift par rapport aux valeurs initiales (t=0).

    scope='last'  → valeur finale par run (bar chart)
    scope='all'   → évolution temporelle
    """
    fig = go.Figure()

    for k, rd in enumerate(runs_data):
        color = _PALETTE[k % len(_PALETTE)]
        label = rd['run_id']
        ds = _mean_series([
            utility_drift_series(member['history'], member['layer_sizes'])
            for member in _member_runs(rd)
        ])

        if scope == 'last':
            fig.add_trace(go.Bar(
                x=[label], y=[float(ds[-1])],
                name=label, marker_color=color,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=list(range(len(ds))), y=ds.tolist(),
                mode='lines', name=label,
                line=dict(color=color, width=2),
            ))

    title = ('<b>Utility drift</b> — dernier état'
             if scope == 'last' else '<b>Utility drift</b> au fil du temps')
    fig.update_layout(
        title=title,
        xaxis_title='Run' if scope == 'last' else 'Step',
        yaxis=dict(title='|U(t) - U(0)| moyen', rangemode='nonnegative'),
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=55, r=30, t=55, b=50),
        legend=dict(orientation='h', y=-0.2, font_size=11),
    )
    fig.update_xaxes(showticklabels=False, ticks='')
    fig.update_yaxes(showticklabels=False, ticks='')
    return fig
