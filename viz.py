"""
viz.py — Visualisation interactive de la simulation EconomicHierarchicalDAG

Lancer en standalone (dans un terminal séparé) :
    python viz.py                               # charge l'interface vide
    python viz.py runs/20240101_123456_abc.pkl  # pré-charge un run spécifique
    python viz.py --open-browser                # ouvre aussi le navigateur
"""

import argparse
import gzip
import os
import sys
import json
import glob
import pickle
import threading
import time
import webbrowser

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go

import metrics as M


# ─── État global (rechargé depuis fichier) ────────────────────────────────────

_snap_lock = threading.Lock()
_snapshot  = {
    'path':  'runs/latest.pkl',
    'mtime': 0.0,
    'data':  None,
}


def _reload_if_needed():
    path = _snapshot['path']
    if not path or not os.path.exists(path):
        return
    try:
        mtime = os.path.getmtime(path)
        if mtime <= _snapshot['mtime']:
            return
        try:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
        except (OSError, gzip.BadGzipFile):
            with open(path, 'rb') as f:
                data = pickle.load(f)
        with _snap_lock:
            _snapshot['data']  = data
            _snapshot['mtime'] = mtime
    except Exception:
        pass


def _get_data():
    with _snap_lock:
        return _snapshot['data']

def _history():
    d = _get_data()
    return d['history'] if d else {'r': [], 'u': [], 'util': [], 'P_up': [], 'P_down': [], 'steps': []}


def _steps(history):
    """Retourne les vrais numéros de step pour l'axe x (compatibilité avant/après HIST_STRIDE)."""
    s = history.get('steps')
    if s:
        return list(s)
    return list(range(len(history['r'])))

def _layer_sizes():
    d = _get_data()
    return d['layer_sizes'] if d else []

def _n_res():
    d = _get_data()
    return d['n_res'] if d else 1

def _run_id():
    d = _get_data()
    return d.get('run_id', '—') if d else '—'


# ─── Recalcul des ressources et utilités en mode agrégé ──────────────────────

def _compute_aggregate_state(history, strict_cd=False):
    """
    En mode agrégé, r et util de chaque nœud l≥1 incluent la contribution
    pondérée des enfants (couche l-1) via P_up. Calcul sur les r originaux.
    Requiert history['u']. Retourne (r_eff, util_eff).
    """
    T = len(history['r'])
    r_eff_all, util_eff_all = [], []
    for t in range(T):
        r_frame, util_frame = [], []
        for l in range(len(history['r'][t])):
            r = np.asarray(history['r'][t][l], dtype=np.float64)
            u = np.asarray(history['u'][t][l], dtype=np.float64)
            row_sum = u.sum(axis=1, keepdims=True)
            u = u / np.where(row_sum < 1e-8, 1.0, row_sum)

            if l > 0:
                p = history['P_up'][t][l - 1]
                if p is not None:
                    p = np.asarray(p, dtype=np.float64)
                    denom = p.sum(axis=1, keepdims=True).clip(1e-8)
                    alpha = p / denom
                    r_ch = np.asarray(history['r'][t][l - 1], dtype=np.float64)
                    r_eval = r + np.einsum('kim,km->im', alpha, r_ch)
                else:
                    r_eval = r
            else:
                r_eval = r

            if strict_cd:
                log_r = np.log(np.clip(r_eval, 1e-8, None))
            else:
                log_r = np.log1p(np.clip(r_eval, 0.0, None))

            r_frame.append(r_eval.astype(np.float32))
            util_frame.append((u * log_r).sum(axis=1).astype(np.float32))
        r_eff_all.append(r_frame)
        util_eff_all.append(util_frame)
    return r_eff_all, util_eff_all


def _get_effective_history(history, aggregate, strict_cd=False):
    """r et util augmentés si aggregate=True et history['u'] disponible."""
    if not aggregate or not history.get('u'):
        return history
    r_eff, util_eff = _compute_aggregate_state(history, strict_cd)
    return {**history, 'r': r_eff, 'util': util_eff}


# ─── Cache de positions ───────────────────────────────────────────────────────

_pos_cache = {'key': None, 'pos': {}}

def _pos():
    ls = tuple(_layer_sizes())
    if ls != _pos_cache['key']:
        _pos_cache['key'] = ls
        _pos_cache['pos'] = compute_positions(list(ls)) if ls else {}
    return _pos_cache['pos']


# ─── Positions des nœuds ─────────────────────────────────────────────────────

def compute_positions(layer_sizes):
    pos = {}
    for l, n in enumerate(layer_sizes):
        for i in range(n):
            pos[(l, i)] = (l * 5.0, (i - (n - 1) / 2.0) * 2.5)
    return pos


# ─── Couleurs ─────────────────────────────────────────────────────────────────

_DIR = {
    'out_up':   dict(rgba='rgba( 31,119,180,{a})', label='Sortant ↑'),
    'out_down': dict(rgba='rgba( 44,160, 44,{a})', label='Sortant ↓'),
    'in_below': dict(rgba='rgba(255,127, 14,{a})', label='Entrant ↓'),
    'in_above': dict(rgba='rgba(214, 39, 40,{a})', label='Entrant ↑'),
}

_BINS_NORMAL = [
    dict(lo=0.00, hi=0.25, color='rgba(180,180,180,0.10)', width=0.5),
    dict(lo=0.25, hi=0.50, color='rgba(253,174, 97,0.40)', width=1.2),
    dict(lo=0.50, hi=0.75, color='rgba(244,109, 67,0.65)', width=2.2),
    dict(lo=0.75, hi=1.01, color='rgba(165,  0, 38,0.90)', width=3.5),
]
_BINS_FADED = [
    dict(lo=0.00, hi=1.01, color='rgba(180,180,180,0.07)', width=0.4),
]

_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00',
           '#984ea3', '#a65628', '#f781bf', '#999999']


# ─── Figures simulation ───────────────────────────────────────────────────────

def make_network_fig(t, history, layer_sizes, n_res, pos, selected_node=None):
    n_layers = len(layer_sizes)
    util_t   = history['util'][t]
    r_t      = history['r'][t]
    p_up_t   = history['P_up'][t]
    p_down_t = history['P_down'][t]

    all_u = np.concatenate(util_t)
    vmin, vmax = float(all_u.min()), float(all_u.max())
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0

    traces = []
    bins = _BINS_FADED if selected_node is not None else _BINS_NORMAL

    for l in range(n_layers - 1):
        p_up = p_up_t[l]
        if p_up is None:
            continue
        p_mean   = p_up.mean(axis=2)
        edge_max = float(p_mean.max()) or 1.0
        edges = [(i, j, float(p_mean[i, j]) / edge_max)
                 for i in range(layer_sizes[l])
                 for j in range(layer_sizes[l + 1])]
        for b in bins:
            ex, ey = [], []
            for i, j, v in edges:
                if b['lo'] <= v < b['hi']:
                    x0, y0 = pos[(l, i)]
                    x1, y1 = pos[(l + 1, j)]
                    ex += [x0, x1, None]
                    ey += [y0, y1, None]
            if ex:
                traces.append(go.Scatter(x=ex, y=ey, mode='lines',
                                         line=dict(color=b['color'], width=b['width']),
                                         hoverinfo='none', showlegend=False))

    if selected_node is not None:
        l_s, i_s = int(selected_node[0]), int(selected_node[1])
        N_up   = layer_sizes[l_s + 1] if l_s < n_layers - 1 else 0
        N_down = layer_sizes[l_s - 1] if l_s > 0 else 0

        def _draw_connections(nbr_layer, nbr_count, p_out, p_in, color_rgba):
            if nbr_count == 0:
                return
            props = []
            for nb in range(nbr_count):
                v_out = float(p_out[nb]) if p_out is not None else 0.0
                v_in  = float(p_in[nb])  if p_in  is not None else 0.0
                props.append((v_out + v_in) / 2.0)
            p_max = max(props) or 1.0
            for nb, p in enumerate(props):
                inten = p / p_max
                if inten < 0.03:
                    continue
                x0, y0 = pos[(l_s, i_s)]
                x1, y1 = pos[(nbr_layer, nb)]
                traces.append(go.Scatter(
                    x=[x0, x1], y=[y0, y1], mode='lines',
                    line=dict(color=color_rgba.format(a=0.3 + 0.65 * inten),
                              width=0.8 + 5.2 * inten),
                    hoverinfo='none', showlegend=False,
                ))

        p_out_up = p_up_t[l_s][i_s].mean(axis=1)           if (N_up  and p_up_t[l_s]     is not None) else None
        p_in_up  = p_down_t[l_s+1][:, i_s, :].mean(axis=1) if (N_up  and l_s < n_layers-1 and p_down_t[l_s+1] is not None) else None
        _draw_connections(l_s + 1, N_up,   p_out_up, p_in_up,  _DIR['out_up']['rgba'])

        p_out_dn = p_down_t[l_s][i_s].mean(axis=1)          if (N_down and p_down_t[l_s]   is not None) else None
        p_in_dn  = p_up_t[l_s-1][:, i_s, :].mean(axis=1)    if (N_down and l_s > 0         and p_up_t[l_s-1]  is not None) else None
        _draw_connections(l_s - 1, N_down, p_out_dn, p_in_dn, _DIR['out_down']['rgba'])

        for key, lbl in [('out_up', '↑ Haut'), ('out_down', '↓ Bas')]:
            traces.append(go.Scatter(x=[None], y=[None], mode='lines',
                                     line=dict(color=_DIR[key]['rgba'].format(a=0.9), width=3),
                                     name=lbl, showlegend=True))

    nx_, ny_, nc_, ntxt_, ncd_ = [], [], [], [], []
    for l in range(n_layers):
        for i in range(layer_sizes[l]):
            x, y = pos[(l, i)]
            u = float(util_t[l][i])
            r = r_t[l][i]
            res_str = '  '.join(f'r{k}={r[k]:.2f}' for k in range(n_res))
            nx_.append(x); ny_.append(y); nc_.append(u)
            ntxt_.append(f"<b>Nœud ({l},{i})</b><br>Utilité : {u:.4f}<br>{res_str}")
            ncd_.append([l, i, -1, 0])

    traces.append(go.Scatter(
        x=nx_, y=ny_, mode='markers',
        marker=dict(size=18, color=nc_, colorscale='RdYlGn', cmin=vmin, cmax=vmax,
                    showscale=True,
                    colorbar=dict(title='Utilité', thickness=12, len=0.55, x=1.03,
                                  tickfont=dict(size=10)),
                    line=dict(width=0)),
        text=ntxt_, customdata=ncd_,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False, name='nodes',
    ))

    if selected_node is not None:
        l_s, i_s = int(selected_node[0]), int(selected_node[1])
        xs, ys = pos[(l_s, i_s)]
        traces.append(go.Scatter(
            x=[xs], y=[ys], mode='markers',
            marker=dict(size=26, color='rgba(0,0,0,0)',
                        line=dict(color='gold', width=3)),
            hoverinfo='none', showlegend=False,
        ))

    annotations = [
        dict(x=pos[(l, 0)][0], y=pos[(l, layer_sizes[l]-1)][1] + 3.5,
             text=f'<b>C{l}</b>', showarrow=False,
             font=dict(size=11, color='#555'), xanchor='center')
        for l in range(n_layers)
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f'<b>Réseau — Step {t} / {len(history["r"]) - 1}</b>',
                   font_size=13),
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=5, r=70, t=40, b=5),
        hovermode='closest', annotations=annotations,
        legend=dict(orientation='h', x=0, y=-0.02, font_size=11,
                    bgcolor='rgba(255,255,255,0.8)'),
        uirevision='network',
    )
    return fig


def _step_x(history, t_index):
    ts = _steps(history)
    if not ts:
        return 0
    idx = min(max(int(t_index or 0), 0), len(ts) - 1)
    return ts[idx]


def make_global_fig(history, layer_sizes, n_res, t_current=0):
    ts = _steps(history)
    mean_utils = [float(np.concatenate(history['util'][i]).mean()) for i in range(len(ts))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=mean_utils, mode='lines',
                             name='Utilité moyenne',
                             line=dict(color='black', width=2.5)))
    fig.add_vline(x=_step_x(history, t_current), line_dash='dot',
                  line_color='gray', line_width=1.5)
    fig.update_layout(
        title='<b>Utilité moyenne du système</b>',
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=55, r=20, t=55, b=40),
        xaxis_title='Step', yaxis_title='Utilité moyenne',
        showlegend=False,
    )
    return fig


def make_edge_evolution_fig(edge_ids, t_current, history, layer_sizes, n_res):
    n_avail = len(history['r'])
    x_vals  = _steps(history)

    if not edge_ids or n_avail == 0:
        return make_global_fig(history, layer_sizes, n_res, t_current)

    fig = go.Figure()
    for eid in edge_ids:
        try:
            direction, l_s, src_s, dst_s = eid.split('|')
            l, src, dst = int(l_s), int(src_s), int(dst_s)
        except ValueError:
            continue

        pool  = history['P_up'] if direction == 'up' else history['P_down']
        label = (f"({l},{src})→↑({l+1},{dst})" if direction == 'up'
                 else f"({l},{src})→↓({l-1},{dst})")
        vals  = []
        for idx in range(n_avail):
            mat = pool[idx][l]
            if mat is not None and src < mat.shape[0] and dst < mat.shape[1]:
                vals.append(float(mat[src, dst, :].mean()))
            else:
                vals.append(0.0)
        fig.add_trace(go.Scatter(x=x_vals, y=vals, mode='lines', name=label,
                                 line=dict(width=2)))

    fig.add_vline(x=_step_x(history, t_current), line_dash='dot',
                  line_color='gray', line_width=1.5)
    fig.update_layout(
        title='<b>Évolution des arêtes sélectionnées</b>',
        xaxis_title='Step',
        yaxis=dict(title='Proportion', rangemode='nonnegative'),
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=55, r=20, t=45, b=50),
        legend=dict(orientation='h', y=-0.35, font_size=10),
    )
    return fig


def make_detail_fig(hovered_cd, current_t, history, n_res):
    x_vals  = _steps(history)
    n_steps = len(x_vals)

    def empty():
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text='Survolez un <b>nœud</b>',
                              showarrow=False, font=dict(size=13, color='#888'),
                              x=0.5, y=0.5, xref='paper', yref='paper')],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor='white', paper_bgcolor='#fafafa',
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig

    if hovered_cd is None:
        return empty()
    try:
        l, i, j, typ = (int(hovered_cd[k]) for k in range(4))
    except (IndexError, TypeError, ValueError):
        return empty()

    fig = go.Figure()

    if typ == 0:
        for k in range(n_res):
            vals = [float(history['r'][idx][l][i, k]) for idx in range(n_steps)]
            fig.add_trace(go.Scatter(x=x_vals, y=vals, mode='lines', name=f'r[{k}]',
                                     line=dict(color=_COLORS[k % len(_COLORS)], width=2)))
        utils = [float(history['util'][idx][l][i]) for idx in range(n_steps)]
        fig.add_trace(go.Scatter(x=x_vals, y=utils, mode='lines', name='Utilité',
                                 line=dict(color='black', width=2.5, dash='dash'),
                                 yaxis='y2'))
        fig.add_vline(x=_step_x(history, current_t), line_dash='dot',
                      line_color='gray', line_width=1.5)
        fig.update_layout(
            title=f'<b>Nœud ({l},{i})</b> — Ressources & Utilité',
            xaxis_title='Step',
            yaxis=dict(title='Ressources', side='left'),
            yaxis2=dict(title='Utilité', overlaying='y', side='right', showgrid=False),
            uirevision=f'node-{l}-{i}',
        )
    else:
        for k in range(n_res):
            vals = [float(history['P_up'][idx][l][i, j, k])
                    if history['P_up'][idx][l] is not None else 0.0
                    for idx in range(n_steps)]
            fig.add_trace(go.Scatter(x=x_vals, y=vals, mode='lines', name=f'res {k}',
                                     line=dict(color=_COLORS[k % len(_COLORS)], width=2)))
        fig.add_vline(x=_step_x(history, current_t), line_dash='dot',
                      line_color='gray', line_width=1.5)
        fig.update_layout(
            title=f'<b>Arête ({l},{i}) → ({l+1},{j})</b>',
            xaxis_title='Step',
            yaxis=dict(title='Proportion', rangemode='nonnegative'),
            uirevision=f'edge-{l}-{i}-{j}',
        )

    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='#fafafa',
        margin=dict(l=55, r=65, t=55, b=55),
        legend=dict(orientation='h', y=-0.3, font_size=11),
    )
    return fig


# ─── Utilitaires runs ─────────────────────────────────────────────────────────

def _scan_runs():
    """
    Scanne runs/ et retourne une liste de dicts de métadonnées.
    Lit les .json légers; s'il n'y en a pas, charge juste le nom du fichier.
    """
    entries = []

    # Runs complétés (fichiers .json)
    for jpath in sorted(glob.glob('runs/*.json'), reverse=True):
        try:
            with open(jpath) as f:
                meta = json.load(f)
            pkl_path = jpath.replace('.json', '.pkl')
            if not os.path.exists(pkl_path):
                continue
            entries.append({
                'run_id':     meta.get('run_id', os.path.basename(jpath)),
                'timestamp':  meta.get('timestamp', ''),
                'n_steps':    meta.get('n_steps', '?'),
                'n_res':      meta.get('n_res', meta.get('config', {}).get('n_res', 5)),
                'layer_sizes': meta.get('layer_sizes', []),
                'mode':       meta.get('config', {}).get('MODE', '?'),
                'config':     meta.get('config', {}),
                'pkl_path':   pkl_path,
                'completed':  True,
            })
        except Exception:
            continue

    return entries


def _load_run(pkl_path):
    """Charge un run depuis son fichier pkl (gzip ou brut)."""
    try:
        with gzip.open(pkl_path, 'rb') as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)


# ─── Application Dash ─────────────────────────────────────────────────────────

app = dash.Dash(__name__, title='Economic DAG Viz')
app.config.suppress_callback_exceptions = True

_FF       = 'sans-serif'
_BTN      = {'border': '1px solid #bbb', 'borderRadius': '4px',
             'backgroundColor': 'white', 'cursor': 'pointer', 'fontFamily': _FF}
_BTN_LG   = {**_BTN, 'padding': '6px 14px', 'fontSize': '12px'}
_BTN_SM   = {**_BTN, 'padding': '1px 6px',  'fontSize': '10px'}
_BTN_PRI  = {**_BTN_LG, 'backgroundColor': '#cce5ff', 'borderColor': '#004085'}
_PANEL    = {'fontFamily': _FF, 'fontSize': '11px', 'padding': '6px 10px',
             'borderTop': '1px solid #e0e0e0', 'overflowY': 'auto',
             'maxHeight': '22vh', 'backgroundColor': '#fafafa'}
_LOADING_INLINE = {
    'display': 'inline-flex',
    'alignItems': 'center',
    'minHeight': '28px',
    'overflow': 'visible',
}

_C = {
    'ib': _DIR['in_below']['rgba'].format(a=1),
    'ob': _DIR['out_down']['rgba'].format(a=1),
    'ia': _DIR['in_above']['rgba'].format(a=1),
    'oa': _DIR['out_up']['rgba'].format(a=1),
}

LIVE_TICK_MS = 1000
PLAY_TICK_MS = 50


def _grp_header(label, color, btn_id):
    return html.Div([
        html.Span(label, style={'color': color, 'fontWeight': 'bold'}),
        html.Button('Tout', id=btn_id, n_clicks=0,
                    style={**_BTN_SM, 'marginLeft': '4px'}),
    ], style={'marginBottom': '2px', 'display': 'flex', 'alignItems': 'center',
              'gap': '3px'})


# ── Tab 1 : simulation live ────────────────────────────────────────────────────
_tab1_content = html.Div([
    # ── Sélecteur de run ──────────────────────────────────────────────────────
    html.Div([
        html.Span('Run :', style={'fontFamily': _FF, 'fontSize': '12px',
                                  'fontWeight': '600', 'marginRight': '8px',
                                  'whiteSpace': 'nowrap'}),
        dcc.Dropdown(id='explorer-dd', options=[], value=None,
                     clearable=False, placeholder='Sélectionner un run…',
                     style={'flex': 1, 'fontSize': '11px', 'minWidth': '300px'}),
        dcc.Loading(
            html.Span(id='explorer-status', children='live',
                      style={'fontFamily': _FF, 'fontSize': '11px', 'color': '#888',
                             'whiteSpace': 'nowrap'}),
            type='circle', color='#007bff',
            style={**_LOADING_INLINE, 'marginLeft': '10px'},
        ),
        dcc.Checklist(
            id='util-agg-toggle',
            options=[{'label': ' Utilité agrégée (+ enfants)', 'value': 'agg'}],
            value=[],
            style={'fontFamily': _FF, 'fontSize': '12px', 'whiteSpace': 'nowrap',
                   'marginLeft': '18px'},
            inputStyle={'marginRight': '4px'},
        ),
    ], style={'display': 'flex', 'alignItems': 'center',
              'padding': '6px 16px 4px', 'gap': '4px'}),

    html.Div(id='status-bar',
             style={'textAlign': 'center', 'fontFamily': _FF,
                    'fontSize': '12px', 'color': '#888', 'margin': '4px 0 6px'}),

    html.Div([
        html.Div(
            dcc.Loading(
                dcc.Graph(id='network-graph', style={'height': '60vh'},
                          config={'displayModeBar': False}, clear_on_unhover=True),
                type='circle', color='#007bff',
                style={'height': '60vh'},
            ),
            style={'flex': '3', 'minWidth': 0},
        ),
        html.Div([
            dcc.Graph(id='detail-graph', style={'flex': '1', 'minHeight': 0},
                      config={'displayModeBar': False}),
            html.Div(id='edge-panel', style={**_PANEL, 'display': 'none'}, children=[
                html.Div([
                    html.Span(id='panel-title',
                              style={'fontWeight': 'bold', 'fontSize': '12px'}),
                    html.Button('✕', id='clear-sel-btn', n_clicks=0,
                                style={**_BTN_SM, 'marginLeft': '8px', 'color': '#888'}),
                ], style={'marginBottom': '5px', 'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    html.Div([
                        _grp_header('◀ Entrant ↓', _C['ib'], 'btn-all-ib'),
                        dcc.Checklist(id='cl-in-below', options=[], value=[],
                                      labelStyle={'display': 'inline-block', 'marginRight': '4px',
                                                  'color': _C['ib']},
                                      inputStyle={'marginRight': '2px'}),
                    ], style={'flex': 1, 'minWidth': 0}),
                    html.Div([
                        _grp_header('▶ Sortant ↓', _C['ob'], 'btn-all-ob'),
                        dcc.Checklist(id='cl-out-down', options=[], value=[],
                                      labelStyle={'display': 'inline-block', 'marginRight': '4px',
                                                  'color': _C['ob']},
                                      inputStyle={'marginRight': '2px'}),
                    ], style={'flex': 1, 'minWidth': 0}),
                    html.Div([
                        _grp_header('◀ Entrant ↑', _C['ia'], 'btn-all-ia'),
                        dcc.Checklist(id='cl-in-above', options=[], value=[],
                                      labelStyle={'display': 'inline-block', 'marginRight': '4px',
                                                  'color': _C['ia']},
                                      inputStyle={'marginRight': '2px'}),
                    ], style={'flex': 1, 'minWidth': 0}),
                    html.Div([
                        _grp_header('▶ Sortant ↑', _C['oa'], 'btn-all-oa'),
                        dcc.Checklist(id='cl-out-up', options=[], value=[],
                                      labelStyle={'display': 'inline-block', 'marginRight': '4px',
                                                  'color': _C['oa']},
                                      inputStyle={'marginRight': '2px'}),
                    ], style={'flex': 1, 'minWidth': 0}),
                ], style={'display': 'flex', 'gap': '4px'}),
            ]),
        ], style={'flex': '2', 'minWidth': 0, 'display': 'flex',
                  'flexDirection': 'column', 'height': '60vh'}),
    ], style={'display': 'flex', 'gap': '8px', 'padding': '0 16px'}),

    html.Div([
        html.Div([
            html.Button('Play',   id='play-btn', n_clicks=0, style=_BTN_LG),
            html.Span('Durée replay (s)', style={'fontFamily': _FF, 'fontSize': '12px',
                      'color': '#555', 'alignSelf': 'center'}),
            dcc.Input(id='play-duration', type='number', min=1, step=1, value=20,
                      debounce=True, style={'width': '80px', 'padding': '6px 8px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '10px',
                  'marginBottom': '8px'}),
        dcc.Slider(id='time-slider', min=0, max=0, step=1, value=0,
                   marks={0: '0'},
                   tooltip={'placement': 'top', 'always_visible': False}),
    ], style={'padding': '8px 36px 20px'}),
])

# ── Tab 2 : navigateur de runs + métriques ────────────────────────────────────
_tab2_content = html.Div([
    html.Div([

        # ── Panneau gauche : liste des runs ───────────────────────────────────
        html.Div([
            html.Div([
                html.Span('Runs disponibles',
                          style={'fontFamily': _FF, 'fontWeight': '600',
                                 'fontSize': '14px'}),
                html.Button('↺', id='runs-refresh-btn', n_clicks=0,
                            title='Actualiser',
                            style={**_BTN_SM, 'marginLeft': '6px', 'fontSize': '14px'}),
            ], style={'display': 'flex', 'alignItems': 'center',
                      'marginBottom': '8px'}),

            html.Div(
                dcc.Checklist(
                    id='run-checklist',
                    options=[],
                    value=[],
                    labelStyle={
                        'display': 'block',
                        'padding': '5px 4px',
                        'borderBottom': '1px solid #f0f0f0',
                        'cursor': 'pointer',
                        'fontSize': '11px',
                        'fontFamily': _FF,
                        'lineHeight': '1.4',
                    },
                    inputStyle={'marginRight': '8px', 'cursor': 'pointer',
                                'accentColor': '#007bff'},
                ),
                style={'overflowY': 'auto', 'maxHeight': '65vh',
                       'border': '1px solid #ddd', 'borderRadius': '4px',
                       'padding': '6px', 'backgroundColor': 'white'},
            ),

            html.Div([
                html.Button('Tout', id='run-sel-all', n_clicks=0,
                            style={**_BTN_SM, 'marginRight': '4px'}),
                html.Button('Aucun', id='run-sel-none', n_clicks=0,
                            style=_BTN_SM),
            ], style={'marginTop': '6px'}),

            dcc.Store(id='run-selection', data=[]),
            dcc.Interval(id='runs-scan-interval', interval=5000, n_intervals=0),
        ], style={'width': '28%', 'minWidth': '220px', 'paddingRight': '16px',
                  'fontFamily': _FF, 'fontSize': '12px'}),

        # ── Panneau droit : métriques ─────────────────────────────────────────
        html.Div([
            # Contrôles
            html.Div([
                html.Div([
                    html.Span('Métrique', style={'fontFamily': _FF, 'fontSize': '12px',
                                                 'marginRight': '6px', 'fontWeight': '600'}),
                    dcc.Dropdown(
                        id='metric-select',
                        options=[
                            {'label': 'Distribution des richesses + ECDF', 'value': 'wealth_dist'},
                            {'label': 'Coefficient de Gini',               'value': 'gini'},
                            {'label': 'Mobilité sociale',                  'value': 'social_mobility'},
                            {'label': 'Exclusivité des partenariats',      'value': 'exclusivity'},
                            {'label': 'Oligarchie (part + persistance)',   'value': 'oligarchy'},
                            {'label': 'Convergence monnaie de réserve',   'value': 'currency_convergence'},
                            {'label': 'Utility drift',                    'value': 'utility_drift'},
                        ],
                        value='wealth_dist',
                        clearable=False,
                        style={'width': '280px', 'fontSize': '12px'},
                    ),
                ], style={'display': 'flex', 'alignItems': 'center'}),

                html.Div([
                    html.Div([
                        html.Span('Période', style={'fontFamily': _FF, 'fontSize': '12px',
                                                    'marginRight': '6px', 'fontWeight': '600'}),
                        dcc.RadioItems(
                            id='metric-scope',
                            options=[
                                {'label': ' Dernier état',      'value': 'last'},
                                {'label': ' Tout l\'historique', 'value': 'all'},
                            ],
                            value='last',
                            inline=True,
                            style={'fontFamily': _FF, 'fontSize': '12px'},
                            inputStyle={'marginRight': '3px'},
                            labelStyle={'marginRight': '12px'},
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginRight': '16px'}),

                    dcc.Checklist(
                        id='aggregate-toggle',
                        options=[{'label': ' Agréger les samples', 'value': 'agg'}],
                        value=['agg'],
                        style={'fontFamily': _FF, 'fontSize': '12px'},
                        inputStyle={'marginRight': '4px'},
                    ),

                    html.Div(
                        dcc.Checklist(
                            id='util-agg-toggle-2',
                            options=[{'label': ' Ressources/utilité agrégées (+ enfants)', 'value': 'agg'}],
                            value=[],
                            style={'fontFamily': _FF, 'fontSize': '12px'},
                            inputStyle={'marginRight': '4px'},
                        ),
                        id='util-agg-toggle-2-wrap',
                        title='Applicable aux métriques basées sur les ressources et à "Utility drift"',
                    ),

                    html.Button('Calculer', id='compute-btn', n_clicks=0,
                                style=_BTN_PRI),
                ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap',
                          'gap': '8px', 'marginTop': '8px'}),
            ], style={'display': 'flex', 'flexDirection': 'column',
                      'alignItems': 'flex-start', 'padding': '10px 0 8px',
                      'borderBottom': '1px solid #e0e0e0', 'marginBottom': '10px'}),

            dcc.Loading(
                html.Div(id='metric-status',
                         style={'fontFamily': _FF, 'fontSize': '11px',
                                'color': '#888', 'marginBottom': '6px',
                                'minHeight': '18px'}),
                type='circle',
                color='#007bff',
                style={**_LOADING_INLINE, 'minHeight': '30px', 'marginBottom': '2px'},
            ),

            dcc.Graph(id='metric-graph',
                      style={'height': '65vh'},
                      config={'displayModeBar': True}),
        ], style={'flex': 1, 'minWidth': 0}),

    ], style={'display': 'flex', 'padding': '12px 16px', 'minHeight': '80vh'}),
])

# ── Layout principal ───────────────────────────────────────────────────────────
app.layout = html.Div([
    html.H2('Economic DAG Viz',
            style={'textAlign': 'center', 'fontFamily': _FF,
                   'fontWeight': '600', 'margin': '12px 0 4px'}),

    dcc.Tabs(id='main-tabs', value='tab-live', children=[
        dcc.Tab(label='Exploration d’un run', value='tab-live',
                children=_tab1_content,
                style={'fontFamily': _FF},
                selected_style={'fontFamily': _FF, 'fontWeight': '600'}),
        dcc.Tab(label='Analyse des runs', value='tab-metrics',
                children=_tab2_content,
                style={'fontFamily': _FF},
                selected_style={'fontFamily': _FF, 'fontWeight': '600'}),
    ]),

    # Stores et intervals globaux
    dcc.Interval(id='live-interval', interval=LIVE_TICK_MS, n_intervals=0, disabled=True),
    dcc.Interval(id='play-interval', interval=PLAY_TICK_MS, n_intervals=0, disabled=True),
    dcc.Store(id='ui-state',       data={'playing': False, 'start_step': 0, 'start_tick': 0}),
    dcc.Store(id='selected-node',  data=None),
    dcc.Store(id='selected-edges', data=[]),
    dcc.Store(id='explorer-mode',  data=None),

], style={'backgroundColor': '#fafafa', 'minHeight': '100vh'})


# ─── Helpers explorer ─────────────────────────────────────────────────────────

_BASELINE_CFG = {
    'n_res':                5,
    'MODE':                 'reciprocity',
    'TRANSFORM_MODE':       'mass_action',
    'ATOMS_ONLY_TRANSFORM': True,
    'BUDGET_MODE':          'joint',
}
_BASELINE_LAYERS = [10] * 10

_CFG_KEYS = list(_BASELINE_CFG.keys())

_SHORT = {
    'n_res':                lambda v: f'n_res={v}',
    'MODE':                 lambda v: f'MODE={v}',
    'TRANSFORM_MODE':       lambda v: f'TRANSFORM={v}',
    'ATOMS_ONLY_TRANSFORM': lambda v: f'atoms_only={v}',
    'BUDGET_MODE':          lambda v: f'budget={v}',
}


def _layers_str(ls):
    if not ls:
        return '?'
    if len(set(ls)) == 1:
        return f'[{ls[0]}]×{len(ls)}'
    return '[' + ','.join(str(x) for x in ls) + ']'


def _run_base_label(cfg, layer_sizes):
    """Label d'expérience sans numéro de seed (pour grouper les samples)."""
    diffs = [_SHORT[k](cfg[k]) for k in _CFG_KEYS
             if cfg.get(k) != _BASELINE_CFG[k]]
    variant = '  +  '.join(diffs) if diffs else 'baseline'
    return f'{variant}  |  {_layers_str(layer_sizes)}'


def _run_label(cfg, layer_sizes):
    base   = _run_base_label(cfg, layer_sizes)
    seed   = cfg.get('seed')
    suffix = f'  #s{seed}' if seed is not None else ''
    return f'{base}{suffix}'


def _avg_histories(histories):
    """Moyenne élément par élément de N historiques de même longueur."""
    def _mean_optional(values):
        arrays = [np.asarray(v) for v in values if v is not None]
        if not arrays:
            return None
        return np.mean(arrays, axis=0)

    T = len(histories[0]['r'])
    keys = [k for k in ('r', 'u', 'util', 'P_up', 'P_down') if histories[0].get(k)]
    avg = {k: [] for k in keys}
    for t in range(T):
        for key in keys:
            n_layers = len(histories[0][key][t])
            frame = [
                _mean_optional([h[key][t][l] for h in histories])
                for l in range(n_layers)
            ]
            avg[key].append(frame)
    avg['steps'] = histories[0]['steps']
    return avg


def _explorer_options():
    opts = []
    for jpath in sorted(glob.glob('runs/*.json'), reverse=True):
        try:
            with open(jpath) as f:
                meta = json.load(f)
            pkl = jpath.replace('.json', '.pkl')
            if not os.path.exists(pkl):
                continue
            cfg = meta.get('config', {})
            ls  = meta.get('layer_sizes', [])
            label = _run_label(cfg, ls)
            opts.append({'label': label, 'value': pkl})
        except Exception:
            pass
    return opts


# ─── Callbacks Explorer (Tab 1) ───────────────────────────────────────────────

@app.callback(
    Output('explorer-dd', 'options'),
    Output('explorer-dd', 'value'),
    Input('live-interval', 'n_intervals'),
    State('explorer-dd', 'value'),
)
def refresh_explorer_options(_, current_value):
    options = _explorer_options()
    valid_values = {opt['value'] for opt in options}
    if current_value in valid_values:
        return options, current_value
    default_value = options[0]['value'] if options else None
    return options, default_value


@app.callback(
    Output('explorer-mode',   'data'),
    Output('explorer-status', 'children'),
    Input('explorer-dd', 'value'),
)
def load_explorer_run(pkl_path):
    if not pkl_path:
        return None, 'Sélectionner un run'
    try:
        data = _load_run(pkl_path)
        with _snap_lock:
            _snapshot['data']  = data
            _snapshot['mtime'] = os.path.getmtime(pkl_path)
            _snapshot['path']  = pkl_path
        n = len(data['history']['r'])
        rid = data.get('run_id', os.path.basename(pkl_path))
        return pkl_path, f'{rid}  ({n} frames)'
    except Exception as exc:
        return None, f'Erreur : {exc}'


# ─── Callbacks Tab 1 ──────────────────────────────────────────────────────────

@app.callback(
    Output('time-slider', 'max'),
    Output('time-slider', 'marks'),
    Output('time-slider', 'value'),
    Output('status-bar', 'children'),
    Input('live-interval', 'n_intervals'),
    Input('play-interval', 'n_intervals'),
    Input('explorer-mode', 'data'),
    State('ui-state', 'data'),
    State('play-duration', 'value'),
    State('time-slider', 'value'),
)
def tick(live_n, play_n, explorer_mode, state, duration_s, current_t):
    state = state or {'playing': False, 'start_step': 0, 'start_tick': 0}
    history = _history()
    n_avail = len(history['r'])
    if n_avail == 0:
        return 0, {0: '0'}, 0, 'Aucun run sélectionné'
    max_t = n_avail - 1
    marks = {i: {'label': str(i), 'style': {'fontSize': '10px'}}
             for i in range(0, n_avail, max(1, n_avail // 10))}
    d = _get_data()
    n_total     = d['config'].get('n_steps', '?') if d else '?'
    n_completed = (d.get('metadata', {}).get('n_steps_completed', 0)
                   if d else 0)
    done = (n_completed >= n_total) if isinstance(n_total, int) else False
    rid  = _run_id()
    if done:
        status = f'Run {rid} — terminé ({n_completed} / {n_total} steps, {n_avail} frames)'
    elif isinstance(n_total, int) and n_total > 0:
        pct    = min(100, round(100 * n_completed / n_total))
        status = html.Div([
            html.Span(f'Run {rid} — en cours… {n_completed} / {n_total} steps ({pct}%)',
                      style={'fontFamily': _FF, 'fontSize': '12px', 'color': '#555'}),
            html.Div(
                html.Div(style={
                    'width': f'{pct}%', 'height': '100%',
                    'backgroundColor': '#007bff',
                    'borderRadius': '3px',
                    'transition': 'width 0.4s ease',
                }),
                style={
                    'width': '300px', 'height': '8px',
                    'backgroundColor': '#e0e0e0',
                    'borderRadius': '3px',
                    'marginTop': '4px',
                },
            ),
        ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
    else:
        status = f'Run {rid} — en cours…'
    if ctx.triggered_id == 'explorer-mode':
        return max_t, marks, min(int(current_t or 0), max_t), status
    if state.get('playing'):
        dur = max(1.0, float(duration_s or 20))
        elapsed = max(0, int(play_n) - int(state.get('start_tick', 0)))
        progressed = elapsed * PLAY_TICK_MS * max(1, max_t) / (dur * 1000.0)
        target = min(max_t, int(state.get('start_step', 0) + progressed))
        if target >= max_t:
            return max_t, marks, target if target != current_t else no_update, status
        return max_t, marks, target if target != current_t else no_update, status
    return max_t, marks, no_update, status


@app.callback(
    Output('ui-state', 'data'),
    Output('play-interval', 'disabled'),
    Output('play-btn', 'children'),
    Input('play-btn', 'n_clicks'),
    Input('play-interval', 'n_intervals'),
    Input('time-slider', 'drag_value'),
    Input('explorer-mode', 'data'),
    State('ui-state', 'data'),
    State('time-slider', 'value'),
    prevent_initial_call=True,
)
def control_buttons(_play_clicks, play_n, _dv, _explorer_mode, state, current_t):
    state = state or {'playing': False, 'start_step': 0, 'start_tick': 0}
    trigger = ctx.triggered_id
    n_avail = len(_history()['r'])
    max_t = max(0, n_avail - 1)
    current_t = min(int(current_t or 0), max_t)

    if trigger == 'play-btn':
        if state.get('playing'):
            return {**state, 'playing': False}, True, 'Play'
        start = 0 if n_avail <= 1 else current_t
        return ({**state, 'playing': True,
                 'start_step': int(start), 'start_tick': int(play_n)},
                False, 'Pause')
    if trigger == 'play-interval' and state.get('playing') and current_t >= max_t:
        return {**state, 'playing': False}, True, 'Play'
    if trigger == 'time-slider':
        return {**state, 'playing': False}, True, 'Play'
    if trigger == 'explorer-mode':
        return {**state, 'playing': False}, True, 'Play'
    return no_update, no_update, no_update


@app.callback(
    Output('selected-node', 'data'),
    Input('network-graph', 'clickData'),
    Input('clear-sel-btn', 'n_clicks'),
    Input('explorer-mode', 'data'),
    State('selected-node', 'data'),
    prevent_initial_call=True,
)
def on_click(click_data, _clear, _explorer_mode, current_sel):
    if ctx.triggered_id == 'explorer-mode':
        return None
    if ctx.triggered_id == 'clear-sel-btn':
        return None
    if not click_data:
        return None
    pts = click_data.get('points', [])
    if not pts:
        return None
    cd = pts[0].get('customdata')
    if cd is None or int(cd[3]) != 0:
        return None
    new = [int(cd[0]), int(cd[1])]
    if current_sel and new == list(current_sel):
        return None
    return new


@app.callback(
    Output('panel-title',  'children'),
    Output('edge-panel',   'style'),
    Output('cl-in-below',  'options'), Output('cl-in-below',  'value'),
    Output('cl-out-down',  'options'), Output('cl-out-down',  'value'),
    Output('cl-in-above',  'options'), Output('cl-in-above',  'value'),
    Output('cl-out-up',    'options'), Output('cl-out-up',    'value'),
    Output('btn-all-ib',   'n_clicks'),
    Output('btn-all-ob',   'n_clicks'),
    Output('btn-all-ia',   'n_clicks'),
    Output('btn-all-oa',   'n_clicks'),
    Input('selected-node', 'data'),
)
def update_panel(sel):
    empty_opts = []
    hidden = {**_PANEL, 'display': 'none'}
    if sel is None:
        return ('', hidden,
                empty_opts, [], empty_opts, [], empty_opts, [], empty_opts, [],
                0, 0, 0, 0)
    l_s, i_s    = int(sel[0]), int(sel[1])
    layer_sizes = _layer_sizes()
    n_layers    = len(layer_sizes)
    N_up   = layer_sizes[l_s + 1] if l_s < n_layers - 1 else 0
    N_down = layer_sizes[l_s - 1] if l_s > 0 else 0

    def opts(pairs):
        return [{'label': f' N{j}', 'value': v} for j, v in pairs]

    ib = opts([(k, f'up|{l_s-1}|{k}|{i_s}')   for k in range(N_down)])
    ob = opts([(j, f'down|{l_s}|{i_s}|{j}')   for j in range(N_down)])
    ia = opts([(k, f'down|{l_s+1}|{k}|{i_s}') for k in range(N_up)])
    oa = opts([(j, f'up|{l_s}|{i_s}|{j}')     for j in range(N_up)])

    return (f'Nœud ({l_s},{i_s})', {**_PANEL, 'display': 'block'},
            ib, [], ob, [], ia, [], oa, [],
            0, 0, 0, 0)


for btn_id, cl_id in [('btn-all-ib', 'cl-in-below'), ('btn-all-ob', 'cl-out-down'),
                       ('btn-all-ia', 'cl-in-above'), ('btn-all-oa', 'cl-out-up')]:
    @app.callback(
        Output(cl_id, 'value', allow_duplicate=True),
        Input(btn_id, 'n_clicks'),
        State(cl_id, 'options'),
        State(cl_id, 'value'),
        prevent_initial_call=True,
    )
    def _toggle_all(n, opts, current, _bid=btn_id):
        if not opts:
            return []
        all_vals = [o['value'] for o in opts]
        return [] if set(current) == set(all_vals) else all_vals


@app.callback(
    Output('selected-edges', 'data'),
    Input('cl-in-below',  'value'),
    Input('cl-out-down',  'value'),
    Input('cl-in-above',  'value'),
    Input('cl-out-up',    'value'),
    Input('selected-node', 'data'),
    prevent_initial_call=True,
)
def aggregate_edges(ib, ob, ia, oa, sel_node):
    if ctx.triggered_id == 'selected-node':
        return []
    return (ib or []) + (ob or []) + (ia or []) + (oa or [])


@app.callback(
    Output('network-graph', 'figure'),
    Input('time-slider', 'value'),
    Input('selected-node', 'data'),
    Input('explorer-mode', 'data'),
    Input('util-agg-toggle', 'value'),
)
def update_network(t, sel_node, _explorer_mode, util_agg):
    history     = _history()
    layer_sizes = _layer_sizes()
    n_res       = _n_res()
    n_avail     = len(history['r'])
    if n_avail == 0 or not layer_sizes:
        return go.Figure()
    pos = _pos()
    d = _get_data()
    strict_cd = bool(d and d.get('config', {}).get('STRICT_COBB_DOUGLAS', False))
    eff = _get_effective_history(history, bool(util_agg), strict_cd)
    return make_network_fig(min(t or 0, n_avail - 1),
                            eff, layer_sizes, n_res, pos, sel_node)


@app.callback(
    Output('detail-graph', 'figure'),
    Input('time-slider', 'value'),
    Input('selected-edges', 'data'),
    Input('selected-node', 'data'),
    Input('network-graph', 'clickData'),
    Input('explorer-mode', 'data'),
    Input('util-agg-toggle', 'value'),
)
def update_detail(t, sel_edges, sel_node, click_data, _explorer_mode, util_agg):
    history     = _history()
    layer_sizes = _layer_sizes()
    n_res       = _n_res()
    n_avail     = len(history['r'])
    if n_avail == 0:
        return go.Figure()
    t = min(t or 0, n_avail - 1)
    d = _get_data()
    strict_cd = bool(d and d.get('config', {}).get('STRICT_COBB_DOUGLAS', False))
    eff = _get_effective_history(history, bool(util_agg), strict_cd)
    if ctx.triggered_id != 'explorer-mode' and sel_edges:
        return make_edge_evolution_fig(sel_edges, t, eff, layer_sizes, n_res)
    if sel_node is not None:
        cd = [sel_node[0], sel_node[1], 0, 0]
        return make_detail_fig(cd, t, eff, n_res)
    if ctx.triggered_id != 'explorer-mode' and click_data:
        pts = click_data.get('points', [])
        cd  = pts[0].get('customdata') if pts else None
        if cd is not None:
            return make_detail_fig(cd, t, eff, n_res)
    return make_global_fig(eff, layer_sizes, n_res, t)


# ─── Callbacks Tab 2 ──────────────────────────────────────────────────────────

_AGGREGATE_EFFECTIVE_METRICS = {
    'wealth_dist',
    'gini',
    'social_mobility',
    'oligarchy',
    'currency_convergence',
    'utility_drift',
}


@app.callback(
    Output('util-agg-toggle-2-wrap', 'style'),
    Input('metric-select', 'value'),
)
def _toggle2_visibility(metric):
    active = metric in _AGGREGATE_EFFECTIVE_METRICS
    return {'opacity': '1' if active else '0.35',
            'pointerEvents': 'auto' if active else 'none'}


@app.callback(
    Output('run-checklist', 'options'),
    Input('runs-scan-interval', 'n_intervals'),
    Input('runs-refresh-btn',   'n_clicks'),
    Input('live-interval',      'n_intervals'),
)
def refresh_run_list(_, __, ___):
    entries = _scan_runs()
    opts = []
    for e in entries:
        cfg_e = {**e.get('config', {})}
        cfg_e.setdefault('n_res', e.get('n_res', 5))
        opts.append({
            'label': _run_label(cfg_e, e['layer_sizes']),
            'value': e['pkl_path'],
        })
    return opts


@app.callback(
    Output('run-selection', 'data'),
    Input('run-checklist', 'value'),
)
def sync_selection(value):
    return value or []


@app.callback(
    Output('run-checklist', 'value'),
    Input('run-sel-all',  'n_clicks'),
    Input('run-sel-none', 'n_clicks'),
    Input('explorer-dd',  'value'),
    State('run-checklist', 'options'),
    prevent_initial_call=True,
)
def sel_all_none(_, __, explorer_pkl_path, options):
    if ctx.triggered_id == 'run-sel-none':
        return []
    if ctx.triggered_id == 'explorer-dd':
        if explorer_pkl_path and os.path.exists(explorer_pkl_path):
            return [explorer_pkl_path]
        return no_update
    return [o['value'] for o in (options or [])]


@app.callback(
    Output('metric-graph',  'figure'),
    Output('metric-status', 'children'),
    Input('compute-btn',         'n_clicks'),
    Input('run-selection',       'data'),
    Input('metric-select',       'value'),
    Input('metric-scope',        'value'),
    Input('aggregate-toggle',    'value'),
    Input('util-agg-toggle-2',   'value'),
    prevent_initial_call=True,
)
def compute_metrics(_, selected_paths, metric, scope, aggregate_val, util_agg_val):
    if not selected_paths:
        return go.Figure(), 'Aucun run sélectionné.'

    do_aggregate  = bool(aggregate_val)
    do_util_agg   = bool(util_agg_val) and metric in _AGGREGATE_EFFECTIVE_METRICS

    runs_data = []
    errors    = []
    for path in selected_paths:
        try:
            d = _load_run(path)
            cfg = d.get('config', {})
            ls  = d.get('layer_sizes', [])
            strict_cd = bool(cfg.get('STRICT_COBB_DOUGLAS', False))
            hist = _get_effective_history(d['history'], do_util_agg, strict_cd)
            runs_data.append({
                'run_id':     _run_label(cfg, ls) if cfg else d.get('run_id', os.path.basename(path)),
                'base_label': _run_base_label(cfg, ls) if cfg else d.get('run_id', os.path.basename(path)),
                'history':    hist,
                'layer_sizes': ls,
            })
        except Exception as exc:
            errors.append(f'{os.path.basename(path)}: {exc}')

    if do_aggregate and runs_data:
        groups = {}
        for rd in runs_data:
            groups.setdefault(rd['base_label'], []).append(rd)
        runs_data = [
            {
                'run_id':       base,
                'layer_sizes':  members[0]['layer_sizes'],
                'members':      members,
            }
            for base, members in groups.items()
        ]

    if not runs_data:
        return go.Figure(), 'Erreur de chargement : ' + '; '.join(errors)

    status = f'{len(runs_data)} run(s) chargé(s).'
    if do_util_agg:
        status += '  Ressources/utilités agrégées (+ enfants).'
    if errors:
        status += '  Erreurs : ' + '; '.join(errors)

    if metric == 'wealth_dist':
        fig = M.fig_wealth_distribution(runs_data, scope=scope)
    elif metric == 'gini':
        fig = M.fig_gini(runs_data, scope=scope)
    elif metric == 'social_mobility':
        fig = M.fig_social_mobility(runs_data, scope=scope)
    elif metric == 'exclusivity':
        fig = M.fig_exclusivity(runs_data, scope=scope)
    elif metric == 'oligarchy':
        fig = M.fig_oligarchy(runs_data, scope=scope)
    elif metric == 'currency_convergence':
        fig = M.fig_currency_convergence(runs_data, scope=scope)
    elif metric == 'utility_drift':
        fig = M.fig_utility_drift(runs_data, scope=scope)
    else:
        fig = go.Figure()
        status += f'  Métrique inconnue : {metric}'

    return fig, status


# ─── Point d'entrée standalone ────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualisation interactive EconomicHierarchicalDAG.')
    parser.add_argument('snapshot', nargs='?', default=None,
                        help='Chemin vers un .pkl/.pkl.gz à pré-charger (optionnel).')
    parser.add_argument('--open-browser', action='store_true',
                        help='Ouvre automatiquement http://127.0.0.1:8050 dans le navigateur.')
    args = parser.parse_args()

    if args.snapshot:
        _snapshot['path'] = args.snapshot
        print(f"Pré-chargement : {_snapshot['path']}")
        _reload_if_needed()
    else:
        print("En attente d'une sélection de run dans l'interface…")
    if args.open_browser:
        threading.Thread(
            target=lambda: (time.sleep(1.2), webbrowser.open('http://127.0.0.1:8050')),
            daemon=True,
        ).start()
    app.run(debug=False, port=8050, use_reloader=False)
