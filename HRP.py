"""
HRP - Hierarchical Risk Parity aplicado al Merval
Basado en: López de Prado (2016)
Comparación: HRP vs IVP vs Equal Weight
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
import os
warnings.filterwarnings("ignore")

# --------------------------
# Carpeta de resultados
# --------------------------
os.makedirs("results", exist_ok=True)

# --------------------------
# Datos
# --------------------------

tickers = [
    'ALUA.BA','BBAR.BA','BYMA.BA','CEPU.BA','COME.BA',
    'CRES.BA','EDN.BA','GGAL.BA','IRSA.BA','LOMA.BA',
    'MIRG.BA','PAMP.BA','SUPV.BA','TECO2.BA','TGNO4.BA',
    'TGSU2.BA','TRAN.BA','TXAR.BA','VALO.BA','YPFD.BA'
]


data = yf.download(tickers, start="2020-01-01", progress=False)["Close"]
data = data.dropna(axis=1, thresh=int(0.8 * len(data)))
data = data.ffill().dropna()
returns = data.pct_change().dropna()

tickers = list(returns.columns)

print(f"Tickers: {tickers}")
print(f"Observaciones: {len(returns)}\n")


# --------------------------
# HRP
# --------------------------

def correl_dist(corr):
    """Distancia basada en correlacion """
    return ((1 - corr) / 2.) ** 0.5

def get_ivp(cov_matrix):
    """Inverse-Variance Portfolio"""
    ivp = 1. / np.diag(np.array(cov_matrix))
    return ivp / ivp.sum()

def get_cluster_var(cov_matrix, cluster_items):
    """Varianza de un cluster bajo pesos IVP"""
    cov_ = cov_matrix.iloc[cluster_items, cluster_items]
    w_ = get_ivp(cov_)
    return float(np.dot(w_, np.dot(np.array(cov_), w_)))

def get_quasi_diag(link_matrix):
    """quasi diagonalización"""
    link_matrix = link_matrix.astype(int)
    sort_ix = [int(link_matrix[-1, 0]), int(link_matrix[-1, 1])]
    num_items = int(link_matrix[-1, 3])

    while max(sort_ix) >= num_items:
        new_sort_ix = []
        for item in sort_ix:
            if item >= num_items:
                idx = item - num_items
                new_sort_ix += [int(link_matrix[idx, 0]), int(link_matrix[idx, 1])]
            else:
                new_sort_ix.append(item)
        sort_ix = new_sort_ix
    return sort_ix

def get_hrp(cov_matrix, corr_matrix):
    """clustering + quasi-diag + recursive bisection"""
    # Tree clustering
    dist = correl_dist(corr_matrix)
    link_matrix = linkage(dist, method='single')

    # Quasi-diagonalización
    sort_ix = get_quasi_diag(link_matrix)
    sort_labels = corr_matrix.index[sort_ix].tolist()

    # Recursive bisection
    w = pd.Series(1.0, index=sort_labels)
    clusters = [sort_labels]

    while len(clusters) > 0:
        new_clusters = []
        for c in clusters:
            if len(c) > 1:
                half = len(c) // 2
                new_clusters.append(c[:half])
                new_clusters.append(c[half:])
        clusters = new_clusters

        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                break
            c0 = clusters[i]
            c1 = clusters[i + 1]

            idx0 = [cov_matrix.index.get_loc(t) for t in c0]
            idx1 = [cov_matrix.index.get_loc(t) for t in c1]

            var0 = get_cluster_var(cov_matrix, idx0)
            var1 = get_cluster_var(cov_matrix, idx1)

            alpha = 1 - var0 / (var0 + var1)

            w[c0] *= alpha
            w[c1] *= (1 - alpha)

    return w.sort_index()

# --------------------------
# Pesos portfolio
# --------------------------
cov = returns.cov()
corr = returns.corr()

w_hrp = get_hrp(cov, corr)
w_ivp = pd.Series(get_ivp(cov), index=tickers)
w_ew  = pd.Series(1.0 / len(tickers), index=tickers)

print("=" * 50)
print("PESOS DEL PORTFOLIO (Full Sample)")
print("=" * 50)
comparison = pd.DataFrame({
    'HRP (%)': (w_hrp * 100).round(2),
    'IVP (%)': (w_ivp * 100).round(2),
    'EW (%)':  (w_ew  * 100).round(2),
})
print(comparison.sort_values('HRP (%)', ascending=False).to_string())

# --------------------------
# Grafico pesos
# --------------------------
fig_weights = go.Figure()
x = w_hrp.sort_values(ascending=False).index

fig_weights.add_trace(go.Bar(name='HRP', x=x, y=(w_hrp[x]*100).values,
    marker_color='#00b4d8'))
fig_weights.add_trace(go.Bar(name='IVP', x=x, y=(w_ivp[x]*100).values,
    marker_color='#f77f00'))
fig_weights.add_trace(go.Bar(name='Equal Weight', x=x, y=(w_ew[x]*100).values,
    marker_color='#4caf50', opacity=0.7))

fig_weights.update_layout(
    title='Comparación de Pesos: HRP vs IVP vs Equal Weight',
    barmode='group',
    xaxis_title='Ticker',
    yaxis_title='Peso (%)',
    template='plotly_dark',
    height=500
)
fig_weights.write_html("results/hrp_weights.html")
print("\nGráfico de pesos guardado: results/hrp_weights.html")

# --------------------------
# Grafico Heatmaps
# --------------------------
dist_matrix = correl_dist(corr)
link_full = linkage(dist_matrix, method='single')
sort_ix = get_quasi_diag(link_full)
sort_labels = corr.index[sort_ix].tolist()
corr_clustered = corr.loc[sort_labels, sort_labels]

fig_heat = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Correlación Original', 'Correlación Clusterizada (HRP Stage 2)')
)

fig_heat.add_trace(
    go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
               colorscale='RdBu_r', zmin=-1, zmax=1, showscale=False),
    row=1, col=1
)
fig_heat.add_trace(
    go.Heatmap(z=corr_clustered.values, x=corr_clustered.columns, y=corr_clustered.index,
               colorscale='RdBu_r', zmin=-1, zmax=1, colorbar=dict(title='ρ')),
    row=1, col=2
)

fig_heat.update_layout(
    title='Matriz de Correlación: Antes y Después del Clustering HRP',
    template='plotly_dark',
    height=600,
    width=1100
)
fig_heat.write_html("results/hrp_heatmaps.html")
print("Heatmaps guardados: results/hrp_heatmaps.html")

# --------------------------
# Grafico Dendograma
# --------------------------

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig_dend, ax = plt.subplots(figsize=(14, 6), facecolor='#1a1a2e')
ax.set_facecolor('#1a1a2e')

dend = dendrogram(
    link_full,
    labels=corr.index.tolist(),
    ax=ax,
    color_threshold=0.5 * max(link_full[:, 2]),
    above_threshold_color='#8ecae6'
)

ax.set_title('Dendrograma de Clustering – Merval (HRP Stage 1)',
             color='white', fontsize=13, pad=15)
ax.tick_params(colors='white', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#444')
ax.yaxis.label.set_color('white')
ax.set_ylabel('Distancia', color='white')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/hrp_dendrogram.png", dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.close()
print("Dendrograma guardado: results/hrp_dendrogram.png")


# --------------------------
# Backtest
# --------------------------

lookback  = 252   # 1 año de historia
rebal_freq = 21   # rebalanceo mensual

portfolio_returns = {
    'HRP': [],
    'IVP': [],
    'EW' : [],
}

ret_arr = returns.values
ret_idx = returns.index
n_obs   = len(ret_arr)

all_dates = []
for start in range(lookback, n_obs - rebal_freq + 1, rebal_freq):
    window = ret_arr[start - lookback: start]
    cov_w  = pd.DataFrame(np.cov(window.T), index=tickers, columns=tickers)
    corr_w = pd.DataFrame(np.corrcoef(window.T), index=tickers, columns=tickers)

    try:
        w_h = get_hrp(cov_w, corr_w).values
    except Exception:
        w_h = np.ones(len(tickers)) / len(tickers)

    w_i = get_ivp(cov_w)
    w_e = np.ones(len(tickers)) / len(tickers)

    end = min(start + rebal_freq, n_obs)
    for k in range(start, end):
        day_ret = ret_arr[k]
        portfolio_returns['HRP'].append(np.dot(day_ret, w_h))
        portfolio_returns['IVP'].append(np.dot(day_ret, w_i))
        portfolio_returns['EW' ].append(np.dot(day_ret, w_e))
        all_dates.append(ret_idx[k])

backtest_df = pd.DataFrame(portfolio_returns, index=all_dates)

# Equity curves
cum = (1 + backtest_df).cumprod()

# --------------------------
# Metricas
# --------------------------
def sharpe(r, periods=252):
    return r.mean() / r.std() * np.sqrt(periods)

def max_drawdown(cum_series):
    roll_max = cum_series.cummax()
    dd = (cum_series - roll_max) / roll_max
    return dd.min()

def ann_vol(r, periods=252):
    return r.std() * np.sqrt(periods)

def ann_ret(cum_series, periods=252):
    n = len(cum_series)
    return cum_series.iloc[-1] ** (periods / n) - 1

metrics = pd.DataFrame({
    'Retorno Anualizado (%)': {k: ann_ret(cum[k]) * 100 for k in cum.columns},
    'Volatilidad Anualizada (%)': {k: ann_vol(backtest_df[k]) * 100 for k in backtest_df.columns},
    'Sharpe Ratio': {k: sharpe(backtest_df[k]) for k in backtest_df.columns},
    'Max Drawdown (%)': {k: max_drawdown(cum[k]) * 100 for k in cum.columns},
    'Valor Final ($1)': {k: cum[k].iloc[-1] for k in cum.columns},
}).round(3)

print("\n" + "=" * 60)
print("MÉTRICAS OUT-OF-SAMPLE")
print("=" * 60)
print(metrics.to_string())

# --------------------------
# Grafico Equity Curves
# --------------------------
fig_bt = make_subplots(
    rows=2, cols=1,
    row_heights=[0.7, 0.3],
    subplot_titles=('Equity Curve (Out-of-Sample)', 'Volatilidad 21d (Anualizada %)'),
    vertical_spacing=0.12
)

colors = {'HRP': '#00b4d8', 'IVP': '#f77f00', 'EW': '#4caf50'}
for method in ['HRP', 'IVP', 'EW']:
    fig_bt.add_trace(
        go.Scatter(x=cum.index, y=cum[method], name=method,
                   line=dict(color=colors[method], width=2)),
        row=1, col=1
    )

roll_vol = backtest_df.rolling(21).std() * np.sqrt(252) * 100
for method in ['HRP', 'IVP', 'EW']:
    fig_bt.add_trace(
        go.Scatter(x=roll_vol.index, y=roll_vol[method], name=method,
                   line=dict(color=colors[method], width=1.5), showlegend=False),
        row=2, col=1
    )

metrics_text = (
    f"<b>Métricas Out-of-Sample</b><br>"
    f"{'Método':<6} {'Ret%':>7} {'Vol%':>7} {'Sharpe':>8} {'MaxDD%':>9}<br>"
)
for m in ['HRP', 'IVP', 'EW']:
    metrics_text += (
        f"{m:<6} "
        f"{metrics.loc[m,'Retorno Anualizado (%)']:>6.1f}% "
        f"{metrics.loc[m,'Volatilidad Anualizada (%)']:>6.1f}% "
        f"{metrics.loc[m,'Sharpe Ratio']:>7.2f}  "
        f"{metrics.loc[m,'Max Drawdown (%)']:>7.1f}%<br>"
    )

fig_bt.add_annotation(
    text=metrics_text,
    xref="paper", yref="paper",
    x=0.01, y=0.70,
    showarrow=False,
    font=dict(family="monospace", size=11, color="white"),
    align="left",
    bgcolor="rgba(30,30,50,0.85)",
    bordercolor="#555",
    borderwidth=1
)

fig_bt.update_layout(
    title='Backtest HRP vs IVP vs Equal Weight – Merval',
    template='plotly_dark',
    height=750,
    hovermode='x unified',
    legend=dict(orientation='h', y=1.02)
)
fig_bt.update_yaxes(title_text='Valor Acumulado ($1)', row=1, col=1)
fig_bt.update_yaxes(title_text='Vol. Anualizada (%)', row=2, col=1)

fig_bt.write_html("results/hrp_backtest.html")
print("Backtest guardado: results/hrp_backtest.html")