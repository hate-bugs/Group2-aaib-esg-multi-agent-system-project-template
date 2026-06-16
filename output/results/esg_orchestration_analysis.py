"""
ESG Multi-Agent Orchestration Analysis
=======================================

Purpose
-------
Compare several agent-orchestration experiments for automated ESG report scoring
and answer one business question:

    "Is an agent-based pipeline suitable for ESG report analysis,
     and if so, which orchestration is the better bet?"

The script loads up to five experiment files (each produced by the same harness)
and builds a tidy dataset, then produces a set of business-oriented plots and a
composite "suitability" scorecard.

Input format (per file)
------------------------
{
  "dataset": str, "sample_size": int, "trials": int,
  "patterns": [<orchestration pattern names>],
  "summary":   { <pattern>: { <metric>: value, ... } },
  "comparison":[ { "pattern": ..., <metric>: value, ... } ],
  "results":   { <pattern>: [ { report_id, total_score, confidence,
                                domain_scores{domain:{estimated_score,confidence,label,...}},
                                comparison{actual_total, absolute_error,
                                           percentage_error, direction},
                                metrics{ ... }, metadata{...} }, ... ] }
}

NOTE ON "ORCHESTRATION TYPE"
----------------------------
Each file already contains several orchestration *patterns* internally
(e.g. parallel_concurrent / handoff_hierarchical / review_critique).
So there are two possible comparison axes:
  * 'experiment' -> compare the 5 files (the framing "each file = one type")
  * 'pattern'    -> compare the internal orchestration patterns
  * 'both'       -> compare every experiment x pattern combination
Set COMPARE_BY below. Everything downstream adapts automatically.
"""

import os
import json
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

# =============================================================================
# 1. CONFIG  --  fill in the five file paths here
# =============================================================================
# The uploaded ("control") file. Replace the placeholder 'x' with its real path.
CONTROL_FILE = 'C:/Users/danie/Documents/00_BIT/master/AAIB/project/Group2-aaib-esg-multi-agent-system-project-template/output/esg_experiment_sustainalytics_control_20260602_173153.json'
print(os.getcwd())
# The other four orchestration files. Replace each placeholder with a real path
# (a string path) OR with an already-loaded dict — the loader accepts both.
x = '../esg_experiment_sustainalytics_high_handoff_20260602_224824.json'
y = '../esg_experiment_sustainalytics_high_parallel_20260602_205554.json'
z = '../esg_experiment_sustainalytics_high_review_20260603_003012.json'
a = '../esg_experiment_sustainalytics_no_critique_20260602_191347.json'

# Ordered map of experiment label -> source (path string or loaded dict).
# Rename the labels to whatever is meaningful for your write-up.
EXPERIMENTS = OrderedDict([
    ('control', CONTROL_FILE),
    ('orch_x',  x),
    ('orch_y',  y),
    ('orch_z',  z),
    ('orch_a',  a),
])

# Comparison axis: 'experiment' | 'pattern' | 'both'
COMPARE_BY = 'experiment'

# Where plots are written (also shown interactively if a display is available).
OUTPUT_DIR = 'esg_analysis_output'

# Composite-suitability weighting (business priorities; must be > 0, auto-normalised).
SUITABILITY_WEIGHTS = {
    'validity':        0.35,   # how close to the real Sustainalytics scores
    'trust':           0.25,   # low hallucination + high consistency
    'coverage':        0.15,   # how much of the rubric is actually addressed
    'cost':            0.15,   # latency / token efficiency (cheaper = better)
    'judge_agreement': 0.10,   # judge accuracy + inter-run agreement
}

# Per-report metrics carried into the tidy frame (higher_is_better noted later).
REPORT_METRIC_KEYS = [
    'mae_total', 'accuracy', 'bias_total', 'judge_accuracy',
    'hallucination_unsupported', 'hallucination_partial',
    'consistency_quantitative', 'latency_total', 'latency_critical_path',
    'token_efficiency', 'coverage_weighted', 'coverage_partial',
    'coverage_per_call', 'agreement_pairwise_pearson', 'agreement_fleiss_kappa',
    'conflict_detection_rate', 'resolution_quality', 'deliberation_quality',
    'dominance_ratio',
]

# Direction map: True = higher is better, False = lower is better.
HIGHER_IS_BETTER = {
    'accuracy': True, 'coverage_weighted': True, 'coverage_partial': True,
    'coverage_per_call': True, 'consistency_quantitative': True,
    'judge_accuracy': True, 'agreement_pairwise_pearson': True,
    'agreement_fleiss_kappa': True, 'token_efficiency': True,
    'deliberation_quality': True, 'resolution_quality': True,
    'mae_total': False, 'bias_total': False, 'hallucination_unsupported': False,
    'hallucination_partial': False, 'latency_total': False,
    'latency_critical_path': False, 'percentage_error': False,
    'absolute_error': False,
}


# =============================================================================
# 2. LOADING & PARSING
# =============================================================================
def _load_one(source):
    """Accept a path string or an already-loaded dict and return the dict."""
    if isinstance(source, dict):
        return source
    if isinstance(source, str):
        with open(source, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    raise TypeError(f"Unsupported experiment source type: {type(source)}")


def _safe(d, key, default=np.nan):
    v = d.get(key, default)
    return v if v is not None else default


def load_experiments(experiments):
    """
    Returns three tidy DataFrames:
      reports_df  : one row per (experiment, pattern, report)
      domains_df  : one row per (experiment, pattern, report, domain)
      summary_df  : one row per (experiment, pattern) from the file-level summary
    Missing/unparseable files are skipped with a warning so the script still runs
    when only some paths are filled in.
    """
    report_rows, domain_rows, summary_rows = [], [], []

    for exp_label, source in experiments.items():
        try:
            data = _load_one(source)
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as err:
            print(f"  [skip] '{exp_label}': could not load ({err})")
            continue

        dataset = data.get('dataset')
        patterns = data.get('patterns') or list((data.get('results') or {}).keys())

        # --- file-level summary block ---
        for pat, smap in (data.get('summary') or {}).items():
            row = {'experiment': exp_label, 'pattern': pat, 'dataset': dataset}
            row.update(smap)
            summary_rows.append(row)

        # --- per-report results block ---
        for pat in patterns:
            for rec in (data.get('results') or {}).get(pat, []):
                comp = rec.get('comparison', {}) or {}
                metr = rec.get('metrics', {}) or {}
                meta = rec.get('metadata', {}) or {}

                row = {
                    'experiment':      exp_label,
                    'pattern':         pat,
                    'report_id':       rec.get('report_id'),
                    'total_score':     _safe(rec, 'total_score'),
                    'confidence':      _safe(rec, 'confidence'),
                    'actual_total':    _safe(comp, 'actual_total'),
                    'absolute_error':  _safe(comp, 'absolute_error'),
                    'percentage_error':_safe(comp, 'percentage_error'),
                    'direction':       comp.get('direction'),
                    'conflicts':       meta.get('conflicts'),
                    'resolved':        meta.get('resolved'),
                }
                for mk in REPORT_METRIC_KEYS:
                    row[mk] = _safe(metr, mk)
                report_rows.append(row)

                # --- domain breakdown ---
                for dom, dv in (rec.get('domain_scores') or {}).items():
                    domain_rows.append({
                        'experiment':     exp_label,
                        'pattern':        pat,
                        'report_id':      rec.get('report_id'),
                        'domain':         dom,
                        'estimated_score':_safe(dv, 'estimated_score'),
                        'confidence':     _safe(dv, 'confidence'),
                        'label':          dv.get('label'),
                    })

    reports_df = pd.DataFrame(report_rows)
    domains_df = pd.DataFrame(domain_rows)
    summary_df = pd.DataFrame(summary_rows)
    return reports_df, domains_df, summary_df


def make_group_key(df, compare_by=COMPARE_BY):
    """Create a single 'group' column according to the chosen comparison axis."""
    if compare_by == 'experiment':
        df['group'] = df['experiment']
    elif compare_by == 'pattern':
        df['group'] = df['pattern']
    else:  # 'both'
        df['group'] = df['experiment'].astype(str) + ' | ' + df['pattern'].astype(str)
    return df


# =============================================================================
# 3. HELPERS
# =============================================================================
def _palette(n):
    base = plt.cm.tab10(np.linspace(0, 1, max(n, 3)))
    return base[:n]


def _minmax(series, higher_is_better=True):
    """Scale to 0..1; flip if lower is better. Constant series -> 0.5."""
    s = pd.to_numeric(series, errors='coerce')
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(0.5, index=series.index)
    scaled = (s - lo) / (hi - lo)
    return scaled if higher_is_better else 1 - scaled


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    print(f"  saved -> {path}")


# =============================================================================
# 4. PLOTS
# =============================================================================
def plot_accuracy_validity(reports_df):
    """2x2: accuracy, MAE (with spread), signed bias, error distribution."""
    groups = list(reports_df['group'].unique())
    colors = _palette(len(groups))
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Scoring Validity — how close are agent scores to Sustainalytics?',
                 fontsize=13, fontweight='bold')

    agg = reports_df.groupby('group')

    # (a) accuracy
    acc = agg['accuracy'].mean().reindex(groups)
    ax[0, 0].bar(groups, acc.values, color=colors)
    ax[0, 0].set_title('Mean accuracy (1 - normalised error)')
    ax[0, 0].set_ylim(0, 1)
    for i, v in enumerate(acc.values):
        ax[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)

    # (b) MAE on the total score, with std of per-report absolute error
    mae = agg['absolute_error'].mean().reindex(groups)
    sd = agg['absolute_error'].std().reindex(groups)
    ax[0, 1].bar(groups, mae.values, yerr=sd.values, capsize=4, color=colors)
    ax[0, 1].set_title('Mean absolute error on total score (± std)')
    ax[0, 1].set_ylabel('score points')

    # (c) signed bias (over/under prediction)
    bias = agg.apply(lambda g: (g['total_score'] - g['actual_total']).mean()).reindex(groups)
    ax[1, 0].bar(groups, bias.values,
                 color=['#c0504d' if v > 0 else '#4f81bd' for v in bias.values])
    ax[1, 0].axhline(0, color='k', lw=0.8)
    ax[1, 0].set_title('Signed bias (positive = over-scoring)')
    ax[1, 0].set_ylabel('predicted − actual')

    # (d) absolute error distribution
    data = [reports_df.loc[reports_df['group'] == g, 'absolute_error'].dropna() for g in groups]
    bp = ax[1, 1].boxplot(data, labels=groups, patch_artist=True, showmeans=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax[1, 1].set_title('Per-report absolute error distribution')
    ax[1, 1].set_ylabel('score points')

    for a_ in ax.flat:
        a_.tick_params(axis='x', rotation=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, '01_accuracy_validity.png')
    return fig


def plot_pred_vs_actual(reports_df):
    """Predicted vs actual total score, one panel per group, with y=x line."""
    groups = list(reports_df['group'].unique())
    n = len(groups)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    colors = _palette(n)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.3 * rows), squeeze=False)
    fig.suptitle('Predicted vs actual total score (calibration)',
                 fontsize=13, fontweight='bold')

    lim_lo = np.nanmin([reports_df['total_score'].min(), reports_df['actual_total'].min()])
    lim_hi = np.nanmax([reports_df['total_score'].max(), reports_df['actual_total'].max()])
    pad = 0.05 * (lim_hi - lim_lo + 1e-9)
    lim = (lim_lo - pad, lim_hi + pad)

    for i, g in enumerate(groups):
        ax = axes[i // cols][i % cols]
        sub = reports_df[reports_df['group'] == g]
        ax.scatter(sub['actual_total'], sub['total_score'],
                   color=colors[i], alpha=0.7, edgecolor='white', s=45)
        ax.plot(lim, lim, 'k--', lw=1, label='perfect (y = x)')
        # least-squares fit for the eye
        m = sub[['actual_total', 'total_score']].dropna()
        if len(m) >= 2:
            b1, b0 = np.polyfit(m['actual_total'], m['total_score'], 1)
            xs = np.array(lim)
            ax.plot(xs, b1 * xs + b0, color=colors[i], lw=1.5, alpha=0.8, label='fit')
            r = np.corrcoef(m['actual_total'], m['total_score'])[0, 1]
            ax.text(0.04, 0.92, f'r = {r:.2f}', transform=ax.transAxes, fontsize=9)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(str(g), fontsize=10)
        ax.set_xlabel('actual'); ax.set_ylabel('predicted')
        ax.legend(fontsize=7, loc='lower right')

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, '02_predicted_vs_actual.png')
    return fig


def plot_direction_and_error(reports_df):
    """Over/under prediction mix + percentage-error distribution."""
    groups = list(reports_df['group'].unique())
    colors = _palette(len(groups))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Direction of error & relative error', fontsize=13, fontweight='bold')

    over = [(reports_df[(reports_df['group'] == g)]['direction'] == 'over').mean() for g in groups]
    under = [(reports_df[(reports_df['group'] == g)]['direction'] == 'under').mean() for g in groups]
    ax1.bar(groups, over, label='over-scores', color='#c0504d')
    ax1.bar(groups, under, bottom=over, label='under-scores', color='#4f81bd')
    ax1.set_title('Share of reports over- vs under-scored')
    ax1.set_ylabel('fraction of reports'); ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=20)

    data = [reports_df.loc[reports_df['group'] == g, 'percentage_error'].dropna() for g in groups]
    bp = ax2.boxplot(data, labels=groups, patch_artist=True, showmeans=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax2.set_title('Percentage error per report (%)')
    ax2.set_ylabel('% error')
    ax2.tick_params(axis='x', rotation=20)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, '03_error_direction.png')
    return fig


def plot_trust_reliability(reports_df):
    """Hallucination, consistency, agreement, judge accuracy."""
    groups = list(reports_df['group'].unique())
    colors = _palette(len(groups))
    agg = reports_df.groupby('group')
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Trustworthiness & reliability', fontsize=13, fontweight='bold')

    panels = [
        ('hallucination_unsupported', 'Unsupported-claim rate (lower = better)', False),
        ('consistency_quantitative',  'Run-to-run score consistency', True),
        ('agreement_pairwise_pearson','Inter-run agreement (pairwise r)', True),
        ('judge_accuracy',            'Judge accuracy', True),
    ]
    for k, (metric, title, hib) in enumerate(panels):
        a_ = ax[k // 2][k % 2]
        if metric not in reports_df.columns:
            a_.axis('off'); a_.set_title(f'{title}\n(not available)'); continue
        vals = agg[metric].mean().reindex(groups)
        a_.bar(groups, vals.values, color=colors)
        a_.set_title(title)
        for i, v in enumerate(vals.values):
            if np.isfinite(v):
                a_.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        a_.tick_params(axis='x', rotation=20)
        if not hib:
            a_.set_facecolor('#fbf3f3')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, '04_trust_reliability.png')
    return fig


def plot_cost_efficiency(reports_df):
    """Latency, token efficiency, coverage-per-call."""
    groups = list(reports_df['group'].unique())
    colors = _palette(len(groups))
    agg = reports_df.groupby('group')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle('Cost & efficiency (the price of the answer)',
                 fontsize=13, fontweight='bold')

    lat = agg['latency_total'].mean().reindex(groups)
    crit = agg['latency_critical_path'].mean().reindex(groups)
    xpos = np.arange(len(groups))
    ax[0].bar(xpos - 0.2, lat.values, width=0.4, label='total', color='#4f81bd')
    ax[0].bar(xpos + 0.2, crit.values, width=0.4, label='critical path', color='#9bbb59')
    ax[0].set_xticks(xpos); ax[0].set_xticklabels(groups, rotation=20)
    ax[0].set_title('Latency (s)'); ax[0].legend()

    tok = agg['token_efficiency'].mean().reindex(groups)
    ax[1].bar(groups, tok.values, color=colors)
    ax[1].set_title('Token efficiency (higher = cheaper per unit)')
    ax[1].tick_params(axis='x', rotation=20)

    cpc = agg['coverage_per_call'].mean().reindex(groups)
    ax[2].bar(groups, cpc.values, color=colors)
    ax[2].set_title('Coverage per agent call')
    ax[2].tick_params(axis='x', rotation=20)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, '05_cost_efficiency.png')
    return fig


def plot_coverage(reports_df):
    groups = list(reports_df['group'].unique())
    agg = reports_df.groupby('group')
    fig, ax = plt.subplots(figsize=(9, 5))
    xpos = np.arange(len(groups))
    cw = agg['coverage_weighted'].mean().reindex(groups)
    cp = agg['coverage_partial'].mean().reindex(groups)
    ax.bar(xpos - 0.2, cw.values, width=0.4, label='weighted', color='#8064a2')
    ax.bar(xpos + 0.2, cp.values, width=0.4, label='partial', color='#c0a8d8')
    ax.set_xticks(xpos); ax.set_xticklabels(groups, rotation=20)
    ax.set_title('Rubric coverage', fontweight='bold')
    ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout()
    _save(fig, '06_coverage.png')
    return fig


def plot_quality_cost_tradeoff(reports_df):
    """The key business chart: accuracy vs latency, bubble = hallucination."""
    groups = list(reports_df['group'].unique())
    colors = _palette(len(groups))
    agg = reports_df.groupby('group')
    acc = agg['accuracy'].mean().reindex(groups)
    lat = agg['latency_total'].mean().reindex(groups)
    hall = agg['hallucination_unsupported'].mean().reindex(groups)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    sizes = 300 + 1800 * _minmax(hall, higher_is_better=True)  # bigger = more hallucination
    for i, g in enumerate(groups):
        ax.scatter(lat.iloc[i], acc.iloc[i], s=sizes.iloc[i],
                   color=colors[i], alpha=0.65, edgecolor='black', linewidth=1.2)
        ax.annotate(str(g), (lat.iloc[i], acc.iloc[i]),
                    xytext=(6, 6), textcoords='offset points', fontsize=9)
    ax.set_xlabel('Mean latency (s)  →  more expensive')
    ax.set_ylabel('Mean accuracy  →  better')
    ax.set_title('Quality vs cost trade-off\n(bubble size = unsupported-claim rate; smaller is safer)',
                 fontweight='bold')
    # "good corner" guide
    ax.annotate('ideal: cheap & accurate', xy=(0.02, 0.97), xycoords='axes fraction',
                fontsize=9, color='green')
    fig.tight_layout()
    _save(fig, '07_quality_cost_tradeoff.png')
    return fig


def plot_radar(reports_df):
    """Normalised multi-metric profile (one polygon per group)."""
    groups = list(reports_df['group'].unique())
    colors = _palette(len(groups))
    agg = reports_df.groupby('group').mean(numeric_only=True)

    axes_spec = [
        ('accuracy', True, 'Accuracy'),
        ('coverage_weighted', True, 'Coverage'),
        ('hallucination_unsupported', False, 'Low halluc.'),
        ('consistency_quantitative', True, 'Consistency'),
        ('latency_total', False, 'Speed'),
        ('judge_accuracy', True, 'Judge acc.'),
    ]
    axes_spec = [s for s in axes_spec if s[0] in agg.columns]
    labels = [s[2] for s in axes_spec]
    norm = pd.DataFrame({s[2]: _minmax(agg[s[0]], s[1]) for s in axes_spec},
                        index=agg.index).reindex(groups)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, g in enumerate(groups):
        vals = norm.loc[g].tolist() + [norm.loc[g].tolist()[0]]
        ax.plot(angles, vals, color=colors[i], lw=2, label=str(g))
        ax.fill(angles, vals, color=colors[i], alpha=0.12)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title('Normalised multi-metric profile\n(further out = better on every axis)',
                 fontweight='bold', pad=24)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=8)
    fig.tight_layout()
    _save(fig, '08_radar_profile.png')
    return fig


def plot_domain_heatmaps(domains_df):
    """Mean estimated score and mean confidence per domain x group."""
    if domains_df.empty:
        return None
    domains_df = make_group_key(domains_df)
    score_pivot = domains_df.pivot_table(index='group', columns='domain',
                                         values='estimated_score', aggfunc='mean')
    conf_pivot = domains_df.pivot_table(index='group', columns='domain',
                                        values='confidence', aggfunc='mean')

    fig, axes = plt.subplots(1, 2, figsize=(15, 0.9 * len(score_pivot) + 4))
    for ax, pivot, title, cmap in [
        (axes[0], score_pivot, 'Mean estimated domain score', 'YlGnBu'),
        (axes[1], conf_pivot, 'Mean domain confidence', 'OrRd'),
    ]:
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title, fontweight='bold')
        ax.grid(False)
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                v = pivot.values[r, c]
                if np.isfinite(v):
                    ax.text(c, r, f'{v:.1f}', ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Domain-level behaviour', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, '09_domain_heatmaps.png')
    return fig


def plot_confidence_calibration(reports_df):
    """Per-report stated confidence vs realised accuracy."""
    groups = list(reports_df['group'].unique())
    colors = _palette(len(groups))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for i, g in enumerate(groups):
        sub = reports_df[reports_df['group'] == g]
        ax.scatter(sub['confidence'], sub['accuracy'],
                   color=colors[i], alpha=0.6, label=str(g), edgecolor='white', s=40)
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', lw=1, label='perfect calibration')
    ax.set_xlabel('Stated confidence'); ax.set_ylabel('Realised accuracy')
    ax.set_title('Confidence calibration', fontweight='bold')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, '10_confidence_calibration.png')
    return fig


# =============================================================================
# 5. COMPOSITE SUITABILITY SCORECARD
# =============================================================================
def composite_suitability(reports_df):
    """
    Build a 0..1 'suitability for ESG analysis' index per group from five
    business pillars, then render a scorecard (composite bar + sub-score heatmap).
    Returns the scorecard DataFrame.
    """
    agg = reports_df.groupby('group').mean(numeric_only=True)
    groups = list(agg.index)

    def col(name, hib):
        return _minmax(agg[name], hib) if name in agg.columns else pd.Series(np.nan, index=agg.index)

    pillars = pd.DataFrame(index=agg.index)
    pillars['validity'] = pd.concat([
        col('accuracy', True),
        1 - _minmax(agg.get('absolute_error', pd.Series(np.nan, index=agg.index)), True),
    ], axis=1).mean(axis=1)
    pillars['trust'] = pd.concat([
        col('hallucination_unsupported', False),
        col('consistency_quantitative', True),
    ], axis=1).mean(axis=1)
    pillars['coverage'] = pd.concat([
        col('coverage_weighted', True),
        col('coverage_partial', True),
    ], axis=1).mean(axis=1)
    pillars['cost'] = pd.concat([
        col('latency_total', False),
        col('token_efficiency', True),
    ], axis=1).mean(axis=1)
    pillars['judge_agreement'] = pd.concat([
        col('judge_accuracy', True),
        col('agreement_pairwise_pearson', True),
    ], axis=1).mean(axis=1)

    w = pd.Series(SUITABILITY_WEIGHTS)
    w = w / w.sum()
    pillars = pillars[w.index]
    composite = (pillars * w).sum(axis=1)
    pillars['SUITABILITY'] = composite
    pillars = pillars.reindex(groups)

    # ---- render ----
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 0.8 * len(groups) + 4),
        gridspec_kw={'width_ratios': [1, 1.4]})
    order = composite.sort_values(ascending=True).index
    bars = ax1.barh(range(len(order)), composite.reindex(order).values,
                    color=plt.cm.RdYlGn(composite.reindex(order).values))
    ax1.set_yticks(range(len(order))); ax1.set_yticklabels(order)
    ax1.set_xlim(0, 1)
    ax1.set_title('Composite suitability index', fontweight='bold')
    for i, v in enumerate(composite.reindex(order).values):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    sub = pillars.drop(columns='SUITABILITY').reindex(order)
    im = ax2.imshow(sub.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(sub.shape[1]))
    ax2.set_xticklabels(sub.columns, rotation=20, ha='right')
    ax2.set_yticks(range(sub.shape[0])); ax2.set_yticklabels(order)
    ax2.set_title('Pillar sub-scores (normalised 0–1)', fontweight='bold')
    ax2.grid(False)
    for r in range(sub.shape[0]):
        for c in range(sub.shape[1]):
            v = sub.values[r, c]
            if np.isfinite(v):
                ax2.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig.suptitle('ESG agent-orchestration suitability scorecard',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, '11_suitability_scorecard.png')
    return pillars.sort_values('SUITABILITY', ascending=False)


# =============================================================================
# 6. TEXT SUMMARY (business read-out)
# =============================================================================
def print_business_summary(reports_df, scorecard):
    agg = reports_df.groupby('group').mean(numeric_only=True)
    print('\n' + '=' * 70)
    print('BUSINESS READ-OUT')
    print('=' * 70)
    n_reports = reports_df.groupby('group')['report_id'].nunique()
    print(f"Groups compared ({COMPARE_BY}): {list(agg.index)}")
    print(f"Reports per group: {n_reports.to_dict()}\n")

    print('Headline metrics (mean per group):')
    cols = [c for c in ['accuracy', 'absolute_error', 'coverage_weighted',
                        'hallucination_unsupported', 'consistency_quantitative',
                        'latency_total', 'judge_accuracy'] if c in agg.columns]
    print(agg[cols].round(3).to_string())

    best = scorecard.index[0]
    print(f"\nHighest composite suitability: '{best}' "
          f"({scorecard.loc[best, 'SUITABILITY']:.3f}/1.00)")
    print("Pillar profile of the leader:")
    print(scorecard.drop(columns='SUITABILITY').loc[best].round(3).to_string())

    # crude go / no-go heuristic for the overall question
    overall_acc = agg['accuracy'].max() if 'accuracy' in agg else np.nan
    overall_hall = agg['hallucination_unsupported'].min() if 'hallucination_unsupported' in agg else np.nan
    print('\nWhole-question signal (is agent-based ESG scoring viable?):')
    print(f"  best mean accuracy across groups : {overall_acc:.3f}")
    print(f"  lowest unsupported-claim rate    : {overall_hall:.3f}")
    if np.isfinite(overall_acc) and np.isfinite(overall_hall):
        if overall_acc >= 0.85 and overall_hall <= 0.4:
            verdict = "PROMISING — strong scoring accuracy with controllable hallucination."
        elif overall_acc >= 0.80:
            verdict = "CONDITIONAL — usable accuracy, but hallucination needs mitigation before production."
        else:
            verdict = "NOT YET — accuracy below a defensible threshold for ESG decisions."
        print(f"  heuristic verdict                : {verdict}")
    print('=' * 70 + '\n')


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    print("Loading experiments...")
    reports_df, domains_df, summary_df = load_experiments(EXPERIMENTS)
    if reports_df.empty:
        raise SystemExit("No data loaded — fill in the file paths in the CONFIG section.")

    reports_df = make_group_key(reports_df, COMPARE_BY)
    print(f"Loaded {len(reports_df)} report rows across "
          f"{reports_df['group'].nunique()} group(s).")

    print("\nGenerating plots...")
    plot_accuracy_validity(reports_df)
    plot_pred_vs_actual(reports_df)
    plot_direction_and_error(reports_df)
    plot_trust_reliability(reports_df)
    plot_cost_efficiency(reports_df)
    plot_coverage(reports_df)
    plot_quality_cost_tradeoff(reports_df)
    plot_radar(reports_df)
    plot_domain_heatmaps(domains_df)
    plot_confidence_calibration(reports_df)

    print("\nBuilding suitability scorecard...")
    scorecard = composite_suitability(reports_df)

    print_business_summary(reports_df, scorecard)

    # Export the tidy tables for any further work / write-up.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reports_df.to_csv(os.path.join(OUTPUT_DIR, 'reports_tidy.csv'), index=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_tidy.csv'), index=False)
    scorecard.to_csv(os.path.join(OUTPUT_DIR, 'suitability_scorecard.csv'))
    print(f"Tidy tables and {11} figures written to '{OUTPUT_DIR}/'.")

    # plt.show()  # uncomment to display interactively


if __name__ == '__main__':
    main()
