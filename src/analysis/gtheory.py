"""Random-effects Method-of-Moments Generalizability Theory (G-study, D-study, rank correlations)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger(__name__)

DSTUDY_NS = [5, 10, 15, 20, 30, 50, 100]

FULL_FACTORS = [
    ("model", ["model"]),
    ("item", ["qid"]),
    ("regime", ["regime"]),
    ("format", ["format"]),
    ("model_x_item", ["model", "qid"]),
    ("model_x_regime", ["model", "regime"]),
]

REGIME_FACTORS_N1 = [
    ("model", ["model"]),
    ("item", ["qid"]),
    ("format", ["format"]),
    ("model_x_item", ["model", "qid"]),
]

REGIME_FACTORS_SIMPLE = [
    ("model", ["model"]),
    ("item", ["qid"]),
    ("model_x_item", ["model", "qid"]),
]


def load_config(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def estimate_variance_components_mom(
    df: pd.DataFrame,
    factors: list[tuple[str, list[str]]],
    y_col: str = "correct",
) -> dict[str, float]:
    """Henderson's Method 1 (exact unbalanced Method-of-Moments) random-effects variance components.

    Clamps negative variance estimates to 0.0.
    """
    y = df[y_col].to_numpy(dtype=float)
    n_obs = len(y)
    if n_obs == 0:
        return {name: 0.0 for name, _ in factors} | {"residual": 0.0, "total": 0.0}

    # Encode each factor's group labels as integer codes 0..G_k-1
    codes_list: list[np.ndarray] = []
    n_groups_list: list[int] = []
    group_counts_list: list[np.ndarray] = []
    t_vals: list[float] = []

    for _, cols in factors:
        if len(cols) == 1:
            codes, uniques = pd.factorize(df[cols[0]], sort=False)
        else:
            codes, uniques = pd.factorize(pd.MultiIndex.from_frame(df[cols]), sort=False)
        g_count = len(uniques)
        counts = np.bincount(codes, minlength=g_count).astype(float)
        sums = np.bincount(codes, weights=y, minlength=g_count)
        t_k = float(np.sum((sums ** 2) / counts))
        codes_list.append(codes)
        n_groups_list.append(g_count)
        group_counts_list.append(counts)
        t_vals.append(t_k)

    t0 = float((y.sum() ** 2) / n_obs)
    t_tot = float(np.sum(y ** 2))

    k_fac = len(factors)
    dim = k_fac + 1  # factors + residual
    C = np.zeros((dim, dim), dtype=float)
    c0 = np.zeros(dim, dtype=float)

    # Grand-mean expectations c_{0, b}
    for b in range(k_fac):
        c0[b] = float(np.sum(group_counts_list[b] ** 2) / n_obs)
    c0[k_fac] = 1.0  # residual

    # Factor expectations c_{a, b}
    for a in range(k_fac):
        codes_a = codes_list[a]
        counts_a = group_counts_list[a]
        for b in range(k_fac):
            if a == b:
                C[a, b] = float(n_obs)
            else:
                codes_b = codes_list[b]
                g_b = n_groups_list[b]
                # Cross-tabulate (a, b) via linear index
                pair_idx = codes_a * g_b + codes_b
                _, pair_counts = np.unique(pair_idx, return_counts=True)
                # Map each unique pair back to group a
                uniq_pairs = np.unique(pair_idx)
                a_of_pair = uniq_pairs // g_b
                C[a, b] = float(np.sum((pair_counts.astype(float) ** 2) / counts_a[a_of_pair]))
        C[a, k_fac] = float(n_groups_list[a])

    # Total row c_{tot, b} = N
    C[k_fac, :] = float(n_obs)

    M = C - c0[None, :]
    q = np.array([t_k - t0 for t_k in t_vals] + [t_tot - t0], dtype=float)

    try:
        sigma2 = np.linalg.solve(M, q)
    except np.linalg.LinAlgError:
        sigma2, _, _, _ = np.linalg.lstsq(M, q, rcond=None)

    sigma2 = np.maximum(0.0, sigma2)
    out = {factors[i][0]: float(sigma2[i]) for i in range(k_fac)}
    out["residual"] = float(sigma2[k_fac])
    out["total"] = float(np.sum(sigma2))
    return out


def compute_rho(
    vc: dict[str, float],
    n_items: float,
    n_regimes: float = 1.0,
    obs_per_model_item: float = 3.0,
) -> float:
    """Compute relative generalizability coefficient E(rho^2)."""
    var_m = vc.get("model", 0.0)
    var_mi = vc.get("model_x_item", 0.0)
    var_mr = vc.get("model_x_regime", 0.0)
    var_e = vc.get("residual", 0.0)
    denom = var_m + (var_mi / n_items) + (var_mr / n_regimes) + (var_e / (n_items * obs_per_model_item))
    if denom <= 0:
        return 0.0
    return float(var_m / denom)


def bootstrap_rho_ci(
    df: pd.DataFrame,
    factors: list[tuple[str, list[str]]],
    n_items: int,
    n_regimes: float,
    obs_per_model_item: float,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Item-cluster bootstrap 95% CI for generalizability coefficient rho."""
    vc_point = estimate_variance_components_mom(df, factors)
    rho_point = compute_rho(vc_point, n_items, n_regimes, obs_per_model_item)

    qids = sorted(df["qid"].unique())
    qid_groups = {q: grp for q, grp in df.groupby("qid")}
    rng = np.random.default_rng(seed)

    boot_rhos = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        sampled = rng.choice(qids, size=len(qids), replace=True)
        parts = []
        for idx, q in enumerate(sampled):
            part = qid_groups[q].copy()
            part["qid"] = f"b{idx}"
            parts.append(part)
        df_b = pd.concat(parts, ignore_index=True)
        vc_b = estimate_variance_components_mom(df_b, factors)
        boot_rhos[b] = compute_rho(vc_b, n_items, n_regimes, obs_per_model_item)

    lo = float(np.percentile(boot_rhos, 2.5))
    hi = float(np.percentile(boot_rhos, 97.5))
    return rho_point, lo, hi


def compute_rank_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman and Kendall rank correlations across models for key contrasts."""
    pairs = [
        ("N1 vs N2", df[df["regime"] == "N1"], df[df["regime"] == "N2"]),
        ("N2 vs N3", df[df["regime"] == "N2"], df[df["regime"] == "N3"]),
        ("CSV vs MD (N1)", df[(df["regime"] == "N1") & (df["format"] == "csv")], df[(df["regime"] == "N1") & (df["format"] == "markdown")]),
    ]
    rows = []
    for label, d1, d2 in pairs:
        s1 = d1.groupby("model")["correct"].mean()
        s2 = d2.groupby("model")["correct"].mean()
        common = sorted(set(s1.index) & set(s2.index))
        if len(common) < 2:
            continue
        x = s1.loc[common].to_numpy()
        y = s2.loc[common].to_numpy()
        sp_r, sp_p = spearmanr(x, y)
        kt_r, kt_p = kendalltau(x, y)
        rows.append({
            "comparison": label,
            "n_models": len(common),
            "spearman_rho": round(float(sp_r), 4),
            "spearman_p": round(float(sp_p), 4),
            "kendall_tau": round(float(kt_r), 4),
            "kendall_p": round(float(kt_p), 4),
        })
    return pd.DataFrame(rows)


def generate_tab4_gtheory(
    vc_full: dict[str, float],
    rho_rows: list[dict],
    corr_df: pd.DataFrame,
    tables_dir: Path,
):
    """Write Table 4 (Markdown and LaTeX): Variance decomposition, G-coefficients, and rank correlations."""
    comp_labels = [
        ("model", "Model ($m$)"),
        ("item", "Item ($i$)"),
        ("regime", "Regime ($r$)"),
        ("format", "Format ($f$)"),
        ("model_x_item", "Model $\\times$ Item ($m \\times i$)"),
        ("model_x_regime", "Model $\\times$ Regime ($m \\times r$)"),
        ("residual", "Residual ($e$)"),
    ]
    tot = vc_full["total"] if vc_full["total"] > 0 else 1.0

    # Markdown
    md = [
        "### Variance Components (Full Random-Effects MoM G-Study)",
        "",
        "| Component | Variance ($\\hat{\\sigma}^2$) | % of Total |",
        "|---|---:|---:|",
    ]
    for key, label in comp_labels:
        v = vc_full.get(key, 0.0)
        md.append(f"| {label.replace('$', '')} | {v:.5f} | {100.0 * v / tot:.2f}% |")
    md.append(f"| **Total** | **{vc_full['total']:.5f}** | **100.00%** |")
    md.extend([
        "",
        "### Generalizability Coefficient ($E\\rho^2$) with 95% Bootstrap CI",
        "",
        "| Scope | $n_{\\text{items}}$ | $E\\rho^2$ | 95% CI |",
        "|---|---:|---:|---:|",
    ])
    for r in rho_rows:
        md.append(f"| {r['scope']} | {r['n_items']} | {r['rho']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    md.extend([
        "",
        "### Model Rank Correlations Across Regimes and Formats",
        "",
        "| Comparison | Spearman $\\rho_s$ ($p$) | Kendall $\\tau$ ($p$) |",
        "|---|---:|---:|",
    ])
    for _, r in corr_df.iterrows():
        md.append(
            f"| {r['comparison']} | {r['spearman_rho']:.3f} ({r['spearman_p']:.3f}) "
            f"| {r['kendall_tau']:.3f} ({r['kendall_p']:.3f}) |"
        )
    (tables_dir / "tab4_gtheory_variance.md").write_text("\n".join(md) + "\n")

    # LaTeX
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Generalizability Theory (Method-of-Moments) variance decomposition, generalizability coefficients ($E\rho^2$, 95\% item-bootstrap CI), and model rank correlations.}",
        r"\label{tab:gtheory_variance}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Source of Variance & $\hat{\sigma}^2$ & \% of Total \\",
        r"\midrule",
    ]
    for key, label in comp_labels:
        v = vc_full.get(key, 0.0)
        tex.append(f"{label} & {v:.5f} & {100.0 * v / tot:.2f}\\% \\\\")
    tex.append(r"\midrule")
    tex.append(f"\\textbf{{Total}} & \\textbf{{{vc_full['total']:.5f}}} & \\textbf{{100.00\\%}} \\\\")
    tex.extend([
        r"\midrule",
        r"\multicolumn{3}{l}{\textbf{Generalizability Coefficient ($E\rho^2$, $n=36$ items)}} \\",
        r"Scope & $E\rho^2$ & 95\% Bootstrap CI \\",
        r"\midrule",
    ])
    for r in rho_rows:
        tex.append(f"{r['scope']} & {r['rho']:.4f} & [{r['ci_low']:.4f}, {r['ci_high']:.4f}] \\\\")
    tex.extend([
        r"\midrule",
        r"\multicolumn{3}{l}{\textbf{Cross-Condition Model Rank Correlations ($N=7$ models)}} \\",
        r"Comparison & Spearman $\rho_s$ ($p$) & Kendall $\tau$ ($p$) \\",
        r"\midrule",
    ])
    for _, r in corr_df.iterrows():
        tex.append(
            f"{r['comparison']} & {r['spearman_rho']:.3f} ({r['spearman_p']:.3f}) "
            f"& {r['kendall_tau']:.3f} ({r['kendall_p']:.3f}) \\\\"
        )
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    (tables_dir / "tab4_gtheory_variance.tex").write_text("\n".join(tex) + "\n")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n-resamples", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    accuracy_path = Path(cfg["outputs"]["accuracy"])
    out_path = Path(cfg["outputs"]["gtheory"])
    tables_dir = Path(cfg["outputs"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(accuracy_path)
    n_items = int(df["qid"].nunique())
    n_regimes = float(df["regime"].nunique())
    obs_per_mi_all = float(len(df) / (df["model"].nunique() * n_items))

    # 1. Full G-study
    vc_full = estimate_variance_components_mom(df, FULL_FACTORS)
    rho_all, lo_all, hi_all = bootstrap_rho_ci(
        df, FULL_FACTORS, n_items, n_regimes, obs_per_mi_all, n_resamples=args.n_resamples
    )

    vc_rows = [{"scope": "Overall", **vc_full, "rho": round(rho_all, 4), "rho_ci_low": round(lo_all, 4), "rho_ci_high": round(hi_all, 4)}]
    rho_rows = [{"scope": "Overall", "n_items": n_items, "rho": rho_all, "ci_low": lo_all, "ci_high": hi_all}]

    # 2. Per-regime G-study
    dstudy_rows = []
    for n_d in DSTUDY_NS:
        dstudy_rows.append({
            "scope": "Overall",
            "n_items": n_d,
            "rho": round(compute_rho(vc_full, n_d, n_regimes, obs_per_mi_all), 4),
        })

    for reg in ["N1", "N2", "N3"]:
        sub = df[df["regime"] == reg]
        if sub.empty:
            continue
        facs = REGIME_FACTORS_N1 if reg == "N1" else REGIME_FACTORS_SIMPLE
        obs_per_mi = float(len(sub) / (sub["model"].nunique() * sub["qid"].nunique()))
        vc_r = estimate_variance_components_mom(sub, facs)
        rho_r, lo_r, hi_r = bootstrap_rho_ci(
            sub, facs, n_items, 1.0, obs_per_mi, n_resamples=args.n_resamples
        )
        vc_rows.append({"scope": reg, **vc_r, "rho": round(rho_r, 4), "rho_ci_low": round(lo_r, 4), "rho_ci_high": round(hi_r, 4)})
        rho_rows.append({"scope": reg, "n_items": n_items, "rho": rho_r, "ci_low": lo_r, "ci_high": hi_r})
        for n_d in DSTUDY_NS:
            dstudy_rows.append({
                "scope": reg,
                "n_items": n_d,
                "rho": round(compute_rho(vc_r, n_d, 1.0, obs_per_mi), 4),
            })

    df_vc = pd.DataFrame(vc_rows).fillna(0.0)
    df_dstudy = pd.DataFrame(dstudy_rows)
    df_corr = compute_rank_correlations(df)

    df_vc.to_csv(out_path, index=False)
    df_dstudy.to_csv(tables_dir / "gtheory_dstudy.csv", index=False)
    df_corr.to_csv(tables_dir / "gtheory_correlations.csv", index=False)

    generate_tab4_gtheory(vc_full, rho_rows, df_corr, tables_dir)

    logger.info(f"G-study summary:\n{df_vc.to_string(index=False)}")
    logger.info(f"Rank correlations:\n{df_corr.to_string(index=False)}")
    logger.info(f"Saved G-theory tables to {tables_dir}")


if __name__ == "__main__":
    main()
