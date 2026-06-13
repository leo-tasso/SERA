"""Build the report figures and LaTeX number macros from the experiment JSON.

Reads report/results/*.json and writes vector figures into report/figures/ and a
results_macros.tex with every number the report quotes, so the LaTeX never
contains a hand-typed result.

Now multi-seed aware: framework metrics are aggregated as mean +/- standard
deviation across the replication seeds (improvement #1), with companion figures
for training convergence (#2), the clustered Pareto frontier (#4), and a
provenance summary of the panel (#7).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPORT_DIR / "results"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Headline frameworks plotted in the trajectory/delta figures and the table.
OBJECTIVE_ORDER = ["utilitarian", "rawlsian", "cvar", "egalitarian", "wellbeing"]
COLORS = {
    "baseline": "#555555",
    "utilitarian": "#1f77b4",
    "rawlsian": "#d62728",
    "cvar": "#ff7f0e",
    "egalitarian": "#2ca02c",
    "wellbeing": "#9467bd",
}
LABELS = {
    "baseline": "Baseline",
    "utilitarian": "Utilitarian",
    "rawlsian": "Rawlsian (maximin)",
    "cvar": "Rawlsian (CVaR)",
    "egalitarian": "Egalitarian (Sen)",
    "wellbeing": "Wellbeing",
}
# Short tokens for LaTeX macro names, e.g. \cmpUtilGdp.
SHORT = {
    "utilitarian": "Util",
    "rawlsian": "Rawls",
    "cvar": "Cvar",
    "egalitarian": "Egal",
    "wellbeing": "Well",
}


def series_to_xy(series):
    xs = [int(point["year"]) for point in series]
    ys = [float(point["value"]) for point in series]
    return xs, ys


def stack_series(per_seed_results, oid, key):
    """(years, value_matrix[seed, year]) for one objective/series across seeds."""
    years = None
    rows = []
    for entry in per_seed_results:
        result = {r["objectiveId"]: r for r in entry["results"]}.get(oid)
        if not result or key not in result:
            continue
        xs, ys = series_to_xy(result[key])
        years = xs
        rows.append(ys)
    return years, np.array(rows, dtype=float) if rows else np.zeros((0, 0))


def final_values(per_seed_results, oid, key):
    """Vector over seeds of one scalar final-year metric for an objective."""
    values = []
    for entry in per_seed_results:
        result = {r["objectiveId"]: r for r in entry["results"]}.get(oid)
        if result and result.get(key) is not None:
            values.append(float(result[key]))
    return np.array(values, dtype=float)


def main() -> None:
    comparison = json.loads((RESULTS_DIR / "compare_objectives.json").read_text())
    pareto = json.loads((RESULTS_DIR / "pareto_front.json").read_text())
    clustered_path = RESULTS_DIR / "pareto_front_clustered.json"
    clustered = json.loads(clustered_path.read_text()) if clustered_path.exists() else None
    prov_path = RESULTS_DIR / "provenance.json"
    provenance = json.loads(prov_path.read_text()) if prov_path.exists() else {}

    baseline = comparison["baseline"]
    per_seed = comparison["perSeed"]
    seeds = comparison["seeds"]
    present = [
        oid
        for oid in OBJECTIVE_ORDER
        if any(oid in {r["objectiveId"] for r in e["results"]} for e in per_seed)
    ]

    # ------------------------------------------------------------------ #
    # Figure 1: GDP / Gini / worst-off trajectories with +/-1 std bands
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [
        ("gdpByYear", "Total national GDP ($\\times 10^3$, model units)", 1e-3),
        ("giniByYear", "Inter-provincial Gini on GDP p.c.", 1.0),
        ("worstGdpByYear", "Worst-off province GDP p.c. (model units)", 1.0),
    ]
    for ax, (key, title, scale) in zip(axes, panels):
        xs, ys = series_to_xy(baseline[key])
        ax.plot(xs, np.array(ys) * scale, color=COLORS["baseline"], ls="--",
                lw=1.6, label=LABELS["baseline"])
        for oid in present:
            years, matrix = stack_series(per_seed, oid, key)
            if matrix.size == 0:
                continue
            mean = matrix.mean(axis=0) * scale
            std = matrix.std(axis=0) * scale
            ax.plot(years, mean, color=COLORS[oid], lw=1.8, label=LABELS[oid])
            if len(seeds) > 1:
                ax.fill_between(years, mean - std, mean + std, color=COLORS[oid], alpha=0.15)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trajectories.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Figure 2: who gains, who loses (pooled over all seeds)
    # ------------------------------------------------------------------ #
    base_by_prov = baseline["finalGdpByProvince"]
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    deltas_per_obj = []
    for oid in present:
        pooled = []
        for entry in per_seed:
            result = {r["objectiveId"]: r for r in entry["results"]}.get(oid)
            if not result:
                continue
            obj_by_prov = result["finalGdpByProvince"]
            pooled.extend(
                100.0 * (obj_by_prov[code] - base_by_prov[code]) / base_by_prov[code]
                for code in base_by_prov
                if code in obj_by_prov and base_by_prov[code] > 0
            )
        deltas_per_obj.append(pooled)
    parts = ax.boxplot(
        deltas_per_obj,
        tick_labels=[LABELS[o] for o in present],
        showfliers=True,
        whis=(5, 95),
        patch_artist=True,
    )
    for patch, oid in zip(parts["boxes"], present):
        patch.set_facecolor(COLORS[oid])
        patch.set_alpha(0.45)
    for median in parts["medians"]:
        median.set_color("black")
    ax.axhline(0.0, color="#555555", lw=1.0, ls="--")
    ax.set_ylabel("Final-year provincial GDP p.c. vs baseline (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "province_deltas.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Figure 3: training convergence (improvement #2)
    # Relative improvement over the untrained network, per framework, so the
    # different objective scales are comparable on one axis.
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for oid in present:
        curves = []
        for entry in per_seed:
            result = {r["objectiveId"]: r for r in entry["results"]}.get(oid)
            hist = (result or {}).get("trainInfo", {}).get("history") or []
            if len(hist) < 2:
                continue
            start = hist[0]
            if start == 0:
                rel = [0.0 for _ in hist]
            else:
                rel = [100.0 * (value - start) / abs(start) for value in hist]
            curves.append(rel)
        if not curves:
            continue
        width = min(len(c) for c in curves)
        matrix = np.array([c[:width] for c in curves], dtype=float)
        mean = matrix.mean(axis=0)
        ax.plot(range(width), mean, color=COLORS[oid], lw=1.8, marker="o",
                markersize=3, label=LABELS[oid])
        if len(curves) > 1:
            std = matrix.std(axis=0)
            ax.fill_between(range(width), mean - std, mean + std,
                            color=COLORS[oid], alpha=0.12)
    ax.set_xlabel("ES iteration", fontsize=9)
    ax.set_ylabel("Cumulative objective improvement over untrained (%)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Figure 4: uniform Pareto frontier (efficiency vs floor)
    # ------------------------------------------------------------------ #
    def pareto_arrays(front):
        pts = front["points"]
        return (
            np.array([p["finalGdpTotal"] for p in pts]) * 1e-3,
            np.array([p["finalGini"] for p in pts]),
            np.array([p["worstProvinceGdp"] for p in pts]),
        )

    gdp, gini_vals, worst = pareto_arrays(pareto)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(gdp, worst, color="#1f77b4", s=55, edgecolors="black",
               linewidths=0.4, zorder=3, label="Non-dominated uniform policies")
    pb = pareto["baseline"]
    ax.scatter([pb["finalGdpTotal"] * 1e-3], [pb["worstProvinceGdp"]],
               marker="X", color="#555555", s=110, zorder=4, label="Baseline")
    ax.set_xlabel("Total national GDP ($\\times 10^3$, model units, final year)", fontsize=9)
    ax.set_ylabel("Worst-off province GDP p.c. (model units, final year)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pareto.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Figure 5: clustered Pareto frontier (improvement #4)
    # The informative projection here is efficiency vs inequality, because
    # per-province targeting actually moves the Gini.
    # ------------------------------------------------------------------ #
    if clustered is not None:
        cgdp, cgini, cworst = pareto_arrays(clustered)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        scatter = ax.scatter(cgdp, cgini, c=cworst, cmap="viridis", s=60,
                             edgecolors="black", linewidths=0.4, zorder=3)
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label("Worst-off province GDP p.c.", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        pb = clustered["baseline"]
        ax.scatter([pb["finalGdpTotal"] * 1e-3], [pb["finalGini"]],
                   marker="X", color="#d62728", s=120, zorder=4, label="Baseline")
        ax.set_xlabel("Total national GDP ($\\times 10^3$, model units, final year)", fontsize=9)
        ax.set_ylabel("Inter-provincial Gini (final year)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "pareto_clustered.pdf")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # LaTeX macros with every quoted number
    # ------------------------------------------------------------------ #
    lines = []

    def macro(name, value):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def fmt_gdp(v):
        return f"{v * 1e-3:.1f}"

    def fmt_k(v):
        return f"{v:.1f}"

    def fmt_gini(v):
        return f"{v:.3f}"

    macro("cmpHorizon", comparison["horizon"])
    macro("cmpFinalYear", comparison["finalYear"])
    macro("cmpModel", comparison["modelId"])
    macro("cmpSeeds", len(seeds))

    macro("baseGdp", fmt_gdp(baseline["finalGdpTotal"]))
    macro("baseGini", fmt_gini(baseline["finalGini"]))
    macro("baseWorst", fmt_k(baseline["worstProvinceGdp"]))

    base_gdp = baseline["finalGdpTotal"]
    base_worst = baseline["worstProvinceGdp"]
    for oid in present:
        s = SHORT[oid]
        gdp_v = final_values(per_seed, oid, "finalGdpTotal")
        gini_v = final_values(per_seed, oid, "finalGini")
        worst_v = final_values(per_seed, oid, "worstProvinceGdp")
        impr_v = np.array([
            (
                {r["objectiveId"]: r for r in e["results"]}.get(oid, {})
                .get("trainInfo", {})
                .get("improvement_pct", 0.0)
            )
            for e in per_seed
        ], dtype=float)

        macro(f"cmp{s}Gdp", fmt_gdp(gdp_v.mean()))
        macro(f"cmp{s}GdpStd", fmt_gdp(gdp_v.std()))
        macro(f"cmp{s}Gini", fmt_gini(gini_v.mean()))
        macro(f"cmp{s}GiniStd", fmt_gini(gini_v.std()))
        macro(f"cmp{s}Worst", fmt_k(worst_v.mean()))
        macro(f"cmp{s}WorstStd", fmt_k(worst_v.std()))
        macro(f"cmp{s}GdpDelta", f"{100.0 * (gdp_v.mean() - base_gdp) / base_gdp:+.1f}")
        macro(f"cmp{s}WorstDelta", f"{100.0 * (worst_v.mean() - base_worst) / base_worst:+.1f}")
        macro(f"cmp{s}TrainImpr", f"{impr_v.mean():+.1f}")

    # Uniform Pareto macros (unchanged names).
    macro("parPoints", len(pareto["points"]))
    macro("parEvals", pareto["evaluations"])
    macro("parGens", pareto["generations"])
    macro("parPop", pareto["popsize"])
    macro("parGdpSpanLo", f"{gdp.min():.1f}")
    macro("parGdpSpanHi", f"{gdp.max():.1f}")
    macro("parGiniLo", fmt_gini(float(gini_vals.min())))
    macro("parGiniHi", fmt_gini(float(gini_vals.max())))
    macro("parGiniSpanMicro", f"{(gini_vals.max() - gini_vals.min()) * 1e6:.0f}")
    macro("parWorstLo", f"{worst.min():.1f}")
    macro("parWorstHi", f"{worst.max():.1f}")
    corr = float(np.corrcoef(gdp, worst)[0, 1]) if len(gdp) > 2 else 1.0
    macro("parCorr", f"{corr:.3f}")

    # Clustered Pareto macros (improvement #4): a real Gini span emerges.
    if clustered is not None:
        macro("parcPoints", len(clustered["points"]))
        macro("parcClusters", clustered.get("nClusters", 4))
        macro("parcGdpSpanLo", f"{cgdp.min():.1f}")
        macro("parcGdpSpanHi", f"{cgdp.max():.1f}")
        macro("parcGiniLo", fmt_gini(float(cgini.min())))
        macro("parcGiniHi", fmt_gini(float(cgini.max())))
        macro("parcWorstLo", f"{cworst.min():.1f}")
        macro("parcWorstHi", f"{cworst.max():.1f}")

    # Provenance macros (improvement #7).
    if provenance:
        counts = {}
        for label in provenance.values():
            counts[label] = counts.get(label, 0) + 1
        macro("provTotal", len(provenance))
        macro("provMeasured", counts.get("measured", 0))
        macro("provDisaggNational", counts.get("disaggregated_national", 0))
        macro("provDisaggRegional", counts.get("disaggregated_regional", 0))
        macro("provDisaggTotal",
              counts.get("disaggregated_national", 0) + counts.get("disaggregated_regional", 0))
        macro("provGdp", provenance.get("gdp_per_capita", "unknown").replace("_", " "))

    (REPORT_DIR / "results_macros.tex").write_text("\n".join(lines) + "\n")
    print("Figures and results_macros.tex written.")

    # Console summary.
    print("\n=== SUMMARY (mean over %d seeds) ===" % len(seeds))
    print(f"baseline: GDP {fmt_gdp(base_gdp)}  Gini {fmt_gini(baseline['finalGini'])}  "
          f"worst {fmt_k(base_worst)}")
    for oid in present:
        gdp_v = final_values(per_seed, oid, "finalGdpTotal")
        gini_v = final_values(per_seed, oid, "finalGini")
        worst_v = final_values(per_seed, oid, "worstProvinceGdp")
        print(f"{oid:12s}: GDP {fmt_gdp(gdp_v.mean())}+-{fmt_gdp(gdp_v.std())}  "
              f"Gini {fmt_gini(gini_v.mean())}  worst {fmt_k(worst_v.mean())}+-{fmt_k(worst_v.std())}")
    if clustered is not None:
        print(f"clustered pareto Gini span: {cgini.min():.3f}..{cgini.max():.3f}")
    if provenance:
        print("provenance:", counts)


if __name__ == "__main__":
    main()
