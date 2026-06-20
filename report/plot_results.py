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

import sys

sys.path.insert(0, str(REPORT_DIR.parent / "src"))
from sera.twin.province_mapping import PROVINCE_TO_MACROAREA  # noqa: E402

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


def bootstrap_delta_ci(values, base, n=4000, seed=0):
    """95% bootstrap CI for the percent delta of mean(values) vs baseline.

    Returns (mean_delta_pct, lo, hi, half_width). Resamples the seed-level
    values with replacement, so it reflects run-to-run search variance.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0 or base == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n, values.size))].mean(axis=1)
    deltas = 100.0 * (means - base) / base
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    mean_delta = 100.0 * (values.mean() - base) / base
    return float(mean_delta), float(lo), float(hi), float((hi - lo) / 2.0)


def bootstrap_diff_significant(a, b, n=4000, seed=0):
    """True if the PAIRED difference mean(a - b) has a 95% bootstrap CI off zero.

    The frameworks share seeds (each trains with the same ES RNG per seed), so
    their per-seed results are paired; a paired bootstrap resamples seed indices
    once and differences within seed, which removes the shared search-noise
    component and is far more powerful than treating the two samples as
    independent.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return False
    diff = a - b  # paired per-seed differences
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(n, diff.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return bool(lo > 0 or hi < 0)


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
        ax.plot(
            xs,
            np.array(ys) * scale,
            color=COLORS["baseline"],
            ls="--",
            lw=1.6,
            label=LABELS["baseline"],
        )
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
        ax.plot(
            range(width),
            mean,
            color=COLORS[oid],
            lw=1.8,
            marker="o",
            markersize=3,
            label=LABELS[oid],
        )
        if len(curves) > 1:
            std = matrix.std(axis=0)
            ax.fill_between(range(width), mean - std, mean + std, color=COLORS[oid], alpha=0.12)
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
    ax.scatter(
        gdp,
        worst,
        color="#1f77b4",
        s=55,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
        label="Non-dominated uniform policies",
    )
    pb = pareto["baseline"]
    ax.scatter(
        [pb["finalGdpTotal"] * 1e-3],
        [pb["worstProvinceGdp"]],
        marker="X",
        color="#555555",
        s=110,
        zorder=4,
        label="Baseline",
    )
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
        scatter = ax.scatter(
            cgdp,
            cgini,
            c=cworst,
            cmap="viridis",
            s=60,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label("Worst-off province GDP p.c.", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        pb = clustered["baseline"]
        ax.scatter(
            [pb["finalGdpTotal"] * 1e-3],
            [pb["finalGini"]],
            marker="X",
            color="#d62728",
            s=120,
            zorder=4,
            label="Baseline",
        )
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
        impr_v = np.array(
            [
                (
                    {r["objectiveId"]: r for r in e["results"]}
                    .get(oid, {})
                    .get("trainInfo", {})
                    .get("improvement_pct", 0.0)
                )
                for e in per_seed
            ],
            dtype=float,
        )

        macro(f"cmp{s}Gdp", fmt_gdp(gdp_v.mean()))
        macro(f"cmp{s}GdpStd", fmt_gdp(gdp_v.std()))
        macro(f"cmp{s}Gini", fmt_gini(gini_v.mean()))
        macro(f"cmp{s}GiniStd", fmt_gini(gini_v.std()))
        macro(f"cmp{s}Worst", fmt_k(worst_v.mean()))
        macro(f"cmp{s}WorstStd", fmt_k(worst_v.std()))
        macro(f"cmp{s}GdpDelta", f"{100.0 * (gdp_v.mean() - base_gdp) / base_gdp:+.1f}")
        macro(f"cmp{s}WorstDelta", f"{100.0 * (worst_v.mean() - base_worst) / base_worst:+.1f}")
        macro(f"cmp{s}TrainImpr", f"{impr_v.mean():+.1f}")
        # 95% bootstrap CI half-widths on the deltas (improvement #5).
        _, _, _, gdp_hw = bootstrap_delta_ci(gdp_v, base_gdp)
        _, _, _, worst_hw = bootstrap_delta_ci(worst_v, base_worst)
        macro(f"cmp{s}GdpCI", f"{gdp_hw:.1f}")
        macro(f"cmp{s}WorstCI", f"{worst_hw:.1f}")

    # Significance of the headline claims (95% bootstrap CI on the difference
    # of two frameworks' floors excludes zero?). Emitted as Yes/No so the prose
    # can state inference rather than eyeballing.
    def yesno(flag):
        return "significant" if flag else "not significant"

    worst_by = {oid: final_values(per_seed, oid, "worstProvinceGdp") for oid in present}
    if "egalitarian" in worst_by and "utilitarian" in worst_by:
        macro(
            "sigEgalUtilFloor",
            yesno(bootstrap_diff_significant(worst_by["egalitarian"], worst_by["utilitarian"])),
        )
    if "cvar" in worst_by and "rawlsian" in worst_by:
        macro(
            "sigCvarRawlsFloor",
            yesno(bootstrap_diff_significant(worst_by["cvar"], worst_by["rawlsian"])),
        )

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
        macro(
            "provDisaggTotal",
            counts.get("disaggregated_national", 0) + counts.get("disaggregated_regional", 0),
        )
        macro("provGdp", provenance.get("gdp_per_capita", "unknown").replace("_", " "))

    # Twin-structure / data-review macros (improvement #3 + coupling wiring),
    # sourced from the panel-estimation findings the tool writes.
    findings_path = REPORT_DIR.parent / "docs" / "twin_structure_findings.json"
    if findings_path.exists():
        f = json.loads(findings_path.read_text())
        prov = f["province"]
        macro("strSignAgreeFE", f"{round(prov['sign_agreement_fe'] * 100)}")
        macro("strSignAgreePooled", f"{round(prov['sign_agreement_pooled'] * 100)}")
        macro("strEdgesScored", prov["edges_scored"])
        macro("strDirAcc", f"{round(prov['direction_accuracy'] * 100)}")
        macro("strRandomRtwo", f"{prov['random_r2_pooled']:+.2f}")
        macro("strLeversNational", f["lever_granularity"]["national_or_regional_only"])
        macro("strLevers", f["lever_granularity"]["levers"])
        macro("strEdgesFlipped", f["wiring"]["flipped_count"])
        macro("strEdgesDropped", f["wiring"]["dropped_count"])
        macro("strEdgesTotal", f["wiring"]["edges_total"])

    # ------------------------------------------------------------------ #
    # The North--South divide, per framework (the report's motivating cleavage,
    # read off the same per-province results; figure + macros).
    # ------------------------------------------------------------------ #
    def area_means(by_prov):
        acc = {"North": [], "Centre": [], "South": []}
        for code, value in by_prov.items():
            area = PROVINCE_TO_MACROAREA.get(str(code).strip().upper())
            if area:
                acc[area].append(float(value))
        return {a: (float(np.mean(v)) if v else float("nan")) for a, v in acc.items()}

    base_area = area_means(baseline["finalGdpByProvince"])
    macro("nsBaseNorth", f"{base_area['North']:.0f}")
    macro("nsBaseSouth", f"{base_area['South']:.0f}")
    macro("nsBaseRatio", f"{base_area['South'] / base_area['North']:.3f}")
    area_counts = {"North": 0, "Centre": 0, "South": 0}
    for area in PROVINCE_TO_MACROAREA.values():
        area_counts[area] += 1
    macro("nsProvNorth", area_counts["North"])
    macro("nsProvCentre", area_counts["Centre"])
    macro("nsProvSouth", area_counts["South"])

    north_d, south_d, labels_ns = [], [], []
    for oid in present:
        nd, sd, ratios = [], [], []
        for entry in per_seed:
            result = {r["objectiveId"]: r for r in entry["results"]}.get(oid)
            if not result:
                continue
            m = area_means(result["finalGdpByProvince"])
            nd.append(100 * (m["North"] - base_area["North"]) / base_area["North"])
            sd.append(100 * (m["South"] - base_area["South"]) / base_area["South"])
            ratios.append(m["South"] / m["North"])
        if not nd:
            continue
        north_d.append(float(np.mean(nd)))
        south_d.append(float(np.mean(sd)))
        labels_ns.append(oid)
        s = SHORT[oid]
        macro(f"ns{s}North", f"{np.mean(nd):+.1f}")
        macro(f"ns{s}South", f"{np.mean(sd):+.1f}")
        macro(f"ns{s}Ratio", f"{np.mean(ratios):.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    xs_ns = np.arange(len(labels_ns))
    w = 0.38
    ax.bar(
        xs_ns - w / 2,
        north_d,
        w,
        label=f"North ({area_counts['North']} prov.)",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        xs_ns + w / 2,
        south_d,
        w,
        label=f"South / Mezzogiorno ({area_counts['South']} prov.)",
        color="#d62728",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(xs_ns)
    ax.set_xticklabels([LABELS[o] for o in labels_ns], fontsize=8)
    ax.set_ylabel("Mean provincial GDP p.c. vs baseline (%)", fontsize=9)
    ax.set_title("North vs South response by ethical framework", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "north_south.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Prioritarian continuum sweep (improvement #2)
    # ------------------------------------------------------------------ #
    sweep_path = RESULTS_DIR / "prioritarian_sweep.json"
    if sweep_path.exists():
        sw = json.loads(sweep_path.read_text())
        rhos = [p["rho"] for p in sw["points"]]
        b = sw["baseline"]
        gdp_d = [
            100 * (p["gdp_mean"] - b["finalGdpTotal"]) / b["finalGdpTotal"] for p in sw["points"]
        ]
        worst_d = [
            100 * (p["worst_mean"] - b["worstProvinceGdp"]) / b["worstProvinceGdp"]
            for p in sw["points"]
        ]
        ginis = [p["gini_mean"] for p in sw["points"]]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, ys, title in zip(
            axes,
            [gdp_d, ginis, worst_d],
            [
                "Total GDP vs baseline (%)",
                "Inter-provincial Gini",
                "Worst-off province vs baseline (%)",
            ],
        ):
            ax.plot(rhos, ys, "-o", color="#7b3294", lw=1.8, markersize=4)
            ax.set_xlabel(r"prioritarian concavity $\rho$", fontsize=9)
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)
        axes[0].annotate(
            "utilitarian\nend",
            (rhos[0], gdp_d[0]),
            fontsize=7,
            textcoords="offset points",
            xytext=(6, -2),
        )
        axes[0].annotate(
            "maximin\nend",
            (rhos[-1], gdp_d[-1]),
            fontsize=7,
            textcoords="offset points",
            xytext=(-30, 6),
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / "prioritarian_sweep.pdf")
        plt.close(fig)
        macro("sweepSeeds", len(sw.get("seeds", [])))
        macro("sweepRhoHi", f"{rhos[-1]:.2f}")
        macro("sweepGdpAtZero", f"{gdp_d[0]:+.1f}")
        macro("sweepGdpAtHi", f"{gdp_d[-1]:+.1f}")
        macro("sweepWorstAtZero", f"{worst_d[0]:+.1f}")
        macro("sweepWorstAtHi", f"{worst_d[-1]:+.1f}")
        macro("sweepGiniAtZero", fmt_gini(ginis[0]))
        macro("sweepGiniAtHi", fmt_gini(ginis[-1]))

    # ------------------------------------------------------------------ #
    # Propagation-mode ablation (improvement #3)
    # ------------------------------------------------------------------ #
    abl_path = RESULTS_DIR / "ablation.json"
    if abl_path.exists():
        ab = json.loads(abl_path.read_text())

        def mode_metric(mode_block, oid, key):
            vals = []
            for entry in mode_block["perSeed"]:
                r = {x["objectiveId"]: x for x in entry["results"]}.get(oid)
                if r and r.get(key) is not None:
                    vals.append(float(r[key]))
            return float(np.mean(vals)) if vals else float("nan")

        mode_tokens = {"signless": "Signless", "documented": "Documented", "reviewed": "Reviewed"}
        for mode, tok in mode_tokens.items():
            block = ab["byMode"].get(mode)
            if not block:
                continue
            bw = block["baseline"]["worstProvinceGdp"]
            bg = block["baseline"]["finalGdpTotal"]
            # Egalitarian is the most sign-sensitive row; utilitarian Gini as a control.
            egal_gini = mode_metric(block, "egalitarian", "finalGini")
            egal_worst = mode_metric(block, "egalitarian", "worstProvinceGdp")
            rawls_worst = mode_metric(block, "rawlsian", "worstProvinceGdp")
            util_gini = mode_metric(block, "utilitarian", "finalGini")
            macro(f"abl{tok}EgalGini", fmt_gini(egal_gini))
            macro(f"abl{tok}EgalWorst", f"{100 * (egal_worst - bw) / bw:+.1f}")
            macro(f"abl{tok}RawlsWorst", f"{100 * (rawls_worst - bw) / bw:+.1f}")
            macro(f"abl{tok}UtilGini", fmt_gini(util_gini))
        macro("ablSeeds", len(ab["seeds"]))

    # ------------------------------------------------------------------ #
    # Price of transparency (improvement #1)
    # ------------------------------------------------------------------ #
    trans_path = RESULTS_DIR / "transparency.json"
    if trans_path.exists():
        tr = json.loads(trans_path.read_text())
        by_id = {r["modelId"]: r for r in tr["results"]}
        neural_score = (by_id.get("neural") or {}).get("score_mean")
        badge_text = {
            "white-box": "white box",
            "gray-box": "gray box",
            "black-box": "black box",
            None: "black box",
        }
        tok = {
            "neural": "Neural",
            "linear": "Linear",
            "rules": "Rules",
            "cluster_cem": "Cluster",
            "uniform_cem": "UnifCem",
            "uniform_bayes": "UnifBayes",
        }
        for mid, t in tok.items():
            r = by_id.get(mid)
            if not r or r.get("score_mean") is None:
                continue
            rel = 100.0 * r["score_mean"] / neural_score if neural_score else float("nan")
            macro(f"trans{t}Rel", f"{rel:.1f}")
            macro(f"trans{t}Gap", f"{100.0 - rel:+.1f}")
            macro(f"trans{t}Badge", badge_text.get(r.get("explainability"), "white box"))
            if r.get("gdp_mean") is not None:
                macro(f"trans{t}Gdp", fmt_gdp(r["gdp_mean"]))
                macro(f"trans{t}Gini", fmt_gini(r["gini_mean"]))
        macro("transSeeds", len(tr.get("seeds", [])))
        # Bar chart of the score gap (price of transparency) vs the neural policy.
        order = [m for m in tok if m in by_id and by_id[m].get("score_mean") is not None]
        gaps = [100.0 * (1 - by_id[m]["score_mean"] / neural_score) for m in order]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        colors = ["#999999" if m == "neural" else "#1f77b4" for m in order]
        ax.bar([tok[m] for m in order], gaps, color=colors, edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_ylabel("Objective score below neural (%)", fontsize=9)
        ax.set_title("Price of transparency (utilitarian objective)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "transparency.pdf")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Sufficientarian threshold sweep (improvement #4)
    # ------------------------------------------------------------------ #
    suff_path = RESULTS_DIR / "sufficientarian_sweep.json"
    if suff_path.exists():
        su = json.loads(suff_path.read_text())
        th = [p["theta"] for p in su["points"]]
        b = su["baseline"]
        gdp_d = [
            100 * (p["gdp_mean"] - b["finalGdpTotal"]) / b["finalGdpTotal"] for p in su["points"]
        ]
        worst_d = [
            100 * (p["worst_mean"] - b["worstProvinceGdp"]) / b["worstProvinceGdp"]
            for p in su["points"]
        ]
        ginis = [p["gini_mean"] for p in su["points"]]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, ys, title in zip(
            axes,
            [gdp_d, ginis, worst_d],
            [
                "Total GDP vs baseline (%)",
                "Inter-provincial Gini",
                "Worst-off province vs baseline (%)",
            ],
        ):
            ax.plot(th, ys, "-o", color="#2c7fb8", lw=1.8, markersize=4)
            ax.set_xlabel(r"sufficiency threshold $\theta$ ($\times$ start median)", fontsize=9)
            ax.set_title(title, fontsize=9)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "sufficientarian_sweep.pdf")
        plt.close(fig)
        macro("suffSeeds", len(su.get("seeds", [])))
        macro("suffThetaLo", f"{th[0]:.2f}")
        macro("suffThetaHi", f"{th[-1]:.2f}")
        macro("suffGdpAtLo", f"{gdp_d[0]:+.1f}")
        macro("suffGdpAtHi", f"{gdp_d[-1]:+.1f}")
        macro("suffWorstAtLo", f"{worst_d[0]:+.1f}")
        macro("suffWorstAtHi", f"{worst_d[-1]:+.1f}")
        macro("suffGiniAtLo", fmt_gini(ginis[0]))
        macro("suffGiniAtHi", fmt_gini(ginis[-1]))

    (REPORT_DIR / "results_macros.tex").write_text("\n".join(lines) + "\n")
    print("Figures and results_macros.tex written.")

    # Console summary.
    print("\n=== SUMMARY (mean over %d seeds) ===" % len(seeds))
    print(
        f"baseline: GDP {fmt_gdp(base_gdp)}  Gini {fmt_gini(baseline['finalGini'])}  "
        f"worst {fmt_k(base_worst)}"
    )
    for oid in present:
        gdp_v = final_values(per_seed, oid, "finalGdpTotal")
        gini_v = final_values(per_seed, oid, "finalGini")
        worst_v = final_values(per_seed, oid, "worstProvinceGdp")
        print(
            f"{oid:12s}: GDP {fmt_gdp(gdp_v.mean())}+-{fmt_gdp(gdp_v.std())}  "
            f"Gini {fmt_gini(gini_v.mean())}  worst {fmt_k(worst_v.mean())}+-{fmt_k(worst_v.std())}"
        )
    if clustered is not None:
        print(f"clustered pareto Gini span: {cgini.min():.3f}..{cgini.max():.3f}")
    if provenance:
        print("provenance:", counts)


if __name__ == "__main__":
    main()
