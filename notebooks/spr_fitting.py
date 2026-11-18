"""Reproducible model comparisons for the four SPR teaching notebooks.

Forward binding models come from pyKinGenie. This notebook-only driver makes
event times, occupied initial states, bulk offsets and observation masks explicit.
It does not change the production fitter or infer missing physical units.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pykingenie.utils import signal_surface as signals


@dataclass
class Trace:
    name: str
    time: np.ndarray
    response: np.ndarray
    background: float
    initial: float
    concentration: float | None
    indices: np.ndarray


def prepare(
    experiment, metadata, trim=3.0, spacing=1.0, dissociation_fraction=1.0, shift=0.0
):
    """Select observations without changing the physical clock or filling gaps."""
    meta = dict(metadata)
    meta["association_start"] += shift
    meta["association_end"] += shift
    start, end = meta["association_start"], meta["association_end"]
    stop = end + dissociation_fraction * (meta["fit_end"] - end)
    traces = []
    for i, name in enumerate(experiment.sensor_names):
        t, raw = np.concatenate(experiment.xs[i]), np.concatenate(experiment.ys[i])
        lo, hi = meta["baseline_window"]
        baseline = float(np.median(raw[(t >= lo) & (t <= hi)]))
        occupied = metadata["regenerated"] is False
        background = 0.0 if occupied else baseline
        initial = baseline if occupied else 0.0
        selected = []
        for mask in [
            (t >= start + trim) & (t <= end - trim),
            (t >= end + trim) & (t <= stop),
        ]:
            idx = np.flatnonzero(mask)
            # At most one original observation per spacing interval, not an average.
            bins = np.floor((t[idx] - start) / spacing).astype(int)
            _, first = np.unique(bins, return_index=True)
            selected.extend(idx[first])
        concentration = (
            None if meta["concentrations"] is None else meta["concentrations"][i]
        )
        traces.append(
            Trace(
                name,
                t,
                raw - background,
                background,
                initial,
                concentration,
                np.asarray(selected, dtype=int),
            )
        )
    return traces, meta


def _parameters(family, local, traces, meta):
    c = np.asarray(meta["concentrations"] or [1.0])
    cscale = float(np.median(c))
    max_response = max(float(np.max(t.response[t.indices])) for t in traces)
    off = 1e-3 if not meta["molar_concentrations_verified"] else 1e-4
    kd = cscale / (5 if not meta["molar_concentrations_verified"] else 15)
    spec = []

    def add(name, value, low, high, log=True):
        spec.append(dict(name=name, start=value, low=low, high=high, log=log))

    if family in ("one_site", "mass_transport"):
        add("Kd", kd, c.min() * 1e-5, c.max() * 1e5)
        add("k_off", off, 1e-7, 1)
        if family == "mass_transport":
            kt = off / kd * max_response * 2
            add("Ktr", kt, kt * 1e-5, kt * 1e5)
    elif family == "heterogeneous":
        for j, factor in [(1, 0.2), (2, 5.0)]:
            add(f"Kd{j}", kd * factor, c.min() * 1e-5, c.max() * 1e5)
            add(f"k_off{j}", off * factor, 1e-7, 1)
        add("fraction", 0.5, 0.01, 0.99, log=False)
    elif family == "induced_fit":
        add("k_on", off / kd, 1e-8 / cscale, 10 / cscale)
        add("k_off", off * 2, 1e-7, 1)
        add("k_c", 1e-3, 1e-7, 1)
        add("k_rev", 1e-3, 1e-7, 1)
    elif family in ("single_exp", "single_exp_shared_rates"):
        for i in range(len(traces) if local and family == "single_exp" else 1):
            add(f"k_obs_{i + 1}", 0.03, 1e-5, 1)
            add(f"k_decay_{i + 1}", 0.002, 1e-7, 1)
    elif family == "double_exp":
        for j, factor in [(1, 0.5), (2, 2.0)]:
            add(f"k_obs{j}", 0.03 * factor, 1e-5, 1)
            add(f"k_decay{j}", 0.002 * factor, 1e-7, 1)
        add("fraction", 0.5, 0.01, 0.99, log=False)
    for i in range(len(traces) if local else 1):
        selected = [traces[i]] if local else traces
        ymax = max(float(np.max(t.response[t.indices])) for t in selected)
        low = max(max(t.initial for t in selected) + 0.05, ymax * 0.2, 0.1)
        add(f"Rmax_{i + 1}", max(ymax * 1.5, low * 1.1), low, max_response * 100)
    if meta["bulk_shift"]:
        for i in range(len(traces)):
            add(f"RI_{i + 1}", 0, -max_response * 0.3, max_response * 0.3, log=False)
    return spec


def _encode(spec, field):
    return np.asarray([np.log10(p[field]) if p["log"] else p[field] for p in spec])


def _decode(spec, q):
    return {s["name"]: float(10 ** v if s["log"] else v) for s, v in zip(spec, q)}


def predict(trace, meta, family, local, params, i, time=None):
    """Evaluate at true event-relative times, including masked injection points."""
    time = trace.time if time is None else np.asarray(time)
    start, end = meta["association_start"], meta["association_end"]
    assoc = (time >= start) & (time < end)
    disso = (time >= end) & (time <= meta["fit_end"])
    ta = np.r_[0.0, time[assoc] - start, end - start]
    td = np.r_[0.0, time[disso] - end]
    rmax = params[f"Rmax_{i + 1 if local else 1}"]
    initial = trace.initial
    if family == "one_site":
        a = signals.one_site_association_analytical(
            ta, initial, rmax, params["k_off"], params["Kd"], trace.concentration
        )
        d = signals.one_site_dissociation_analytical(td, a[-1], params["k_off"])
    elif family == "mass_transport":
        a = signals.one_site_mass_transport_association_analytical(
            ta,
            initial,
            trace.concentration,
            params["Kd"],
            params["k_off"],
            params["Ktr"],
            rmax,
        )
        d = signals.one_site_mass_transport_dissociation_analytical(
            td, a[-1], params["Kd"], params["k_off"], params["Ktr"], rmax
        )
    elif family == "heterogeneous":
        a_states = signals.solve_two_site_heterogeneous_ligand_association(
            ta,
            trace.concentration,
            params["Kd1"],
            params["k_off1"],
            params["Kd2"],
            params["k_off2"],
            Rmax=rmax,
            fraction_site1=params["fraction"],
        )
        a = a_states[:, 0]
        d = signals.solve_two_site_heterogeneous_ligand_dissociation(
            td,
            params["k_off1"],
            params["k_off2"],
            s1_0=a_states[-1, 1],
            s2_0=a_states[-1, 2],
        )[:, 0]
    elif family == "induced_fit":
        a_states = signals.solve_induced_fit_association(
            ta,
            trace.concentration,
            params["k_on"],
            params["k_off"],
            params["k_c"],
            params["k_rev"],
            smax=rmax,
        )
        a = a_states[:, 0]
        d = signals.solve_induced_fit_dissociation(
            td,
            params["k_off"],
            params["k_c"],
            params["k_rev"],
            s0=a[-1],
            sP2L=a_states[-1, 2],
            smax=rmax,
        )[:, 0]
    elif family in ("single_exp", "single_exp_shared_rates"):
        tag = i + 1 if local and family == "single_exp" else 1
        a = rmax * (-np.expm1(-params[f"k_obs_{tag}"] * ta))
        d = a[-1] * np.exp(-params[f"k_decay_{tag}"] * td)
    else:
        fraction = params["fraction"]
        a1 = fraction * rmax * (-np.expm1(-params["k_obs1"] * ta))
        a2 = (1 - fraction) * rmax * (-np.expm1(-params["k_obs2"] * ta))
        a = a1 + a2
        d = a1[-1] * np.exp(-params["k_decay1"] * td) + a2[-1] * np.exp(
            -params["k_decay2"] * td
        )
    out = np.full(time.shape, np.nan)
    out[assoc] = a[1:-1] + params.get(f"RI_{i + 1}", 0)
    out[disso] = d[1:]
    return out


def _canonical(spec, q, family):
    values = _decode(spec, q)
    pairs = (
        [("Kd1", "Kd2"), ("k_off1", "k_off2")]
        if family == "heterogeneous"
        else [("k_decay1", "k_decay2"), ("k_obs1", "k_obs2")]
    )
    if (
        family in ("heterogeneous", "double_exp")
        and values[pairs[0][0]] > values[pairs[0][1]]
    ):
        for a, b in pairs:
            values[a], values[b] = values[b], values[a]
        values["fraction"] = 1 - values["fraction"]
        return np.asarray(
            [
                np.log10(values[s["name"]]) if s["log"] else values[s["name"]]
                for s in spec
            ]
        )
    return q


def fit_candidate(
    traces, meta, family, local, n_starts=8, seed=2918, start_params=None, max_nfev=700
):
    spec = _parameters(family, local, traces, meta)
    q0, lo, hi = (_encode(spec, field) for field in ("start", "low", "high"))
    if start_params:
        q0 = np.asarray(
            [
                (
                    np.log10(start_params[s["name"]])
                    if s["log"]
                    else start_params[s["name"]]
                )
                for s in spec
            ]
        )
    target = np.concatenate([t.response[t.indices] for t in traces])
    times = [t.time[t.indices] for t in traces]

    def residual(q):
        params = _decode(spec, q)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            fitted = np.concatenate(
                [
                    predict(t, meta, family, local, params, i, time=time)
                    for i, (t, time) in enumerate(zip(traces, times))
                ]
            )
        return np.nan_to_num(fitted - target, nan=1e10, posinf=1e10, neginf=-1e10)

    rng = np.random.default_rng(seed)
    starts = [q0]
    for j in range(n_starts - 1):
        jitter = np.asarray(
            [
                (
                    rng.normal(0, 0.8)
                    if s["log"]
                    else rng.normal(0, 0.08 * (s["high"] - s["low"]))
                )
                for s in spec
            ]
        )
        starts.append(np.clip(q0 + jitter, lo + 1e-7, hi - 1e-7))
    runs = []
    for start in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = least_squares(
                residual,
                np.clip(start, lo + 1e-7, hi - 1e-7),
                bounds=(lo, hi),
                x_scale="jac",
                max_nfev=max_nfev,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
        runs.append(fit)
    successful = [r for r in runs if r.success]
    if not successful:
        raise RuntimeError(f"{family}: no starting point converged")
    best = min(successful, key=lambda r: np.dot(r.fun, r.fun))
    q = _canonical(spec, best.x.copy(), family)
    r = residual(q)
    rss = float(r @ r)
    n = len(r)
    k = len(q)
    # Central differences in optimizer coordinates; do not use a pseudoinverse
    # to turn a singular direction into an artificially precise confidence interval.
    steps = np.asarray(
        [1e-5 if s["log"] else 1e-5 * max(1, abs(v)) for s, v in zip(spec, q)]
    )
    jac = np.column_stack(
        [
            (residual(q + np.eye(k)[i] * step) - residual(q - np.eye(k)[i] * step))
            / (2 * step)
            for i, step in enumerate(steps)
        ]
    )
    scaled = jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-100)
    condition = float(np.linalg.cond(scaled))
    if np.linalg.matrix_rank(scaled) == k and condition < 1e8:
        _, singular_values, vt = np.linalg.svd(jac, full_matrices=False)
        covariance = (vt.T / singular_values**2) @ vt * rss / (n - k)
        stderr = np.sqrt(np.maximum(np.diag(covariance), 0))
        corr = covariance / np.outer(stderr, stderr)
        max_corr = float(np.max(np.abs(corr - np.eye(k))))
    else:
        stderr = np.full(k, np.nan)
        corr = np.full((k, k), np.nan)
        max_corr = np.nan
    params = _decode(spec, q)
    intervals = []
    for s, value, se in zip(spec, q, stderr):
        low_ci, high_ci = value - 1.96 * se, value + 1.96 * se
        if s["log"]:
            low_ci, high_ci = 10 ** np.clip(low_ci, -300, 300), 10 ** np.clip(
                high_ci, -300, 300
            )
        intervals.append(
            {
                "Parameter": s["name"],
                "Estimate": params[s["name"]],
                "Conditional 95% lower": low_ci,
                "Conditional 95% upper": high_ci,
            }
        )
    near = np.minimum((q - lo) / (hi - lo), (hi - q) / (hi - lo)) < 0.01
    per_trace = []
    offset = 0
    for trace in traces:
        rr = r[offset : offset + len(trace.indices)]
        offset += len(rr)
        t = trace.time[trace.indices]
        pairs = np.flatnonzero(
            (np.diff(t) > 0) & (np.diff(t) <= 3 * np.median(np.diff(t)))
        )
        lag = (
            float(np.corrcoef(rr[pairs], rr[pairs + 1])[0, 1])
            if len(pairs) > 2
            else np.nan
        )
        per_trace.append(
            {
                "Curve": trace.name,
                "Points": len(rr),
                "RMSE": float(np.sqrt(np.mean(rr**2))),
                "Lag-1 residual correlation": lag,
            }
        )
        baseline_idx = np.flatnonzero(trace.time < meta["association_start"] - 3)
        if len(baseline_idx) > 3:
            bins = np.floor(trace.time[baseline_idx]).astype(int)
            _, first = np.unique(bins, return_index=True)
            delta = np.diff(trace.response[baseline_idx[first]])
            sigma = float(
                np.median(np.abs(delta - np.median(delta))) / (0.67448975 * np.sqrt(2))
            )
        else:
            sigma = np.nan
        per_trace[-1]["Baseline noise estimate"] = sigma
        per_trace[-1]["RMSE / baseline noise"] = (
            per_trace[-1]["RMSE"] / sigma if sigma > 0 else np.nan
        )
    near_runs = [
        _decode(spec, _canonical(spec, x.x.copy(), family))
        for x in successful
        if n * np.log(float(x.fun @ x.fun) / rss) <= 2.0 + 1e-8
    ]
    spread = {
        name: max(x[name] for x in near_runs) / min(x[name] for x in near_runs)
        for name in params
        if not name.startswith("RI_")
    }
    return dict(
        family=family,
        local=local,
        params=params,
        n=n,
        k=k,
        rss=rss,
        bic=n * np.log(rss / n) + k * np.log(n),
        rmse=np.sqrt(rss / n),
        redchi=rss / (n - k),
        condition=condition,
        max_corr=max_corr,
        bound_parameters=[s["name"] for s, b in zip(spec, near) if b],
        starts_converged=len(successful),
        starts_tried=len(runs),
        near_optimal_spread=spread,
        parameters=pd.DataFrame(intervals),
        correlation=pd.DataFrame(corr, index=list(params), columns=list(params)),
        residuals=pd.DataFrame(per_trace),
        traces=traces,
        meta=meta,
        failed_starts=len(runs) - len(successful),
    )


def run_analysis(experiment, metadata, n_starts=8, sensitivity=True, progress=False):
    traces, meta = prepare(experiment, metadata)
    empirical = metadata["concentrations"] is None
    if empirical:
        candidates = [
            ("single_exp_shared_rates", True),
            ("single_exp", True),
            ("double_exp", True),
        ]
    else:
        families = ["one_site", "mass_transport"]
        if metadata["regenerated"] is not False:
            families += ["heterogeneous", "induced_fit"]
        candidates = [(family, local) for family in families for local in (False, True)]
    results = []
    failures = []
    for family, local in candidates:
        try:
            fit = fit_candidate(traces, meta, family, local, n_starts=n_starts)
            results.append(fit)
            if progress:
                print(
                    f'{family} / {"local" if local else "shared"} Rmax: BIC {fit["bic"]:.2f}, RMSE {fit["rmse"]:.4g}',
                    flush=True,
                )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            failures.append(
                {"Model": family, "Local Rmax": local, "Failure": str(error)}
            )
            if progress:
                print("FAILED", family, local, str(error), flush=True)
    if not results:
        raise RuntimeError("No candidate model converged")
    results.sort(key=lambda r: r["bic"])
    best = results[0]
    comparisons = pd.DataFrame(
        [
            {
                "Model": r["family"],
                "Rmax": "local" if r["local"] else "shared",
                "Parameters": r["k"],
                "Points": r["n"],
                "Reduced χ² (unit variance)": r["redchi"],
                "RMSE": r["rmse"],
                "BIC": r["bic"],
                "ΔBIC": r["bic"] - best["bic"],
                "Jacobian condition": r["condition"],
                "Max |parameter correlation|": r["max_corr"],
                "Near bounds": ", ".join(r["bound_parameters"]),
                "Converged starts": f'{r["starts_converged"]}/{r["starts_tried"]}',
            }
            for r in results
        ]
    )
    robustness = []
    if sensitivity:
        settings = [
            ("trim 6 s", dict(trim=6.0)),
            ("trim 10 s", dict(trim=10.0)),
            ("5 s observation spacing", dict(spacing=5.0)),
            ("half dissociation window", dict(dissociation_fraction=0.5)),
            ("event times -1 s", dict(shift=-1.0)),
            ("event times +1 s", dict(shift=1.0)),
        ]
        for label, settings_one in settings:
            tt, mm = prepare(experiment, metadata, **settings_one)
            try:
                f = fit_candidate(
                    tt,
                    mm,
                    best["family"],
                    best["local"],
                    n_starts=2,
                    start_params=best["params"],
                    max_nfev=3000,
                )
                row = {"Perturbation": label, "RMSE": f["rmse"]}
                row.update(
                    {
                        name: value / best["params"][name]
                        for name, value in f["params"].items()
                        if not name.startswith(("RI_", "Rmax_"))
                    }
                )
                robustness.append(row)
            except (RuntimeError, ValueError, FloatingPointError) as error:
                robustness.append({"Perturbation": label, "Failure": str(error)})
    return dict(
        best=best,
        results=results,
        comparison=comparisons,
        failures=pd.DataFrame(failures),
        sensitivity=pd.DataFrame(robustness),
        metadata=metadata,
    )


def plot_fit(analysis):
    best = analysis["best"]
    meta = best["meta"]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        constrained_layout=True,
    )
    for i, trace in enumerate(best["traces"]):
        window = (trace.time >= min(meta["baseline_window"])) & (
            trace.time <= meta["fit_end"]
        )
        time = trace.time[window]
        raw = trace.response[window] + trace.background
        pred = (
            predict(
                trace, meta, best["family"], best["local"], best["params"], i, time=time
            )
            + trace.background
        )
        gaps = np.flatnonzero(np.diff(time) > 3 * np.median(np.diff(time))) + 1
        plot_time = np.insert(time, gaps, np.nan)
        axes[0].plot(
            plot_time,
            np.insert(raw, gaps, np.nan),
            color=f"C{i}",
            alpha=0.45,
            linewidth=0.8,
        )
        axes[0].plot(
            plot_time,
            np.insert(pred, gaps, np.nan),
            color=f"C{i}",
            linestyle="--",
            linewidth=2,
            label=f"Curve {i + 1} fit",
        )
        t = trace.time[trace.indices]
        r = trace.response[trace.indices] - predict(
            trace, meta, best["family"], best["local"], best["params"], i, time=t
        )
        axes[1].plot(t, r, ".", color=f"C{i}", markersize=2, label=f"Curve {i + 1}")
    for axis in axes:
        for boundary in [meta["association_start"], meta["association_end"]]:
            axis.axvspan(boundary - 3, boundary + 3, color="grey", alpha=0.12)
        axis.grid(alpha=0.15)
    axes[0].set(
        title=f'Lowest conditional BIC: {best["family"]}, {"local" if best["local"] else "shared"} Rmax',
        ylabel="Measured response",
    )
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Original exported time (s)", ylabel="Data − fit")
    return fig


def parameter_table(analysis):
    """Label concentration-dependent parameters without inventing molar units."""
    table = analysis["best"]["parameters"].copy()
    concentration_unit = (
        "µM" if analysis["metadata"]["molar_concentrations_verified"] else "label unit"
    )
    response_unit = (
        "exported response unit"
        if analysis["metadata"]["file"] == "paired_run.csv"
        else "RU"
    )

    def units(name):
        if name.startswith("Kd"):
            return concentration_unit
        if name == "k_on":
            return f"1/({concentration_unit}·s)"
        if name == "Ktr":
            return f"{response_unit}/({concentration_unit}·s)"
        if name.startswith("k_"):
            return "1/s"
        if name.startswith(("Rmax", "RI_")):
            return response_unit
        return "dimensionless"

    table.insert(1, "Unit", [units(name) for name in table.Parameter])
    values = analysis["best"]["params"]
    for suffix in ("", "1", "2"):
        kd, off = "Kd" + suffix, "k_off" + suffix
        if kd in values and off in values:
            table.loc[len(table)] = {
                "Parameter": "k_on" + suffix,
                "Unit": f"1/({concentration_unit}·s)",
                "Estimate": values[off] / values[kd],
                "Conditional 95% lower": np.nan,
                "Conditional 95% upper": np.nan,
            }
    return table


def trust_assessment(key, analysis):
    """Scientific conclusions for these data, separate from optimizer convergence."""
    sens = analysis["sensitivity"].set_index("Perturbation")
    if key == "paired_run":
        verdict = "No validated kinetic result; lowest BIC is a comparative fit only."
        reasons = [
            "The concentration unit and injection events are not independently verified.",
            "All tested models retain strongly structured residuals; the best residual lag-1 correlations exceed 0.99.",
            "The two-population fit has almost perfectly correlated parameters and a broad second-affinity interval.",
            "Halving the dissociation window changes the second affinity by roughly two orders of magnitude and drives the population fraction to its bound.",
            "No affinity in µM or M can be reported. Quantitative rates remain conditional on inferred timing and a misspecified response model.",
        ]
    elif key == "biacore_cycle_fc":
        verdict = "Do not trust binding constants or the two-component interpretation."
        reasons = [
            "This is a multi-event binning protocol with no concentration metadata; only the early inferred episode is fitted.",
            "The two association rates coalesce and their confidence intervals are effectively unbounded.",
            "Large, smooth residuals persist despite a lower BIC. Small shifts of the inferred event times change the error markedly.",
            "Empirical k_obs / decay rates are not kon, koff or Kd for the complete run.",
        ]
    elif key == "biacore_cycle_sample":
        verdict = "Good descriptive fit; the two-population mechanism and nominal precision are not validated."
        half = sens.loc["half dissociation window"]
        minus = sens.loc["event times -1 s"]
        kd_change = 100 * (half.Kd1 - 1)
        off_change = 100 * (half.k_off1 - 1)
        minor_change = 100 * (minus.Kd2 - 1)
        reasons = [
            "Molar concentrations and event durations come from the original sample sheet; association-only RI terms are fitted and counted.",
            "The local-capacity two-population model wins among the tested models, but its residuals remain strongly correlated and exceed baseline noise.",
            f"Halving dissociation changes the dominant Kd by {kd_change:.0f}% and koff by {off_change:.0f}%; shifting events -1 s changes the minor-population Kd by {minor_change:.0f}%.",
            "Treat the dominant slow component as provisional. Do not infer two molecular site classes or quote the iid least-squares intervals as validated uncertainty.",
        ]
    else:
        verdict = "Use a one-site description; do not trust precise intrinsic kinetic constants from these three curves."
        half = sens.loc["half dissociation window"]
        kd_change = 100 * (half.Kd - 1)
        off_change = 100 * (half.k_off - 1)
        reasons = [
            "Use the verified concentrations and retain nonzero pre-association occupancy; the run was not regenerated.",
            "One-site with local Rmax wins; adding transport does not improve the residuals and its transport parameter is unidentifiable.",
            f"Halving dissociation increases Kd by {kd_change:.0f}% and koff by {off_change:.0f}%, far beyond the nominal iid intervals.",
            "Only about 5% of the signal dissociates in the available window; longer dissociation, more concentrations and controls are needed for precise constants.",
        ]
    return {"Verdict": verdict, "Reasons": reasons}


def save_report(analysis, path, key):
    """Write a compact audit report without embedding raw arrays or executable objects."""
    import scipy

    document = dict(
        case=key,
        assessment=trust_assessment(key, analysis),
        metadata=analysis["metadata"],
        comparison=analysis["comparison"].to_dict(orient="records"),
        parameters=parameter_table(analysis).to_dict(orient="records"),
        residuals=analysis["best"]["residuals"].to_dict(orient="records"),
        sensitivity=analysis["sensitivity"].to_dict(orient="records"),
        all_candidate_parameters=[
            dict(model=r["family"], local_rmax=r["local"], params=r["params"])
            for r in analysis["results"]
        ],
        failures=analysis["failures"].to_dict(orient="records"),
        settings=dict(
            seed=2918,
            multi_starts=analysis["best"]["starts_tried"],
            transition_guard_seconds=3,
            minimum_spacing_seconds=1,
            scipy=scipy.__version__,
            numpy=np.__version__,
            bic="n*log(RSS/n)+k*log(n); conditional independent equal-variance errors",
            intervals="Local iid Jacobian intervals, not calibrated for correlated residuals",
        ),
    )

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(x) for x in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    Path(path).write_text(
        json.dumps(clean(document), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
