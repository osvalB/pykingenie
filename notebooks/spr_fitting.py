"""One-site fits for the two SPR examples with published reference reports.

Uses pyKinGenie forward functions with explicit event times, association
alignment, local capacities and the non-regenerated Carterra initial state.
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
    alignment_offset: float = 0.0


def prepare(experiment, metadata):
    """Select observations without changing the physical clock or filling gaps."""
    meta = dict(metadata)
    start, end = meta["association_start"], meta["association_end"]
    stop = meta["fit_end"]
    trim, spacing = 3.0, 1.0
    traces = []
    for i, name in enumerate(experiment.sensor_names):
        t, raw = np.concatenate(experiment.xs[i]), np.concatenate(experiment.ys[i])
        lo, hi = meta["baseline_window"]
        baseline = float(np.median(raw[(t >= lo) & (t <= hi)]))
        occupied = metadata["regenerated"] is False
        alignment = metadata.get("association_alignment")
        if not alignment:
            raise ValueError(
                "Call align_association and record its offsets before fitting"
            )
        removed = alignment["offsets_by_sensor"][name]
        background = 0.0
        initial = baseline + removed if occupied else 0.0
        alignment_offset = removed if occupied else 0.0
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
                alignment_offset,
            )
        )
    return traces, meta


def _parameters(traces, meta):
    c = np.asarray(meta["concentrations"])
    max_response = max(
        float(np.max(t.response[t.indices] + t.alignment_offset)) for t in traces
    )
    spec = [
        dict(
            name="Kd",
            start=float(np.median(c)) / 15,
            low=c.min() * 1e-5,
            high=c.max() * 1e5,
            log=True,
        ),
        dict(name="k_off", start=1e-4, low=1e-7, high=1, log=True),
    ]
    for i, trace in enumerate(traces):
        ymax = float(np.max(trace.response[trace.indices] + trace.alignment_offset))
        low = max(trace.initial + 0.05, ymax * 0.2, 0.1)
        spec.append(
            dict(
                name=f"Rmax_{i + 1}",
                start=max(ymax * 1.5, low * 1.1),
                low=low,
                high=max_response * 100,
                log=True,
            )
        )
    if meta["bulk_shift"]:
        for i in range(len(traces)):
            spec.append(
                dict(
                    name=f"RI_{i + 1}",
                    start=0,
                    low=-max_response * 0.3,
                    high=max_response * 0.3,
                    log=False,
                )
            )
    return spec


def _encode(spec, field):
    return np.asarray([np.log10(p[field]) if p["log"] else p[field] for p in spec])


def _decode(spec, q):
    return {s["name"]: float(10 ** v if s["log"] else v) for s, v in zip(spec, q)}


def predict(trace, meta, params, i, time=None):
    """One-site binding on the true event clock, in aligned response units."""
    time = trace.time if time is None else np.asarray(time)
    start, end = meta["association_start"], meta["association_end"]
    assoc = (time >= start) & (time < end)
    disso = (time >= end) & (time <= meta["fit_end"])
    ta = np.r_[0.0, time[assoc] - start, end - start]
    td = np.r_[0.0, time[disso] - end]
    a = signals.one_site_association_analytical(
        ta,
        trace.initial,
        params[f"Rmax_{i + 1}"],
        params["k_off"],
        params["Kd"],
        trace.concentration,
    )
    d = signals.one_site_dissociation_analytical(td, a[-1], params["k_off"])
    out = np.full(time.shape, np.nan)
    out[assoc] = a[1:-1] + params.get(f"RI_{i + 1}", 0)
    out[disso] = d[1:]
    return out - trace.alignment_offset


def fit_one_site(experiment, metadata, n_starts=8, seed=2918):
    """Fit the literature's one-site model with per-curve Rmax and optional RI."""
    if not metadata.get("molar_concentrations_verified"):
        raise ValueError("The reference fit requires verified molar concentrations")
    traces, meta = prepare(experiment, metadata)
    spec = _parameters(traces, meta)
    q0, lo, hi = (_encode(spec, field) for field in ("start", "low", "high"))
    target = np.concatenate([t.response[t.indices] for t in traces])
    times = [t.time[t.indices] for t in traces]

    def residual(q):
        params = _decode(spec, q)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            fitted = np.concatenate(
                [
                    predict(t, meta, params, i, time=time)
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
                max_nfev=700,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
        runs.append(fit)
    successful = [r for r in runs if r.success]
    if not successful:
        raise RuntimeError("One-site fit: no starting point converged")
    best = min(successful, key=lambda r: np.dot(r.fun, r.fun))
    params = _decode(spec, best.x)
    r = residual(best.x)
    rss = float(r @ r)
    return dict(
        model="one_site",
        local_rmax=True,
        params=params,
        n=len(r),
        k=len(params),
        rss=rss,
        rmse=float(np.sqrt(rss / len(r))),
        starts_converged=len(successful),
        starts_tried=len(runs),
        seed=seed,
        traces=traces,
        meta=meta,
    )


def plot_fit(fit):
    best = fit
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
        pred = predict(trace, meta, best["params"], i, time=time) + trace.background
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
            trace, meta, best["params"], i, time=t
        )
        axes[1].plot(t, r, ".", color=f"C{i}", markersize=2, label=f"Curve {i + 1}")
    for axis in axes:
        for boundary in [meta["association_start"], meta["association_end"]]:
            axis.axvspan(boundary - 3, boundary + 3, color="grey", alpha=0.12)
        axis.grid(alpha=0.15)
    axes[0].set(
        title="One-site Langmuir fit, local Rmax",
        ylabel=(
            "Association-aligned response"
            if meta.get("association_alignment")
            else "Measured response"
        ),
    )
    axes[0].legend(frameon=False)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Original exported time (s)", ylabel="Data − fit")
    return fig


def parameter_table(fit):
    """Point estimates in the units used by the published reports."""
    values = fit["params"]
    rows = [
        {
            "Parameter": "kon",
            "Unit": "M⁻¹ s⁻¹",
            "Estimate": values["k_off"] / values["Kd"] * 1e6,
        },
        {"Parameter": "koff", "Unit": "s⁻¹", "Estimate": values["k_off"]},
        {"Parameter": "Kd", "Unit": "nM", "Estimate": values["Kd"] * 1e3},
    ]
    rows.extend(
        {"Parameter": name, "Unit": "RU", "Estimate": value}
        for name, value in values.items()
        if name.startswith(("Rmax_", "RI_"))
    )
    return pd.DataFrame(rows)


def literature_comparison(fit):
    """Compare the one-site rates/affinity with the identified author report row."""
    values = parameter_table(fit).iloc[:3]
    reference = fit["meta"]["literature_reference"]["parameters_molar"]
    published = [reference["kon"], reference["koff"], reference["Kd"] * 1e9]
    table = values.rename(columns={"Estimate": "This notebook"}).copy()
    table["Published"] = published
    table["Difference (%)"] = 100 * (table["This notebook"] / table["Published"] - 1)
    return table[["Parameter", "Unit", "Published", "This notebook", "Difference (%)"]]


def save_report(fit, path):
    """Save only this one-site fit and its published comparison."""
    import scipy

    document = dict(
        model=fit["model"],
        local_rmax=fit["local_rmax"],
        metadata=fit["meta"],
        parameters=parameter_table(fit).to_dict(orient="records"),
        literature_comparison=literature_comparison(fit).to_dict(orient="records"),
        observations=fit["n"],
        fitted_parameters=fit["k"],
        rmse=fit["rmse"],
        settings=dict(
            seed=fit["seed"],
            starts_tried=fit["starts_tried"],
            starts_converged=fit["starts_converged"],
            transition_guard_seconds=3,
            minimum_spacing_seconds=1,
            scipy=scipy.__version__,
            numpy=np.__version__,
        ),
        scope="Same source dataset and one-site/local-Rmax model; three selected curves versus five in the published report. Not an exact reproduction.",
    )
    Path(path).write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
