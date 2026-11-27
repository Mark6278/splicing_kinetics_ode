import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from splicing_ode import AlternativeSplicingModel

try:
    from gnn_kinetics import fit_events_pytorch
    HAS_GNN = True
except ImportError:
    HAS_GNN = False


def _split_psi(val) -> list[float]:
    if pd.isna(val):
        return []
    return [float(x.strip()) for x in str(val).split(",") if x.strip() not in ("", "NA", "nan", "None")]


def _fit_event_scipy(ev: dict) -> dict:
    model = AlternativeSplicingModel()
    ev["sf"] = [model.infer_sf_closed_form(float(p)) for p in ev["psi"]]
    ev["k_deg_inc"] = 0.2
    ev["k_deg_skip"] = 0.2
    return ev


def fit_all_events(events: list[dict], workers: int = 2, backend: str = "auto") -> tuple[list[dict], str]:
    max_samples = max(len(e["psi"]) for e in events)
    use_gnn = (backend in ("gnn", "ai", "pytorch")) or (backend == "auto" and HAS_GNN and max_samples >= 6)
    if use_gnn and HAS_GNN:
        max_len = max(len(e["psi"]) for e in events)
        psi_mat = np.zeros((len(events), max_len), dtype=float)
        mask_mat = np.zeros((len(events), max_len), dtype=float)
        init_sf = np.ones((len(events), max_len), dtype=float)
        model = AlternativeSplicingModel()
        for i, ev in enumerate(events):
            n = len(ev["psi"])
            psi_mat[i, :n] = ev["psi"]
            mask_mat[i, :n] = 1.0
            init_sf[i, :n] = [model.infer_sf_closed_form(float(p)) for p in ev["psi"]]
        res = fit_events_pytorch(psi_mat, mask_mat, init_sf, event_ids=[e["event_id"] for e in events])
        for i, ev in enumerate(events):
            n = len(ev["psi"])
            ev["sf"] = [float(x) for x in res["sf"][i, :n]]
            ev["k_deg_inc"] = float(res["k_deg_inc"][i])
            ev["k_deg_skip"] = float(res["k_deg_skip"][i])
        return events, f"pytorch_gnn ({res['device']})"

    w = min(max(1, workers), len(events))
    fitted = list(ProcessPoolExecutor(max_workers=w).map(_fit_event_scipy, events)) if w > 1 else [_fit_event_scipy(e) for e in events]
    return fitted, "scipy_analytical"
    

def load_events(df: pd.DataFrame, labels: list[str] | None, fdr_max: float) -> list[dict]:
    events = []
    if "IncLevel1" in df.columns:
        df = df.dropna(subset=["IncLevel1"]).copy()
        has_g2 = "IncLevel2" in df.columns and df["IncLevel2"].dropna().apply(lambda v: len(_split_psi(v)) > 0).any()
        if has_g2 and "FDR" in df.columns and fdr_max < 1.0:
            sig = df[pd.to_numeric(df["FDR"], errors="coerce") <= fdr_max]
            if not sig.empty:
                df = sig
        for _, r in df.iterrows():
            g1 = [v.strip() for v in str(r["IncLevel1"]).split(",") if v.strip()]
            g2 = [v.strip() for v in str(r["IncLevel2"]).split(",") if v.strip()] if has_g2 else []
            psi, names, groups, idx = [], [], [], 0
            for grp, raw in [("Control" if has_g2 else "Cohort", g1), ("Treatment", g2)]:
                for j, v in enumerate(raw):
                    if v not in ("", "NA", "nan", "None"):
                        psi.append(float(v))
                        names.append(labels[idx] if (labels and idx < len(labels)) else f"{grp[:4]}_{j+1}")
                        groups.append(grp)
                    idx += 1
            if psi:
                arr = np.clip(np.array(psi, dtype=float), 0.01, 0.99)
                t = str(r.get("geneSymbol", r.get("GeneID", "Target"))).strip('"')
                events.append({
                    "target": t, "event_id": f"{t}:exon_{r.get('exonStart_0base','')}-{r.get('exonEnd','')}",
                    "samples": names, "groups": groups, "psi": arr, "mean_psi": float(np.mean(arr)),
                    "dpsi": float(r["IncLevelDifference"]) if ("IncLevelDifference" in r and pd.notna(r["IncLevelDifference"])) else None,
                    "fdr": float(r["FDR"]) if ("FDR" in r and pd.notna(r["FDR"])) else None,
                })
    else:
        scols = [c for c in df.columns if c not in {"event_id", "gene", "geneSymbol", "GeneID", "chr", "strand"}]
        for i, r in df.iterrows():
            s = pd.to_numeric(r[scols], errors="coerce").dropna()
            if not s.empty:
                arr = np.clip(s.values.astype(float), 0.01, 0.99)
                t = str(r.get("gene", r.get("geneSymbol", "Target"))).strip('"')
                events.append({
                    "target": t, "event_id": str(r.get("event_id", f"{t}_SE_{i+1}")),
                    "samples": list(s.index), "groups": ["Cohort"] * len(s), "psi": arr,
                    "mean_psi": float(np.mean(arr)), "dpsi": None, "fdr": None,
                })
    if not events:
        raise ValueError("No valid numeric PSI events found.")
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--samples", default="")
    ap.add_argument("--fdr", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--backend", default="auto", choices=["auto", "scipy", "gnn", "ai"])
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    labels = [s.strip() for s in args.samples.split(",") if s.strip()] or None
    events = load_events(pd.read_csv(args.input), labels, args.fdr)
    fitted, used_backend = fit_all_events(events, workers=args.workers, backend=args.backend)
        
    rows = [
        {"target": e["target"], "event_id": e["event_id"], "group": g, "sample_or_cohort": s,
         "observed_psi": round(float(p), 4), "inferred_sf_activity": round(float(sf), 4),
         "inferred_k_deg_inc": round(e["k_deg_inc"], 4), "inferred_k_deg_skip": round(e["k_deg_skip"], 4),
         "delta_psi": round(e["dpsi"], 4) if e["dpsi"] is not None else np.nan,
         "fdr": round(e["fdr"], 6) if e["fdr"] is not None else np.nan}
        for e in fitted for g, s, p, sf in zip(e["groups"], e["samples"], e["psi"], e["sf"])
    ]
    target = next(iter(fitted))["target"]
    pd.DataFrame(rows).to_csv(out / f"{target}_sf_inference.csv", index=False)

    max_n = max(len(e["psi"]) for e in fitted)
    comp = [e for e in fitted if len(e["psi"]) == max_n]
    sig = [e for e in comp if e["fdr"] is not None]
    dyn = [e for e in comp if 0.05 < e["mean_psi"] < 0.95]
    feat = min(sig, key=lambda e: (e["fdr"], -abs(e["dpsi"] or 0.0))) if sig else (min(dyn, key=lambda e: abs(e["mean_psi"] - 0.5)) if dyn else next(iter(comp)))

    model = AlternativeSplicingModel(k_deg_inc=feat["k_deg_inc"], k_deg_skip=feat["k_deg_skip"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.9), constrained_layout=True)
    for lbl, sf in [("Wild-Type (SF=1.0)", 1.0), ("Partial Knockdown (SF=0.5)", 0.5), ("Severe Perturbation (SF=0.2)", 0.2)]:
        sim = model.simulate((0.0, 120.0), 250, sf)
        line, = ax1.plot(sim["time"], sim["PSI"], lw=2.2, label=f"{lbl} (PSI*={sim['PSI_steady_state']:.2f})")
        ax1.axhline(sim["PSI_steady_state"], color=line.get_color(), ls="--", lw=1.0, alpha=0.55)
    ax1.set(title="A. Co-Transcriptional Splicing Kinetics (ODE)", xlabel="Time (min)", ylabel="Percent Spliced In (PSI)", ylim=(0, 1.05))
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False, fontsize=9)

    x = np.arange(len(feat["samples"]))
    ax2.bar(x, feat["psi"], width=0.45, color=["#2b8cbe" if g != "Treatment" else "#6baed6" for g in feat["groups"]], alpha=0.88, label="Observed PSI")
    ax2.set_ylabel("Observed PSI", color="#2b8cbe", fontsize=10)
    ax2.set_ylim(0, 1.12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(feat["samples"], rotation=25, ha="right", fontsize=9)
    ax2t = ax2.twinx()
    ax2t.plot(x, feat["sf"], color="#d95f0e", marker="o", lw=2, markersize=6, label="Inferred SF Activity")
    ax2t.set_ylabel("Inferred Splicing Factor Activity", color="#d95f0e", fontsize=10)
    ax2t.set_ylim(0, max(feat["sf"]) * 1.25)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=8.5)
    raw_ev = feat["event_id"].split(";")[-1] if ";" in feat["event_id"] else feat["event_id"]
    short_ev = f"{target}:{raw_ev}" if not raw_ev.startswith(target) else raw_ev
    if len(short_ev) > 42:
        short_ev = short_ev[:39] + "..."
    title_b = f"B. Observed PSI & Inferred SF Activity ({short_ev})" + (f"\nΔPSI = {feat['dpsi']:+.3f}  |  FDR = {feat['fdr']:.2e}" if feat["dpsi"] is not None and feat["fdr"] is not None else "")
    ax2.set_title(title_b, fontsize=9.5, pad=10)

    fig.savefig(out / f"{target}_splicing_dynamics.png", dpi=300)
    plt.close(fig)
    print(f"Fitted {len(fitted)} event(s) for {target} [backend={used_backend}] -> {out / f'{target}_sf_inference.csv'}")
    

if __name__ == "__main__":
    main()
