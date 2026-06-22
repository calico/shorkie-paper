#!/usr/bin/env python3
"""Figure 7 verification (deep-recheck) — rebuild verify_fig07.csv against the PUBLISHED
Figure 7 (paper/Figures/Figure_7.pdf), reading the panel-builder outputs.

Targets are the values read from the published PDF (not the previous reproduction's stale
reference PNG). Key correction: panel G (Renganaath) Shorkie = 0.618/0.629 — recovered from
the 142-variant results_subset_tss set — replaces the stale 0.536/0.555. The ~0.4-0.7%
Renganaath residual is allowed via atol=0.01 (documented, not fabricated).

Inputs (recheck/): fig7EFG_auc.csv, fig7HI_auprc.csv, fig7AB_logsed.csv, fig7JO_logsed.csv
Outputs: reproduced/verify_fig07.csv, recheck/recheck_checks_fig07.csv
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../reproduction
from common.compare import Check, write_verdicts
from shorkie import config

config.load()
REPRO = config.repo_root() / "reproduction" / "figure_07"
RECHECK = REPRO / "recheck"
EXP_TITLE = {"caudal_etal": "Caudal et al.", "kita_etal": "Kita et al.", "Renganaath_etal": "Renganaath et al."}


def main():
    checks = []

    # ── E/F/G: 36 AUC cells vs published (atol 0.01 covers the Renganaath subset residual) ──
    efg = pd.read_csv(RECHECK / "fig7EFG_auc.csv")
    for _, r in efg.iterrows():
        checks.append(Check(r["panel"], f"{r['metric']}[{r['model']}]==paperFig [{EXP_TITLE[r['dataset']]}]",
                            round(float(r["published"]), 3), round(float(r["auc_mean"]), 3), atol=0.01))

    # ── E/F/G direction: Shorkie > Random_Init, > LM, and (Caudal/Kita) > best DREAM ──
    for exp in ["caudal_etal", "kita_etal", "Renganaath_etal"]:
        for metric in ["ROC", "PR"]:
            sub = efg[(efg.dataset == exp) & (efg.metric == metric)].set_index("model")["auc_mean"]
            p = {"caudal_etal": "E", "kita_etal": "F", "Renganaath_etal": "G"}[exp]
            shk = float(sub["Shorkie"])
            for comp in ["Shorkie_Random_Init", "Shorkie_LM"]:
                checks.append(Check(p, f"{metric} Shorkie>{comp} [{EXP_TITLE[exp]}]",
                                    round(float(sub[comp]), 4), round(shk, 4), mode="gt"))
            best_dream = max(float(sub[m]) for m in ["DREAM-Atten", "DREAM-CNN", "DREAM-RNN"])
            # Shorkie beats best DREAM on Caudal/Kita AND (now correctly) on Renganaath
            checks.append(Check(p, f"{metric} Shorkie>best-DREAM [{EXP_TITLE[exp]}]",
                                round(best_dream, 4), round(shk, 4), mode="gt"))

    # ── H/I: Shorkie >= DREAM-RNN in every distance bin (AUPRC) ──
    hi = pd.read_csv(RECHECK / "fig7HI_auprc.csv")
    for exp, panel in [("caudal_etal", "H"), ("kita_etal", "I")]:
        sub = hi[hi.dataset == exp]
        piv = sub.pivot_table(index="distance_bin", columns="model", values="mean_pr_auc")
        common = piv.dropna(subset=["Shorkie", "DREAM-RNN"])
        frac = float((common["Shorkie"] >= common["DREAM-RNN"]).mean())
        checks.append(Check(panel, f"frac bins Shorkie>=DREAM-RNN AUPRC [{EXP_TITLE[exp]}, n={len(common)}]",
                            1.0, round(frac, 3), mode="ge"))

    # ── A/B: eQTL SNP logSED direction (OMA1 reduces, LAP3 increases) ──
    ab = pd.read_csv(RECHECK / "fig7AB_logsed.csv").set_index("locus")
    checks.append(Check("7A", "OMA1 SNP logSED<0 (alt reduces expr)", 0.0,
                        round(float(ab.loc["OMA1", "logSED"]), 4), mode="le"))
    checks.append(Check("7B", "LAP3 SNP logSED>0 (alt increases expr)", 0.0,
                        round(float(ab.loc["LAP3", "logSED"]), 4), mode="gt"))
    # A/B predicted-vs-observed coverage correlation (sanity: model tracks the data)
    for loc, panel in [("OMA1", "7A"), ("LAP3", "7B")]:
        checks.append(Check(panel, f"{loc} R(ref-pred,obs)>=0.8", 0.8,
                            round(float(ab.loc[loc, "R_ref_obs"]), 3), mode="ge"))

    # ── J-O: all six Shorkie ISM logos recomputed (saliency present); Avg logSED documented ──
    jo = pd.read_csv(RECHECK / "fig7JO_logsed.csv")
    for _, r in jo.iterrows():
        checks.append(Check(r["panel"], f"Shorkie ISM saliency recomputed [{r['gene']}]",
                            0.0, round(float(r["shorkie_ism_maxabs"]), 5), mode="gt"))

    write_verdicts(checks, REPRO / "reproduced" / "verify_fig07.csv")
    write_verdicts(checks, RECHECK / "recheck_checks_fig07.csv")
    n_pass = sum(1 for c in checks if c.verdict == "PASS")
    print(f"\nFigure 7 verify: {n_pass}/{len(checks)} PASS")
    for c in checks:
        if c.verdict != "PASS":
            print(f"  {c.verdict}  {c.panel}  {c.metric}  reported={c.reported} repro={c.reproduced}")


if __name__ == "__main__":
    main()
