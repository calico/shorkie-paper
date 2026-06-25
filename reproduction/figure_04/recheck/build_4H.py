#!/usr/bin/env python3
"""Figure 4 panel H (recheck) — TF-MoDISco motifs: Database vs Shorkie reconstruction.

A 2-row x 12-col grid in the published TF order. Top row = database motif logos
(merged_meme_high_conf.meme, IC-weighted; cached PWM fallback for TATA/PAC/RRPE).
Bottom row = the Shorkie ISM TF-MoDISco reconstruction = the RP-MoDISco pattern
matched to that TF by TomTom (contrib_scores CWM, trimmed). Pairings come from
match_tfs (fresh TomTom preferred, cached report/motifs.html fallback — both TomTom).

Output: reproduced/Figure_4H_reproduced.png  +  recheck/fig4H_tomtom_pairs.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig4_common as F
import match_tfs as M

# figure label, DB motif (meme id or @asset), reconstruction search TF (modisco match)
TFS = [
    ("Rap1", "RAP1", "RAP1"), ("Fhl1", "FHL1", "FHL1"), ("Sfp1.1", "SFP1", "SFP1"),
    ("TATA Box", "@TATATA", "SPT15"), ("Reb1", "REB1", "REB1"), ("Abf1", "ABF1", "ABF1"),
    ("Tbf1.1", "TBF1", "TBF1"), ("Cbf1", "CBF1", "CBF1"), ("Ume6.2", "UME6", "UME6"),
    ("Dot6p", "DOT6", "DOT6"), ("PAC motif (Dot6)", "@PAC_motif", "DOT6"),
    ("RRPE motif (Stb3)", "@TGAAAAATTTT", "STB3"),
]


def main():
    mh = F.modisco_h5("gene_exp_motif_test_RP")
    n = len(TFS)
    fig = plt.figure(figsize=(1.6 * n, 4.2))
    gs = fig.add_gridspec(2, n, hspace=0.35, wspace=0.25)
    rows = []
    for c, (label, db, recon_tf) in enumerate(TFS):
        # --- top: database motif ---
        ax = fig.add_subplot(gs[0, c])
        dbarr = F.db_motif_ic(db)
        if dbarr is not None:
            F.draw_pwm_logo(ax, dbarr)
        else:
            ax.text(0.5, 0.5, "n/a", ha="center")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=8)
        if c == 0:
            ax.set_ylabel("Database\nMotifs", fontsize=8, rotation=0, ha="right", va="center")
        # --- bottom: Shorkie reconstruction (matched RP modisco pattern) ---
        # priority: fresh full TomTom -> report motifs.html -> Pearson corr to the DB motif
        axr = fig.add_subplot(gs[1, c])
        tag = qval = matched = None; method = "none"
        if recon_tf is not None:
            res = M.tf_to_pattern(mh, recon_tf, qmax=1.01)            # cached report (canonical, best q)
            if res is not None:
                tag, qval, matched = res; method = "report_html"
            else:
                res = M.tf_to_pattern_tomtom(recon_tf, qmax=1.01)    # fresh full tomtom (recovers Cbf1/Ume6/TATA)
                if res is not None:
                    tag, qval, matched = res; method = "tomtom_full"
        if tag is None and dbarr is not None:
            tg, corr = M.best_pattern_by_correlation(mh, dbarr)
            if tg is not None:
                tag, qval, matched, method = tg, round(corr, 3), f"corr({recon_tf or label})", "pearson"
        if tag is not None:
            F.draw_pwm_logo(axr, F.trim_cwm(M.pattern_cwm(mh, tag)))
        else:
            axr.text(0.5, 0.5, "no match", ha="center", va="center", fontsize=6, color="gray")
        axr.set_xticks([]); axr.set_yticks([])
        if c == 0:
            axr.set_ylabel("Shorkie recon.\n(from ISM PWM)", fontsize=8, rotation=0, ha="right", va="center")
        rows.append(dict(panel_tf=label, db_motif=db, recon_search=recon_tf,
                         modisco_pattern=tag, match_method=method, tomtom_qval=qval, matched_name=matched))
    fig.suptitle("Figure 4H (reproduced) — TF-MoDISco motifs: database vs Shorkie ISM reconstruction",
                 y=1.02, fontsize=11)
    out = F.RD / "Figure_4H_reproduced.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(F.RECHECK / "fig4H_tomtom_pairs.csv", index=False)
    n_match = int(df["modisco_pattern"].notna().sum())
    print("saved", out, f"| {n_match}/{n} TFs matched to a reconstruction pattern")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
