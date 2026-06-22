#!/usr/bin/env python3
"""Rewrite reproduce_figure_04.ipynb so each panel cell delegates to the recheck builder
(build_4ABC / build_4D / build_4EFG / build_4H / build_verify_fig04) and displays the
reproduced PNG. Keeps the markdown narrative; clears stale outputs. Run, then
`jupyter nbconvert --execute --inplace`.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "reproduce_figure_04.ipynb"

DELEGATE = {
    "4ABC": ('## Panels A–C — promoter ISM (4 rows: Shorkie LM / Shorkie ISM / Random_Init / Reference DB)\n',
             'Figure_4ABC_reproduced.png', 'build_4ABC.py'),
    "4D":   ('## Panel D — splicing motifs: database vs Shorkie ISM reconstruction\n',
             'Figure_4D_reproduced.png', 'build_4D.py'),
    "4EFG": ('## Panels E–G — splicing ISM + gene model + splice annotations\n',
             'Figure_4EFG_reproduced.png', 'build_4EFG.py'),
    "4H":   ('## Panel H — TF-MoDISco motifs: database vs Shorkie reconstruction\n',
             'Figure_4H_reproduced.png', 'build_4H.py'),
}


def cell_src(c):
    return "".join(c.get("source", []))


def make_code(builder, png):
    return (f"import subprocess, sys\n"
            f"from pathlib import Path\n"
            f"from IPython.display import Image, display\n"
            f"subprocess.run([sys.executable, str(Path('recheck')/'{builder}')], check=True)\n"
            f"display(Image('reproduced/{png}'))\n")


def make_verify():
    return ("import subprocess, sys, pandas as pd\n"
            "from pathlib import Path\n"
            "subprocess.run([sys.executable, str(Path('recheck')/'build_verify_fig04.py')], check=True)\n"
            "df = pd.read_csv('reproduced/verify_fig04.csv')\n"
            "print(df.to_string(index=False))\n"
            "print('\\nPASS', int((df.verdict=='PASS').sum()), '/', len(df))\n")


def main():
    nb = json.loads(NB.read_text())
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = cell_src(c)
        new = None
        if "PROM" in s and "ism_viz" in s:
            b = DELEGATE["4ABC"]; new = make_code(b[2], b[1])
        elif "Canonical S. cerevisiae" in s or ('GTATGT' in s and 'donor' in s):
            b = DELEGATE["4D"]; new = make_code(b[2], b[1])
        elif "SS=[" in s or "Figure_4EFG" in s:
            b = DELEGATE["4EFG"]; new = make_code(b[2], b[1])
        elif "modisco" in s and ("Figure_4H" in s or "FIG4H" in s):
            b = DELEGATE["4H"]; new = make_code(b[2], b[1])
        elif "write_verdicts" in s or "verify_fig04" in s:
            new = make_verify()
        if new is not None:
            c["source"] = new.splitlines(keepends=True)
            c["outputs"] = []
            c["execution_count"] = None
    # clear all remaining outputs/exec counts
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c.setdefault("outputs", [])
            c["outputs"] = []
            c["execution_count"] = None
    NB.write_text(json.dumps(nb, indent=1))
    print("rewrote", NB)


if __name__ == "__main__":
    main()
