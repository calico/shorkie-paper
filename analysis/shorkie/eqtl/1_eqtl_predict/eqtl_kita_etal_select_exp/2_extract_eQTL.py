#!/usr/bin/env python3
import os
import pandas as pd
import pysam

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
root_dir = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL_kita_etal"
raw_eqtl       = os.path.join(root_dir, "pnas.1717421114.sd01.fix.txt")
clean_dir      = os.path.join(root_dir, "fix")
cleaned_eqtl   = os.path.join(clean_dir, "pnas.1717421114.sd01_cleaned.select.txt")

gvcf_file   = "/home/kchao10/scr4_ssalzbe1/khchao/Yeast_ML/data/eQTL/1011Matrix.gvcf"
selected_dir = os.path.join(clean_dir, "selected_eQTL")

local_threshold = 8000  # ±8 kb around TSS → CIS

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(selected_dir, exist_ok=True)

# ── STEP 1: CLEAN & ANNOTATE eQTLs ─────────────────────────────────────────────
# 1a) Read raw eQTLs
eqtl_df = pd.read_csv(raw_eqtl, sep="\t", dtype=str, low_memory=False)

# 1b) Cast numeric & compute distance
eqtl_df["ChrPos"] = eqtl_df["position"].astype(int)
eqtl_df["TSS"]    = eqtl_df["TSS"].astype(int)
eqtl_df["distance_to_TSS"] = (eqtl_df["ChrPos"] - eqtl_df["TSS"]).abs()
eqtl_df["is_local"]        = eqtl_df["distance_to_TSS"] <= local_threshold

# select eqtl rows if 'locationType' is "promoter", "UTR3", "UTR5", "ORF"
eqtl_df = eqtl_df[eqtl_df["locationType"].isin([
    "Promoter", "UTR3", "UTR5", "ORF"
])]

print(f"Loaded eQTLs: {len(eqtl_df)} rows")
# 1c) Harmonize chromosome column for merging
#     Original eQTLs use Roman numerals (I–XVI); GVCF uses "chromosome1"–"chromosome16"
eqtl_df.rename(columns={"chr": "Chr"}, inplace=True)

# Map Roman to Arabic and prepend "chromosome"
roman_to_num = {
    "I":   "1",  "II":  "2",  "III":  "3",  "IV":  "4",
    "V":   "5",  "VI":  "6",  "VII":  "7",  "VIII":"8",
    "IX":  "9",  "X":   "10", "XI":  "11", "XII": "12",
    "XIII":"13", "XIV": "14", "XV":  "15", "XVI": "16"
}
eqtl_df["Chr"] = eqtl_df["Chr"].map(
    lambda x: f"chromosome{roman_to_num.get(x, x)}"
)

# 1d) Write cleaned TSV
eqtl_df.to_csv(cleaned_eqtl, sep="\t", index=False)
print(f"Wrote cleaned eQTLs → {cleaned_eqtl}")

# ── STEP 2: LOAD GVCF INTO A DATAFRAME ────────────────────────────────────────
def read_gvcf_to_df(path):
    records = []
    with pysam.VariantFile(path) as g:
        print(f"Reading GVCF: {path}")
        for rec in g:
            records.append({
                "Chr":       rec.chrom,
                "ChrPos":    rec.pos,  # 1-based
                "Reference": rec.ref,
                "Alternate": ",".join(rec.alts) if rec.alts else None,
                "Quality":   rec.qual
            })
    df = pd.DataFrame(records)
    df["ChrPos"] = df["ChrPos"].astype("int64")
    return df

gvcf_df = read_gvcf_to_df(gvcf_file)

# ── STEP 3: INTERSECT & SPLIT CIS / TRANS ─────────────────────────────────────
def write_df_to_vcf(df, out_vcf, template_vcf):
    with pysam.VariantFile(template_vcf) as tpl:
        header = tpl.header
        valid_contigs = set(header.contigs)
    df = df[df["Chr"].isin(valid_contigs)]
    with pysam.VariantFile(out_vcf, "w", header=header) as out:
        for _, row in df.iterrows():
            if pd.isna(row["Reference"]) or pd.isna(row["Alternate"]):
                continue
            out_rec = out.new_record(
                contig  = row["Chr"],
                start   = int(row["ChrPos"]) - 1,
                stop    = int(row["ChrPos"]),
                alleles = (row["Reference"], row["Alternate"]),
                qual    = row.get("Quality", None)
            )
            out.write(out_rec)
    print(f"Wrote VCF → {out_vcf}")

for dist_type, mask in [("CIS",  eqtl_df["is_local"]),
                        ("TRANS", ~eqtl_df["is_local"])]:
    sub = eqtl_df[mask].copy()
    sub["ChrPos"] = sub["ChrPos"].astype("int64")

    # TSV paths
    int_tsv  = os.path.join(selected_dir, f"intersected_{dist_type}.tsv")
    only_tsv = os.path.join(selected_dir, f"only_eQTL_{dist_type}.tsv")

    # Merge on Chr & ChrPos
    inter = pd.merge(sub, gvcf_df, how="inner", on=["Chr", "ChrPos"])
    inter.to_csv(int_tsv, sep="\t", index=False)
    print(f"{dist_type}: intersected rows = {len(inter)} → {int_tsv}")

    only = sub.merge(gvcf_df, on=["Chr", "ChrPos"], how="left", indicator=True)
    only = only[only["_merge"] == "left_only"].drop(columns="_merge")
    only.to_csv(only_tsv, sep="\t", index=False)
    print(f"{dist_type}: only-eQTL rows   = {len(only)} → {only_tsv}")

    # VCF paths
    int_vcf  = os.path.join(selected_dir, f"intersected_{dist_type}.vcf")
    only_vcf = os.path.join(selected_dir, f"only_eQTL_{dist_type}.vcf")

    write_df_to_vcf(inter, int_vcf, gvcf_file)
    write_df_to_vcf(only, only_vcf, gvcf_file)

print("Done.")
