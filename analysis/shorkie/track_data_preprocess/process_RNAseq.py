import pandas as pd
import sys
import os
import subprocess
import configargparse
import math
import numpy as np

# sys.path += ["/group/singlecell/scpipe/scutil"]
sys.path += ["/home/mo/shared_repos/basenji/basenji"]
import slurm

def generate_basic_bigwig(args, merged_bam, sample_path, sample_name):
    genomecov_cmd = " ".join([
        f"bedtools genomecov -bg -pc -ibam",
        merged_bam,
        ">",
        os.path.join(sample_path, f"{sample_name}.bedgraph")
    ])
    sort_cmd = " ".join([
        f"sort -k 1,1 -k2,2n",
        os.path.join(sample_path, f"{sample_name}.bedgraph"),
        ">",
        os.path.join(sample_path, f"{sample_name}_sort.bedgraph")
    ])
    bigwig_cmd = " ".join([
        f"bedGraphToBigWig",
        os.path.join(sample_path, f"{sample_name}_sort.bedgraph"),
        args.chrsizes,
        os.path.join(sample_path, f"{sample_name}_basic.bw")
    ])
    subprocess.run(
        ";".join([genomecov_cmd, sort_cmd, bigwig_cmd]),
        shell = True
    )
    
    return

def generate_bamcov_bigwig(merged_bam, sample_path, sample_name):
    bamcov_out = f"{os.path.join(sample_path, sample_name)}_bamcov.bw"
    
    if not os.path.exists(bamcov_out):
        bamcov_cmd = " ".join([
            "python /home/mo/shared_repos/basenji/bin/bam_cov.py -a",
            merged_bam,
            bamcov_out
        ])
        subprocess.run(
            bamcov_cmd,
            shell = True
        )
    else:
        print('bam coverage already generated')

    return

# Run alignments, qc, and concatonate reads from replicate experiments
def process_samples(sample_sheet, args):
    sample_sheet['sample_dirs'] = sample_sheet['sample_dirs'].apply(lambda x: x.split(";"))
    
    for i in range(sample_sheet.shape[0]):
        batch_dir = sample_sheet.iloc[i]['batch_dir']
        sample_name = sample_sheet.iloc[i]['sample_name']
        sample_path = os.path.join(args.savedir, sample_name)
        merged_bam = os.path.join(sample_path, f"{sample_name}.bam")
        sample_dirs = sample_sheet.iloc[i]['sample_dirs']
        print(f"processing {sample_name}")
        
        # 1. run aligment for each replicate
        if not os.path.exists(sample_path):
            subprocess.run(
                f"mkdir {sample_path}",
                shell = True
            )
        
        # aligned = [f for f in os.listdir(sample_path) if f.endswith("_rmdup.bam")]
        # if len(aligned) < len(sample_dirs):
            for sample_dir in sample_dirs:
                r1 = os.path.join(batch_dir, sample_dir, f"{sample_dir}_trimmed_R1.fastq.gz")
                r2 = os.path.join(batch_dir, sample_dir, f"{sample_dir}_trimmed_R2.fastq.gz")          
                shell_cmd = " ".join([
                    "bash /home/mo/mmagzoub/yeast_sequence_models/bioinformatics/rnaseq2bam.sh",
                    f"-i {sample_dir}",
                    f"-r {args.reference}",
                    f"-s {args.star_reference}",
                    f"-d {sample_path}",
                    f"-1 {r1}",
                    f"-2 {r2}"
                ])
                subprocess.run(
                    shell_cmd, 
                    shell = True
                )
        else:
            print('already aligned')
            
        # 2. merge replicates
        if not os.path.exists(merged_bam):
            bam_files = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith("_rmdup.bam")]
            if len(bam_files) > 1:
                merge_cmd = f"samtools merge {merged_bam} {' '.join(bam_files)}"
                subprocess.run(
                    merge_cmd, 
                    shell = True,
                )
            elif len(bam_files) == 1:
                subprocess.run(
                    f"mv {bam_files[0]} {merged_bam}", 
                    shell = True,
                )
        else:
            print('replicates already merged')

        # 3. create bigwigs if applicable
        if args.basic and args.chrsizes:
            generate_basic_bigwig(args, merged_bam, sample_path, sample_name)
        if args.bamcov:
            generate_bamcov_bigwig(merged_bam, sample_path, sample_name)
    
    return

def make_parser():
    parser = configargparse.ArgParser(
        description='process RNAseq data from ministat induction experiments.',
    )

    parser.add_argument('--savedir', type=str, required = True)
    parser.add_argument('--sample_sheet', type = str, required = True)
    parser.add_argument('--reference', type=str, required = True)
    parser.add_argument('--star_reference', type=str, required = True)
    parser.add_argument('--n_batches', type=int, required=False, default = 0)
    parser.add_argument('--chrsizes', type=str, required = False, default = None)
    parser.add_argument('--bamcov', default = False, action='store_true')
    parser.add_argument('--basic', default = False, action='store_true')
  
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    sample_sheet = pd.read_csv(args.sample_sheet)
    
    if args.n_batches > 0:
        n_samples = sample_sheet.shape[0] 
        batch_size = math.floor(n_samples / args.n_batches)
        n_extra_samples = n_samples % batch_size
        
        # split samples into batches of equal size
        batch_ids = [[n] * math.floor(batch_size) for n in range(1, args.n_batches)]
        batch_ids.append([args.n_batches] * (batch_size + n_extra_samples))
        batch_ids = list(np.concatenate(batch_ids).flat)
        sample_sheet["batch_id"] = batch_ids
        
        jobs = []
        # re-set up run on batch
        for batch in range(1, args.n_batches + 1):
            batch_sheet = sample_sheet.loc[sample_sheet["batch_id"] == batch]
            batch_sheet.to_csv(os.path.join(args.savedir, f"{batch}_sheet.csv"), index = False)
            batch_cmd = [
                "python /home/mo/mmagzoub/yeast_sequence_models/bioinformatics/process_RNAseq.py",
                f"--sample_sheet {os.path.join(args.savedir, f'{batch}_sheet.csv')}",
                f"--savedir {args.savedir}",
                f"--reference {args.reference}",
                f"--star_reference {args.star_reference}"
            ]
            if args.chrsizes:
                batch_cmd += [f"--chrsizes {args.chrsizes}"]
            if args.bamcov:
                batch_cmd += ["--bamcov"]
            if args.basic:
                batch_cmd += ["--basic"]
            batch_cmd = " ".join(batch_cmd)

            job = slurm.Job(
                batch_cmd,
                name = f"yeast_rna_{batch}",
                out_file = os.path.join(args.savedir, f"yeast_rna_{batch}.out"),
                err_file = os.path.join(args.savedir, f"yeast_rna_{batch}.err"),
                queue = "geforce",
                gpu = 1,
                cpu = 4,
                mem = 32 * 1000,
                time = '72:00:00'
            )
            jobs.append(job)
            
        # run batches on slurm
        print("Launching slurm jobs")
        slurm.multi_run(jobs, verbose=True)

    else:
        
        process_samples(sample_sheet, args)

################################################
# __main__
################################################
        
# # example use
# python /home/mo/mmagzoub/yeast_sequence_models/bioinformatics/process_RNAseq.py \
# --sample_sheet ./sample_sheet.csv \
# --n_batches 10 \
# --savedir ./ \
# --reference /group/idea/yeast_references/S288C_R64-3-1.fasta \
# --star_reference /group/idea/yeast_references/STAR \
# --chrsizes /group/idea/basenji_yeast/data/references/chrom.sizes \
# --bamcov

if __name__ == '__main__':
    main()