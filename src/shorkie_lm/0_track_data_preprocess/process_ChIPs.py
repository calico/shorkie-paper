import sys
import os
import subprocess
import configargparse
import pandas as pd
import numpy as np
import math
import time

sys.path += ["/home/mo/shared_repos/basenji/basenji"]
import slurm

def generate_basic_bigwig(args, merged_bam, sample_path, sample_name):
    genomecov_cmd = " ".join([
        "bedtools genomecov -bg -5 -ibam",
        merged_bam,
        ">",
        os.path.join(sample_path, f"{sample_name}.bedgraph")
    ])
    sort_cmd = " ".join([
        "sort -k 1,1 -k2,2n",
        os.path.join(sample_path, f"{sample_name}.bedgraph"),
        ">",
        os.path.join(sample_path, f"{sample_name}_sort.bedgraph")
    ])
    bigwig_cmd = " ".join([
        "bedGraphToBigWig",
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
    output_file = f"{os.path.join(sample_path, sample_name)}_bamcov.bw"
    
    if not os.path.exists(output_file):
        bamcov_cmd = " ".join([
            "python /home/mo/shared_repos/basenji/bin/bam_cov.py -c",
            merged_bam,
            output_file, 
        ])
        subprocess.run(
            bamcov_cmd,
            shell = True
        )
    else:
        print("bamcov output already exists")
    
    
    return


def generate_macs_bigwig(args, merged_bam, sample_path, sample_name):
    output_file = os.path.join(sample_path, f"{sample_name}_macs.bw")
    
    #suffixes = ["_macs.bw", "
                               
    if not os.path.exists(output_file):                     
        print("LogFE over background with MACS3...")
        callpeak_cmd = " ".join([
            "macs3 callpeak",
            f"--outdir {sample_path}",
            f"--treatment {merged_bam}",
            f"--control {args.control}",
            # TODO: we should actually scale to some common depth
            # ideally we want to not normalize per sample because 
            # the model can learn to weight different quailty samples
            f"--scale-to large",
            "--gsize 12100000",
            "--format BAMPE",
            "--q 0.01",
            "--bdg",
            f"--name {sample_name}"
        ])
        t_file = os.path.join(sample_path, f"{sample_name}_treat_pileup.bdg")
        c_file = os.path.join(sample_path, f"{sample_name}_control_lambda.bdg")
        bdgcmp_cmd =  " ".join([
            "macs3 bdgcmp",
            f"--outdir {sample_path}",
            f"--tfile {t_file}",
            f"--cfile {c_file}",
            f"--o-prefix {sample_name}",
            "--pseudocount 1",
            "--method logFE"
            # "--method ppois"
        ])
        sort_cmd = " ".join([
            f"sort -k 1,1 -k2,2n",
            os.path.join(sample_path, f"{sample_name}_logFE.bdg"),
            ">",
            os.path.join(sample_path, f"{sample_name}_sort_logFE.bedgraph")
        ])
        # zero out negative enrichments
        # having less reads than background is likely not meaningful
        zero_cmd = " ".join([
            "awk -v OFS=\"\t\" ' {if($4<0)$4=0}1 '",
            os.path.join(sample_path, f"{sample_name}_sort_logFE.bedgraph"), 
            ">",
            os.path.join(sample_path, f"{sample_name}_pos_logFE.bedgraph")
        ])
        bigwig_cmd = " ".join([
            "bedGraphToBigWig",
            os.path.join(sample_path, f"{sample_name}_pos_logFE.bedgraph"),
            args.chrsizes,
            output_file
        ])
        subprocess.run(
            ';'.join([callpeak_cmd, bdgcmp_cmd, sort_cmd, zero_cmd, bigwig_cmd]),
            shell = True,
        )
    else:
        print("MACs output already exists")
   
    return


# Run alignments, qc, and concatonate reads from replicate experiments
def process_samples(sample_sheet, args):
    sample_sheet['replicates'] = sample_sheet['replicates'].apply(lambda x: x.split(";"))
    
    for i in range(sample_sheet.shape[0]):
        fastq_dir = sample_sheet.iloc[i]['fastq_dir']
        sample_name = sample_sheet.iloc[i]['sample_name']
        sample_path = os.path.join(args.savedir, sample_name)
        merged_bam = os.path.join(sample_path, f"{sample_name}.bam")

        print(f"processing {sample_name}")
        
        # 1. run aligment for each replicate
        # if not os.path.exists(sample_path):
        subprocess.run(
            f"mkdir {sample_path}",
            shell = True
        )
        for sra in sample_sheet.iloc[i]['replicates']:
            shell_cmd = " ".join([
                "bash /home/mo/mmagzoub/yeast_sequence_models/bioinformatics/chipseq2bam.sh",
                f"-i {sra}",
                f"-f {fastq_dir}",
                f"-r {args.reference}",
                f"-b {args.bwaindex}",
                f"-s {sample_path}"
            ])
            subprocess.run(
                shell_cmd,
                shell = True
            )
            # wait for alignment and qc
            while True:
                if os.path.exists(os.path.join(sample_path, f"{sra}_masked.bam")):
                    break
                else:
                    time.sleep(60)

        # else:
        #     print('already aligned')
            
        # 2. merge replicates
        # if not os.path.exists(merged_bam):
        bam_files = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith("_masked.bam")]
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
        # wait for merged bam to be created
        while True:
            if os.path.exists(merged_bam):
                break
            else:
                time.sleep(60)
        # else:
        #     print('replicates already merged')
  
        # 3. create bigwigs if applicable
        if args.basic and args.chrsizes:
            generate_basic_bigwig(args, merged_bam, sample_path, sample_name)
        if args.bamcov:
            generate_bamcov_bigwig(merged_bam, sample_path, sample_name)
        if args.macs and args.control:
            generate_macs_bigwig(args, merged_bam, sample_path, sample_name)
    
    return


def make_parser():
    parser = configargparse.ArgParser(
        description='process ChIP data from rossi_et_al.',
    )

    parser.add_argument('--sample_sheet', type=str, required=True)
    parser.add_argument('--savedir', type=str, required=True)
    parser.add_argument('--reference', type=str, required=True)
    parser.add_argument('--bwaindex', type=str, required=True)
    parser.add_argument('--chrsizes', type=str, required=False, default = None)
    parser.add_argument('--control', type=str, required=False, default = None)
    parser.add_argument('--n_batches', type=int, required=False, default = 0)
    parser.add_argument('--macs', default=False, action='store_true')
    parser.add_argument('--bamcov', default=False, action='store_true')
    parser.add_argument('--basic', default=False, action='store_true')
    #parser.add_argument('--rmtmp', default=False, action='store_true')
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
                "python /home/mo/mmagzoub/yeast_sequence_models/bioinformatics/process_ChIPs.py",
                f"--sample_sheet {os.path.join(args.savedir, f'{batch}_sheet.csv')}",
                f"--savedir {args.savedir}",
                f"--reference {args.reference}",
                f"--bwaindex {args.bwaindex}"
            ]
            if args.chrsizes:
                batch_cmd += [f"--chrsizes {args.chrsizes}"]
            if args.control:
                batch_cmd += [f"--control {args.control}"]
            if args.macs:
                batch_cmd += ["--macs"]
            if args.bamcov:
                batch_cmd += ["--bamcov"]
            if args.basic:
                batch_cmd += ["--basic"]
            batch_cmd = " ".join(batch_cmd)

            job = slurm.Job(
                batch_cmd,
                name = f"yeast_chip_{batch}",
                out_file = os.path.join(args.savedir, f"yeast_chip_{batch}.out"),
                err_file = os.path.join(args.savedir, f"yeast_chip_{batch}.err"),
                queue = "standard",
                gpu = 0,
                cpu = 4,
                mem = 16 * 1000,
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
# python /group/idea/basenji_yeast/bioinformatics/process_ChIPs.py \
# --sample_sheet sample_sheet.csv \
# --savedir ./ \
# --n_batches 10 \
# --reference /group/idea/basenji_yeast/data/references/S288C_R64-3-1.fsa \
# --bwaindex /group/idea/basenji_yeast/data/references/bwa/S288C_R64_bwaidx \
# --chrsizes /group/idea/basenji_yeast/data/references/chrom.sizes \
# --control /group/idea/basenji_yeast/data/rossi_et_al/bigwigs/negative_control/NOTAG/NOTAG.bam \
# --macs

# python /home/mo/mmagzoub/yeast_sequence_models/bioinformatics/process_ChIPs.py \
# --sample_sheet sample_sheet.csv \
# --savedir ./ \
# --reference /group/idea/basenji_yeast/data/references/S288C_R64-3-1.fsa \
# --bwaindex /group/idea/basenji_yeast/data/references/bwa/S288C_R64_bwaidx \
# --chrsizes /group/idea/basenji_yeast/data/references/chrom.sizes \
# --bamcov

if __name__ == '__main__':
    main()