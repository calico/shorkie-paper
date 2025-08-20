import os
import h5py
import pyBigWig
import pandas as pd
import numpy as np

from optparse import OptionParser
import sys
import scipy.interpolate

class CovFace:
    def __init__(self, cov_file):
        self.cov_file = cov_file
        self.bigwig = False
        self.bed = False

        cov_ext = os.path.splitext(self.cov_file)[1].lower()
        if cov_ext == ".gz":
            cov_ext = os.path.splitext(self.cov_file[:-3])[1].lower()

        if cov_ext in [".bed", ".narrowpeak"]:
            self.bed = True
            self.preprocess_bed()

        elif cov_ext in [".bw", ".bigwig"]:
            self.cov_open = pyBigWig.open(self.cov_file, "r")
            self.bigwig = True

        elif cov_ext in [".h5", ".hdf5", ".w5", ".wdf5"]:
            self.cov_open = h5py.File(self.cov_file, "r")

        else:
            print(
                'Cannot identify coverage file extension "%s".' % cov_ext,
                file=sys.stderr,
            )
            exit(1)

    def preprocess_bed(self):
        # read BED
        bed_df = pd.read_csv(
            self.cov_file, sep="\t", usecols=range(3), names=["chr", "start", "end"]
        )

        # for each chromosome
        self.cov_open = {}
        for chrm in bed_df.chr.unique():
            bed_chr_df = bed_df[bed_df.chr == chrm]

            # find max pos
            pos_max = bed_chr_df.end.max()

            # initialize array
            self.cov_open[chrm] = np.zeros(pos_max, dtype="bool")

            # set peaks
            for peak in bed_chr_df.itertuples():
                self.cov_open[peak.chr][peak.start : peak.end] = 1

    def read(self, chrm, start, end):
        if self.bigwig:
            cov = self.cov_open.values(chrm, start, end, numpy=True).astype("float16")

        else:
            if chrm in self.cov_open:
                cov = self.cov_open[chrm][start:end]

                # handle mysterious inf's
                cov = np.clip(cov, np.finfo(np.float16).min, np.finfo(np.float16).max)

                # pad
                pad_zeros = end - start - len(cov)
                if pad_zeros > 0:
                    cov_pad = np.zeros(pad_zeros, dtype="bool")
                    cov = np.concatenate([cov, cov_pad])

            else:
                print(
                    "WARNING: %s doesn't see %s:%d-%d. Setting to all zeros."
                    % (self.cov_file, chrm, start, end),
                    file=sys.stderr,
                )
                cov = np.zeros(end - start, dtype="float16")

        return cov

    def close(self):
        if not self.bed:
            self.cov_open.close()



def read_coverage(genome_cov_file, chrm, start, end):
    """
    Given a genome coverage file and genomic coordinates, return coverage values.

    Parameters
    ----------
    genome_cov_file : str
        Path to the genome coverage file (bigWig, BED, or HDF5).
    chrm : str
        Chromosome name.
    start : int
        Start coordinate (0-based).
    end : int
        End coordinate.

    Returns
    -------
    cov : numpy.ndarray
        Array of coverage values (float16).
    """
    cov_face = CovFace(genome_cov_file)
    cov = cov_face.read(chrm, start, end)
    cov_face.close()
    return cov


def seq_norm(seq_cov_nt):
    # -w 16 \
    # -c 1024 \
    usage = "usage: %prog [options] <genome_cov_file> <seqs_bed_file> <seqs_cov_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-b",
        dest="blacklist_bed",
        help="Set blacklist nucleotides to a baseline value.",
    )
    parser.add_option(
        "--black_pct",
        dest="blacklist_pct",
        default=0.5,
        type="float",
        help="Clip blacklisted regions to this distribution value [Default: %default",
    )
    parser.add_option(
        "-c",
        dest="clip",
        default=None,
        type="float",
        help="Clip values post-summary to a maximum [Default: %default]",
    )
    parser.add_option(
        "--clip_soft",
        dest="clip_soft",
        default=None,
        type="float",
        help="Soft clip values, applying sqrt to the execess above the threshold [Default: %default]",
    )
    parser.add_option(
        "--clip_pct",
        dest="clip_pct",
        default=0.9999999,
        type="float",
        help="Clip extreme values to this distribution value [Default: %default",
    )
    parser.add_option(
        "--crop",
        dest="crop_bp",
        default=0,
        type="int",
        help="Crop bp off each end [Default: %default]",
    )
    parser.add_option(
        "-i",
        dest="interp_nan",
        default=False,
        action="store_true",
        help="Interpolate NaNs [Default: %default]",
    )
    parser.add_option(
        "-s",
        dest="scale",
        default=1.0,
        type="float",
        help="Scale values by [Default: %default]",
    )
    parser.add_option(
        "-u",
        dest="sum_stat",
        default="sum",
        help="Summary statistic to compute in windows [Default: %default]",
    )
    parser.add_option(
        "-w",
        dest="pool_width",
        default=1,
        type="int",
        help="Average pooling width [Default: %default]",
    )
    (options, args) = parser.parse_args()

    # if len(args) != 3:
    #     parser.error("")
    # else:
    #     genome_cov_file = args[0]
    #     seqs_bed_file = args[1]
    #     seqs_cov_file = args[2]
    assert options.crop_bp >= 0

    # Update options after parsing
    options.pool_width = 16  # Update -w to 16
    options.clip = 1024  # Update -c to 1024
    # print("options: ", options)

    seq_cov_nt = seq_cov_nt.astype("float32")

    # interpolate NaN
    if options.interp_nan:
        seq_cov_nt = interp_nan(seq_cov_nt)

    seq_len_nt = 14336
    seq_len_nt -= 2 * options.crop_bp
    target_length = seq_len_nt // options.pool_width
    # print("** target_length: ", target_length) # 
    # determine baseline coverage
    if target_length >= 8:
        baseline_cov = np.percentile(seq_cov_nt, 100 * options.blacklist_pct)
        baseline_cov = np.nan_to_num(baseline_cov)
    else:
        baseline_cov = 0

    # # set blacklist to baseline
    # if mseq.chr in black_chr_trees:
    #     for black_interval in black_chr_trees[mseq.chr][mseq.start : mseq.end]:
    #         # adjust for sequence indexes
    #         black_seq_start = black_interval.begin - mseq.start
    #         black_seq_end = black_interval.end - mseq.start
    #         black_seq_values = seq_cov_nt[black_seq_start:black_seq_end]
    #         seq_cov_nt[black_seq_start:black_seq_end] = np.clip(
    #             black_seq_values, -baseline_cov, baseline_cov
    #         )
    #         # seq_cov_nt[black_seq_start:black_seq_end] = baseline_cov

    # set NaN's to baseline
    if not options.interp_nan:
        nan_mask = np.isnan(seq_cov_nt)
        seq_cov_nt[nan_mask] = baseline_cov

    # crop
    if options.crop_bp > 0:
        seq_cov_nt = seq_cov_nt[options.crop_bp : -options.crop_bp]

    # scale
    seq_cov_nt = options.scale * seq_cov_nt

    # sum pool
    seq_cov = seq_cov_nt.reshape(target_length, options.pool_width)
    if options.sum_stat == "sum":
        seq_cov = seq_cov.sum(axis=1, dtype="float32")
    elif options.sum_stat == "sum_sqrt":
        seq_cov = seq_cov.sum(axis=1, dtype="float32")
        seq_cov = -1 + np.sqrt(1 + seq_cov)
    elif options.sum_stat == "sum_exp75":
        seq_cov = seq_cov.sum(axis=1, dtype="float32")
        seq_cov = -1 + (1 + seq_cov) ** 0.75
    elif options.sum_stat in ["mean", "avg"]:
        seq_cov = seq_cov.mean(axis=1, dtype="float32")
    elif options.sum_stat in ["mean_sqrt", "avg_sqrt"]:
        seq_cov = seq_cov.mean(axis=1, dtype="float32")
        seq_cov = -1 + np.sqrt(1 + seq_cov)
    elif options.sum_stat == "median":
        seq_cov = seq_cov.median(axis=1)
    elif options.sum_stat == "max":
        seq_cov = seq_cov.max(axis=1)
    elif options.sum_stat == "peak":
        seq_cov = seq_cov.mean(axis=1, dtype="float32")
        seq_cov = np.clip(np.sqrt(seq_cov * 4), 0, 1)
    else:
        print(
            'ERROR: Unrecognized summary statistic "%s".' % options.sum_stat,
            file=sys.stderr,
        )
        exit(1)

    # clip
    if options.clip_soft is not None:
        clip_mask = seq_cov > options.clip_soft
        seq_cov[clip_mask] = (
            options.clip_soft
            - 1
            + np.sqrt(seq_cov[clip_mask] - options.clip_soft + 1)
        )
    if options.clip is not None:
        seq_cov = np.clip(seq_cov, -options.clip, options.clip)

    # clip float16 min/max
    seq_cov = np.clip(seq_cov, np.finfo(np.float16).min, np.finfo(np.float16).max)

    # # save
    # targets.append(seq_cov.astype("float16"))
    return seq_cov
