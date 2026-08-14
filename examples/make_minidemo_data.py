#!/usr/bin/env python3
"""Build a tiny train/valid/test TFRecord dataset for the fine-tuning mini-demo.

The released supervised dataset is ~10 GB (8 folds x ~1.3 GB), which is far more
than a "see it work in a few minutes" demo needs. Rather than re-encode data --
which would risk silently diverging from the real feature schema -- this copies a
handful of *real serialized records* straight out of the released fold-0 shard, so
the schema is correct by construction.

The shard is ZLIB-compressed and the input is a byte-range prefix of it, so the
final record is truncated; we read until TensorFlow reports the tail and keep the
complete records.

    python examples/make_minidemo_data.py \
        --prefix_tfr fold0_prefix.tfr \
        --statistics statistics.json \
        --targets targets.txt \
        --out_dir minidemo_data

Produces:
    minidemo_data/
      statistics.json          (same descriptor, with train/valid/test_seqs)
      targets.txt
      tfrecords/{train,valid,test}-0.tfr
"""
import argparse
import json
import os
import shutil


def read_records(path, limit):
    """Yield up to `limit` complete serialized records from a (possibly truncated) shard."""
    import tensorflow as tf

    out = []
    ds = tf.data.TFRecordDataset(path, compression_type="ZLIB")
    try:
        for rec in ds:
            out.append(rec.numpy())
            if len(out) >= limit:
                break
    except tf.errors.DataLossError:
        # expected: the byte-range prefix ends mid-record
        pass
    return out


def write_records(records, path):
    import tensorflow as tf

    opts = tf.io.TFRecordOptions(compression_type="ZLIB")
    with tf.io.TFRecordWriter(path, opts) as w:
        for r in records:
            w.write(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix_tfr", required=True, help="byte-range prefix of a released fold shard")
    ap.add_argument("--statistics", required=True, help="statistics.json from the released dataset")
    ap.add_argument("--targets", required=True, help="targets.txt (5215-track sheet)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=8)
    ap.add_argument("--n_valid", type=int, default=4)
    ap.add_argument("--n_test", type=int, default=4)
    args = ap.parse_args()

    need = args.n_train + args.n_valid + args.n_test
    recs = read_records(args.prefix_tfr, need)
    if len(recs) < need:
        raise SystemExit(
            f"error: recovered only {len(recs)} complete records from {args.prefix_tfr}, "
            f"need {need}. Fetch a larger byte range (each record is ~6 MB compressed).")

    tfr_dir = os.path.join(args.out_dir, "tfrecords")
    os.makedirs(tfr_dir, exist_ok=True)

    splits = {
        "train": recs[:args.n_train],
        "valid": recs[args.n_train:args.n_train + args.n_valid],
        "test":  recs[args.n_train + args.n_valid:need],
    }
    for label, chunk in splits.items():
        write_records(chunk, os.path.join(tfr_dir, f"{label}-0.tfr"))
        print(f"  wrote {len(chunk):2d} records -> tfrecords/{label}-0.tfr")

    # Same dataset descriptor, re-labelled for train/valid/test instead of fold*.
    stats = json.load(open(args.statistics))
    for k in [k for k in stats if k.endswith("_seqs")]:
        del stats[k]
    for label, chunk in splits.items():
        stats[f"{label}_seqs"] = len(chunk)
    with open(os.path.join(args.out_dir, "statistics.json"), "w") as fh:
        json.dump(stats, fh, indent=4)

    shutil.copy(args.targets, os.path.join(args.out_dir, "targets.txt"))
    print(f"  dataset ready: {args.out_dir} "
          f"({stats['num_targets']} targets, seq_length {stats['seq_length']})")


if __name__ == "__main__":
    main()
