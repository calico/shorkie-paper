# 03_eval — model evaluation

> **Advanced / reproduction-only.** These are the evaluation sweeps behind the paper's figures. To just
> use Shorkie, see [`../../examples/`](../../examples) and <https://khchao.com/shorkie/>.

| Subdirectory | Evaluates |
|--------------|-----------|
| `lm/lm_model_eval/` | Masked-LM test loss and perplexity per corpus tier (feeds Figure 1F/G). |
| `lm/genome_evaluation/` | Genome-level LM analyses: per-window statistics, annotation/BUSCO checks, genome-distance sketches. |
| `lm/model_evaluation/` | Self-supervised vs supervised comparison over the shared eval set. |
| `supervised/track_prediction_eval/` | Track- and gene-level accuracy for the supervised ensemble (feeds Figure 3). |

Every script resolves paths through `config/paths.yaml` and reads the `results.*` keys, which point at
work-directory outputs that are **not** part of the public release — so these stages need the original
training tree, not just a download. Downstream figure notebooks consume the cached outputs instead; see
[`../../reproduction/README.md`](../../reproduction/README.md).

`#SBATCH` headers target the authors' cluster; run portably with
`scripts/common/submit.sh --profile gpu <script>` or `SHORKIE_LOCAL=1` for no scheduler.
