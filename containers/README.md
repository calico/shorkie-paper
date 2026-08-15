# containers/

Container definitions that build the full environment (conda env + both pinned submodules + the
`shorkie` package), for running without a scheduler or a conda install.

| File | Runtime |
|------|---------|
| `Dockerfile` | Docker / Podman |
| `apptainer.def` | Apptainer / Singularity (HPC) |

```bash
# Docker
docker build -t shorkie -f containers/Dockerfile .
docker run --rm -it -v "$PWD":/work -w /work shorkie bash

# Apptainer
apptainer build shorkie.sif containers/apptainer.def
apptainer exec --bind "$PWD":/work shorkie.sif bash
```

Both install `external/baskerville-yeast` and `external/westminster`, so initialise the submodules
first (`git submodule update --init`). Neither image bundles model weights or data — fetch those with
[`../data/download.sh`](../data/download.sh) and bind-mount them in.

GPU use needs the CUDA-enabled TensorFlow build plus `tensorrt==8.6.1`; inference and the figure
reproduction run fine on CPU.
