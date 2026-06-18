# Shorkie reproducibility image (scheduler-free path).
#
# Build from the repo root WITH submodules checked out:
#   git submodule update --init external/baskerville-yeast external/westminster
#   docker build -f containers/Dockerfile -t shorkie:latest .
#
# Run the minimal example:
#   docker run --rm -v $PWD:/work -w /work shorkie:latest \
#     "python minimal_example/run_shorkie_variant.py --help"
FROM continuumio/miniconda3:24.1.2-0

SHELL ["/bin/bash", "-lc"]
WORKDIR /opt/shorkie

# System libraries for pysam / pybigwig / bedtools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget zlib1g-dev libbz2-dev liblzma-dev \
        libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Create the conda env first (cached unless environment.yml changes).
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy
ENV PATH=/opt/conda/envs/yeast_ml/bin:$PATH

# Copy the repository (submodules included) and install editable packages.
COPY . /opt/shorkie
RUN pip install -e external/baskerville-yeast && \
    pip install -e external/westminster && \
    pip install -e .

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["python -c 'import shorkie, baskerville; print(\"shorkie + baskerville OK\")'"]
