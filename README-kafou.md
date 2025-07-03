# Prerequisites

You will need to have `mamba` available to you. It is part of [miniforge](https://github.com/conda-forge/miniforge). As of 2025-7-3,
the installation instructions on kafou are:

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh

Answer yes to the question about shell initialization, log out and log back in, and the `mamba` command should now be
available to you.

# Initial Environment Creation

To create `environment-kafou.yml`, I did the following on `hex-cast`.

    mamba create -n aurora
    mamba activate aurora
    CONDA_OVERRIDE_CUDA=12.8 mamba install cuda-version=12.8 pytorch-gpu arraylake icechunk xarray dask
    CONDA_OVERRIDE_CUDA=12.8 mamba install --only-deps microsoft-aurora
    mamba env export > environment-kafou.yml

The `environment-kafou.yml` file then needs to be edited to remove the `name:` and `prefix:` lines.

The `CONDA_OVERRIDE_CUDA` variable is needed to be set in this way because there are no NVIDIA drivers on `hex-cast`.

# Installation (for development)

Do something like this on `hex-cast`:

      CONDA_OVERRIDE_CUDA=12.8 mamba create -n aurora -f environment-kafou.yml
      mamba activate aurora
      pip install --editable .
