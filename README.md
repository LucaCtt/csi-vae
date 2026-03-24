# CSI-VAE

This repository contains the code for *insert paper citation here*.

It is composed by a `launcher` that creates Optuna studies, which in turn create `jobs` that train a VAE on the CSI data and evaluates the performance of the trained model. The launcher is meant to run locally and be kept alive for the whole duration of the optimization process, while the jobs are run on AWS Batch.

## Installation

To install the required dependencies, use `uv`:

```bash
uv sync --extra launcher
```

To run a training job locally, run:

```bash
uv run python src/csi_vae/jobs/job.py
```

## Acknowledgements

The work is partially supported by the European Office of Aerospace ResearchDevelopment (EOARD) under award number FA8655-22-1-7017 and by the US DEVCOM Army Research Laboratory (ARL) under Cooperative Agreements #W911NF2220243 and #W911NF1720196. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the United States government.

## Authors

- Luca Cotti <luca.cotti@unibs.it>
- Marco Cominelli <marco.cominelli@polimi.it>
