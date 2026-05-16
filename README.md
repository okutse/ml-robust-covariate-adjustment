# ml-robust-covariate-adjustment

Reproducibility repository for the paper *"Machine learning for robust covariate adjustment: enhancing efficiency and mitigating bias in randomized trials with missing outcomes."*

This document provides complete instructions for reproducing all data generation, simulation, and analysis workflows in this project. All analyses are implemented in R with carefully controlled random seeds and frozen package dependencies to ensure full reproducibility.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Repository Structure](#repository-structure)
4. [Data Generation](#data-generation)
5. [Running Complete-Data Analyses](#running-complete-data-analyses)
6. [Running Missing-Outcome Analyses](#running-missing-outcome-analyses)
7. [Simulation Outputs and Checkpointing](#simulation-outputs-and-checkpointing)
8. [Troubleshooting and FAQ](#troubleshooting-and-faq)

---

## Quick Start

For users who want to run analyses on pre-generated data:

```bash
# 1. Clone the repository and navigate to it
git clone https://github.com/[org]/ml-robust-covariate-adjustment.git
cd ml-robust-covariate-adjustment

# 2. Restore the R environment
Rscript -e "renv::restore()"

# 3. Run a single-stage complete-data analysis (using existing datasets)
REPLICATES_TO_RUN=100 USE_PARALLEL=true Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R

# 4. Check results in: simulations/complete_data/complete_data_results/single_stage_cf/
```

---

## Environment Setup

### Prerequisites

- **R >= 4.5.0** (4.5.1 or later recommended)
- **renv >= 0.15.0** (for package management)
- Sufficient disk space (~10–15 GB) for datasets and intermediate results
- For parallel execution: multi-core CPU (analyses auto-detect available cores)

### Installing the R Environment

This project uses **renv** to manage package versions and ensure reproducibility across machines.

#### First-Time Setup (on the development machine):

If the repository already includes `renv.lock` (checked into version control):

```bash
cd ml-robust-covariate-adjustment
Rscript -e "renv::restore()"
```

If you need to initialize renv from scratch (e.g., in a fresh clone):

```bash
Rscript scripts/setup_renv.R
```

#### Restoring on Another Machine:

1. Clone the repository.
2. Open R in the repository root.
3. Run:
   ```r
   renv::restore()
   ```

This will install the exact versions of all dependencies locked in `renv.lock`, ensuring bit-for-bit reproducibility of analyses.

### Verifying the Environment

To confirm packages are correctly installed:

```r
# Open R and run:
renv::status()  # Should report "No differences"
```

---

## Repository Structure

```
ml-robust-covariate-adjustment/
├── README.md                           # This file
├── LICENSE                             # License (e.g., MIT)
├── ml-robust-covariate-adjustment.Rproj  # RStudio project file
├── renv/                               # renv configuration directory
│   ├── activate.R                      # Automatically sources renv on project open
│   ├── settings.dcf
│   ├── settings.json
│   └── library/                        # Cached packages (machine-specific)
├── renv.lock                           # Locked package versions (committed to git)
├── scripts/                            # Root-level utility scripts
│   └── setup_renv.R                    # Initializes renv and dependencies
│
├── simulations/
│   ├── complete_data/                  # Complete-data (no missing outcomes) analyses
│   │   ├── datasets/                   # Generated datasets (1000 replicates per scenario)
│   │   │   ├── complete_n200_r2_0p20.RData
│   │   │   ├── complete_n200_r2_0p40.RData
│   │   │   ├── ... (20 scenario files total)
│   │   │   └── complete_n10000_r2_0p80.RData
│   │   │
│   │   ├── complete_data_results/      # Analysis outputs
│   │   │   ├── single_stage_cf/        # Single-stage cross-fitted results
│   │   │   │   ├── scenario_checkpoint.csv
│   │   │   │   ├── input_manifest.csv
│   │   │   │   ├── complete_n200_r2_0p20/
│   │   │   │   │   ├── method_checkpoint.csv
│   │   │   │   │   ├── replicate_cache/  # Per-replicate outputs
│   │   │   │   │   └── [method]_*.csv    # Aggregated method results
│   │   │   │   └── single_stage_cf_results.csv  # Aggregated across scenarios
│   │   │   │
│   │   │   ├── two_stage_cf/           # Two-stage cross-fitted results
│   │   │   │   └── [similar structure]
│   │   │   │
│   │   │   ├── single_stage/           # Non-cross-fitted single-stage
│   │   │   │   └── [similar structure]
│   │   │   │
│   │   │   └── two_stage/              # Non-cross-fitted two-stage
│   │   │       └── [similar structure]
│   │   │
│   │   └── scripts/
│   │       ├── generate_complete_data.R     # Data generation script
│   │       ├── model_params_using_X.R       # Model parameter estimation
│   │       ├── cf/
│   │       │   ├── cf_single_stage_helpers.R   # Single-stage CF helper functions
│   │       │   ├── cf_two_stage_helpers.R      # Two-stage CF helper functions
│   │       │   ├── single_stage_cf_results.R   # Single-stage CF runner
│   │       │   └── two_stage_cf_results.R      # Two-stage CF runner
│   │       └── non_cf/
│   │           ├── single_stage_model_helpers.R
│   │           ├── single_stage_results.R
│   │           ├── two_stage_model_helpers.R
│   │           └── two_stage_results.R
│   │
│   └── missing_outcomes/              # Missing-outcome analyses
│       ├── miss_datasets/              # Generated datasets (1000 replicates per scenario)
│       │   ├── setting_one/
│       │   │   ├── setting_one_n200_r2_0p20.RData
│       │   │   └── ... (20 scenario files per setting)
│       │   ├── setting_two/
│       │   ├── setting_three/
│       │   └── setting_four/
│       │
│       ├── miss_intermediate_results/  # Per-setting/procedure analysis outputs
│       │   ├── setting_one/
│       │   │   ├── single_stage/
│       │   │   │   └── m1/, m2/        # Per model-spec directories
│       │   │   ├── two_stage/
│       │   │   └── tmle/
│       │   └── [setting_two, setting_three, setting_four]
│       │
│       └── miss_scripts/
│           ├── miss_registry.R         # Model, covariate, procedure registry
│           ├── miss_runner.R           # Orchestration runner
│           ├── generate_miss_data.R    # Data generation script
│           ├── run_missing_single_stage.R
│           ├── run_missing_two_stage.R
│           ├── run_missing_single_stage_drml_bc.R
│           └── run_missing_tmle.R
│
├── logs/                               # Execution logs (gitignored)
│   ├── complete_data/
│   │   ├── single_stage_cf/
│   │   ├── two_stage_cf/
│   │   └── ...
│   └── missing_outcomes/
│       ├── setting_one/
│       └── ...
│
└── paper_results/                      # Final aggregated results for the paper
    └── [publication-ready CSVs and tables]
```

---

## Data Generation

### Quick Summary

The project uses **deterministic, seeded data generation** for full reproducibility. Data generation code is included; users can either:

1. **Use pre-generated datasets** (Recommended for most users)
2. **Regenerate datasets locally** using the provided scripts
3. **Download datasets from Zenodo** for a specific paper version

### Data Generation Details

#### Complete-Data Scenario Generation

**Location:** `simulations/complete_data/scripts/generate_complete_data.R`

Generates synthetic datasets with:
- **Sample sizes:** $n \in \{200, 500, 1000, 2000, 5000\}$ 
- **Target $R^2$:** $\{0.20, 0.40, 0.60, 0.80\}$
- **Replicates:** 1000 per scenario (i.e., 1000 independent datasets per $n/R^2$ combination)
- **Total:** 20 scenario files (~1.5–2 GB)

**Data generating mechanism:**
- Treatment $A$ equally assigned (50/50)
- Confounders $\mathbf{Z} = (Z_1, Z_2, Z_3, Z_4) \sim \mathcal{N}(0, I_4)$ (unobserved)
- Observed covariates $\mathbf{X}$ derived from $\mathbf{Z}$ via known transformations
- Outcome: $Y = 210 + 50 \cdot A + 27.4 \sum Z_i + \varepsilon$, with $\varepsilon$ scaled to achieve target $R^2$
- Seeds: `base_seed = 12345 + n + round(target_r2 * 100)` per scenario, with `doRNG::registerDoRNG()` for reproducible parallel sampling

**File Format:** `.RData` (R binary format)
- **Objects:** `datasets` (list of 1000 data frames), `metadata` (data frame of scenario parameters)
- **Size per file:** ~50–100 MB

#### Missing-Outcome Scenario Generation

**Location:** `simulations/missing_outcomes/miss_scripts/generate_miss_data.R`

Generates datasets with missing outcomes under four **missingness mechanisms**:

- **Setting One (MCAR):** $P(R=1) = \text{expit}(-Z_1 + 0.5 Z_2 - 0.25 Z_3 - 0.1 Z_4)$
- **Setting Two (MAR, Treatment-dependent):** $P(R=1) = \text{expit}(-Z_1 + A + 0.5 Z_2 - 0.25 Z_3 - 0.1 Z_4)$
- **Setting Three (MAR, Treatment $\times$ Covariate interaction):** $P(R=1) = \text{expit}(-Z_1 + A + A \cdot Z_1 + 0.5 Z_2 - 0.25 Z_3 - 0.1 Z_4)$
- **Setting Four (Heterogeneous treatment effects):** Same as Setting Three with treatment effect heterogeneity

Specifications match complete-data (same $n$, $R^2$, 1000 replicates per scenario).

**File Format:** Same `.RData` structure; includes `R` (missingness indicator) and `e` (noise term).

### Option 1: Use Pre-Generated Datasets (Recommended)

If datasets are already in the repository:

```bash
# No action needed; datasets are included in:
# - simulations/complete_data/datasets/*.RData
# - simulations/missing_outcomes/miss_datasets/*/*.RData
```

### Option 2: Regenerate Datasets Locally

#### Complete Data:

```bash
cd ml-robust-covariate-adjustment
Rscript simulations/complete_data/scripts/generate_complete_data.R
```

**Estimated runtime:** ~5–15 minutes (depending on CPU and whether parallelization is enabled).

**Output:**
- 20 `.RData` files in `simulations/complete_data/datasets/`
- Input manifest (MD5 checksums) saved to `simulations/complete_data/complete_data_results/input_manifest.csv`

#### Missing-Outcome Data:

```bash
Rscript simulations/missing_outcomes/miss_scripts/generate_miss_data.R
```

**Estimated runtime:** ~10–25 minutes.

**Output:**
- Four setting subdirectories under `simulations/missing_outcomes/miss_datasets/`
- 80 `.RData` files total (20 scenarios × 4 settings)

### Option 3: Download Datasets from Zenodo

To obtain the exact datasets used in the paper:

```bash
# The runners prefer the project-local copy if it exists.
# If it does not, they resolve the Zenodo DOI, download Data.zip, and unpack
# the nested complete_data and missing_outcomes folders into the expected paths.

# Complete-data example:
zenodo_data_doi="https://doi.org/10.5281/zenodo.19393092" \
Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R

# Missing-outcomes example:
zenodo_data_doi="https://doi.org/10.5281/zenodo.19393092" \
SETTING=setting_three Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R
```

**Data integrity:** After download/generation, verify checksums:

```r
# In R:
source("simulations/complete_data/scripts/generate_complete_data.R")
manifest <- read.csv("simulations/complete_data/complete_data_results/input_manifest.csv")
# Check against manifest$md5 values
```

---

## Running Complete-Data Analyses

### Overview

Complete-data analyses compare machine-learning covariate adjustment methods on data with **no missing outcomes** using both cross-fitted and non-cross-fitted procedures.

### Analyses Available

| Procedure          | Cross-Fitted | Runner Script                              | Description                           |
|--------------------|:-:|-------------------------------------------|------------------------------------|
| Single-stage (CF)  | ✓ | `cf/single_stage_cf_results.R` | Single-stage covariate adjustment with cross-fitting |
| Two-stage (CF)     | ✓ | `cf/two_stage_cf_results.R` | Two-stage adjustment (Cohen & Fogarty 2024) with CF |
| Single-stage       | ✗ | `non_cf/single_stage_results.R` | Single-stage without cross-fitting |
| Two-stage          | ✗ | `non_cf/two_stage_results.R` | Two-stage without cross-fitting |

### Running Single-Stage Cross-Fitted Analysis

**Basic run:**

```bash
cd ml-robust-covariate-adjustment
Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R
```

**Run with custom settings:**

```bash
# Run only 50 replicates per scenario, disable parallelization
REPLICATES_TO_RUN=50 USE_PARALLEL=false \
Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R

# Run with full datasets, enable parallelization (auto-detect cores)
REPLICATES_TO_RUN=500 USE_PARALLEL=true \
Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R
```

### Environment Variables for Complete-Data Analyses

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `REPLICATES_TO_RUN` | 500 | 1–1000 | Number of replicates per scenario to process |
| `USE_PARALLEL` | `true` | `true`, `false` | Enable multi-core parallelization |
| `DATA_SOURCE` | `local` | `local`, `archive`, `regenerate` | Use the project-local copy, an unpacked archive, or regenerate new datasets |
| `RESET_CHECKPOINT` | `false` | `true`, `false` | Clear checkpoints and restart from scratch |
| `ZENODO_URL` | "" | URL string | Direct Zenodo URL for the archive Data.zip file (legacy alias; `ZENODO_DATA_DOI` is preferred) |
| `ZENODO_DATA_DOI` | "" | DOI or DOI URL | Zenodo DOI or URL; used to resolve the archive Data.zip file |
| `SYNC_ARCHIVE_TO_LOCAL` | `false` | `true`, `false` | Copy downloaded archive to `simulations/complete_data/datasets/` |

### Running All Complete-Data Procedures

To run all four procedures sequentially:

```bash
#!/bin/bash
# Run all complete-data analyses using the local datasets folder when present.

cd ml-robust-covariate-adjustment

# Cross-fitted procedures
Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R
Rscript simulations/complete_data/scripts/cf/two_stage_cf_results.R

# Non-cross-fitted procedures
Rscript simulations/complete_data/scripts/non_cf/single_stage_results.R
Rscript simulations/complete_data/scripts/non_cf/two_stage_results.R

echo "All complete-data analyses completed."
```

### Output Structure

Results are saved to `simulations/complete_data/complete_data_results/<procedure>/`:

- **Scenario-level results:** `<scenario>_results.csv` per scenario
- **Aggregated procedure results:** `<procedure>_results.csv` (e.g., `single_stage_cf_results.csv`)
- **Method results per scenario:** `<scenario>/<method>_*.csv` and diagnostics
- **Checkpoints:** `scenario_checkpoint.csv`, method-level checkpoints
- **Input manifest:** `input_manifest.csv` with file checksums

**Column definitions** (see paper):
- `Estimator`: Method name (e.g., `lm`, `rf`, `bart`, `super`)
- `mean_estimate`: Monte Carlo mean of point estimates across replicates
- `mean_bias`: MC bias relative to true effect (true effect = 50)
- `mc_var_estimate`: Monte Carlo variance
- `coverage_95`: Coverage of 95% confidence intervals
- `relative_efficiency`: Efficiency relative to linear model (lm)

### Resuming Interrupted Analyses

If an analysis is interrupted (e.g., timeout, machine crash):

```bash
# Simply re-run the same command
# The runner automatically:
# 1. Reads existing checkpoints
# 2. Skips completed scenarios/methods/replicates
# 3. Resumes from the next unfinished replicate

Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R
```

To **force a complete re-run** from scratch:

```bash
RESET_CHECKPOINT=true Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R
```

---

## Running Missing-Outcome Analyses

### Overview

Missing-outcome analyses apply doubly-robust and targeted learning methods to handle missing outcomes under different missingness mechanisms across four settings.

### Analyses Available

| Procedure | Setting | Model Specs | Runner Script |
|-----------|---------|-------------|---------------|
| Single-stage | All (1–4) | m1, m2 | `run_missing_single_stage.R` |
| Two-stage | All (1–4) | m1, m2 | `run_missing_two_stage.R` |
| DR-ML-BC | All (1–4) | m1, m2 | `run_missing_single_stage_drml_bc.R` |
| TMLE | All (1–4) | m1, m2 | `run_missing_tmle.R` |

**Model specifications:**
- **m1:** Observed covariates $\mathbf{X} = (X_1, X_2, X_3, X_4)$
- **m2:** Subset of covariates $\mathbf{X} = (X_2, X_3, X_4)$ (misses $X_1$)

**Settings:**
- **Setting One (MCAR):** Missingness independent of treatment and outcome
- **Setting Two (MAR):** Missingness depends on treatment
- **Setting Three (MAR):** Missingness depends on treatment × covariate interaction
- **Setting Four (CATE):** Conditional average treatment effects (heterogeneous treatment effects)

### Running Single-Stage Analysis for Setting One

**Basic:**

```bash
cd ml-robust-covariate-adjustment
SETTING=setting_one Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R
```

**With custom settings:**

```bash
SETTING=setting_one REPLICATES_TO_RUN=100 USE_PARALLEL=true \
  Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R

# Run Setting Three (default)
SETTING=setting_three Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R
```

### Running All Missing-Outcome Analyses

```bash
#!/bin/bash
cd ml-robust-covariate-adjustment

# Loop over all settings and procedures
for setting in setting_one setting_two setting_three setting_four; do
  echo "Running $setting analyses..."
  SETTING=$setting Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R
  SETTING=$setting Rscript simulations/missing_outcomes/miss_scripts/run_missing_two_stage.R
  SETTING=$setting Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage_drml_bc.R
  SETTING=$setting Rscript simulations/missing_outcomes/miss_scripts/run_missing_tmle.R
done

echo "All missing-outcome analyses completed."
```

### Environment Variables for Missing-Outcome Analyses

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `SETTING` | `setting_three` | `setting_one`, `setting_two`, `setting_three`, `setting_four` | Missingness mechanism setting |
| `REPLICATES_TO_RUN` | 500 | 1–1000 | Number of replicates per scenario |
| `USE_PARALLEL` | `true` | `true`, `false` | Enable multi-core execution |
| `DATA_SOURCE` | `local` | `local`, `archive`, `regenerate` | Dataset source |
| `RESET_CHECKPOINT` | `false` | `true`, `false` | Clear checkpoints and restart |
| `ARCHIVE_DATASETS_DIR` | "" | Path | Existing project-local archive directory to use instead of downloading Data.zip |
| `ZENODO_DATA_DOI` | "" | DOI or DOI URL | Zenodo DOI or URL used when the local copy is missing |

### Output Structure

Results are saved to `simulations/missing_outcomes/miss_intermediate_results/<setting>/<procedure>/<model_spec>/`:

- **Scenario-level results:** `<setting>_<scenario>_<model_spec>.csv`
- **Aggregated procedure results:** `<setting>_<procedure>_<model_spec>_aggregated.csv`
- **Across model specs:** `<setting>_<procedure>_aggregated.csv`
- **Checkpoints:** `scenario_checkpoint.csv`, method-level checkpoints
- **Logs:** `logs/missing_outcomes/<setting>_<procedure>_logs.txt`

### Resuming Missing-Outcome Analyses

Same as complete-data analyses:

```bash
# Automatic resume
SETTING=setting_one Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R

# Force restart
SETTING=setting_one RESET_CHECKPOINT=true \
  Rscript simulations/missing_outcomes/miss_scripts/run_missing_single_stage.R
```

---

## Simulation Outputs and Checkpointing

### Checkpoint System

All simulations use a **robust checkpoint system** to enable safe interruption and resumption:

1. **Scenario checkpoint:** `scenario_checkpoint.csv`
   - Records which input files have been fully processed
   - Format: `file, status, timestamp, error`

2. **Method checkpoint:** `method_checkpoint.csv` (per scenario)
   - Records completion status of each estimator method
   - Format: `method, status, timestamp, error`

3. **Per-replicate checkpoint:** `<method>_replicate_checkpoint.csv`
   - Atomic per-replicate record of completion
   - Format: `replicate, status, timestamp, error`

### Restarting and Resuming

**Automatic resume (default):**

```bash
# If interrupted, simply re-run
Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R
# Continues from the next unfinished method/replicate
```

**Force from scratch:**

```bash
RESET_CHECKPOINT=true Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R
# Deletes all checkpoints and re-runs everything
```

### Monitoring Progress

Check progress using:

```bash
# View scenario checkpoint
head simulations/complete_data/complete_data_results/single_stage_cf/scenario_checkpoint.csv

# View method checkpoint for a scenario
head simulations/complete_data/complete_data_results/single_stage_cf/complete_n1000_r2_0p20/method_checkpoint.csv

# Check logs
tail logs/complete_data/single_stage_cf/*log
```

---

## High-Performance Computing (HPC) Usage

### Slurm (XSEDE/NERSC/Typical HPC)

Submit a job to run complete-data single-stage CF analysis:

```bash
#!/bin/bash
#SBATCH --job-name=cf_complete_data
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/complete_data/cf_single_stage_%A.out
#SBATCH --error=logs/complete_data/cf_single_stage_%A.err

cd /path/to/ml-robust-covariate-adjustment

# Restore environment
Rscript -e "renv::restore()"

# Run analysis
REPLICATES_TO_RUN=500 USE_PARALLEL=true \
  Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R
```

### Running on a Cluster

For multiple settings or procedures in parallel:

```bash
#!/bin/bash
# Submit multiple analysis jobs

for setting in setting_one setting_two setting_three setting_four; do
  sbatch --export=SETTING=$setting,REPLICATES_TO_RUN=500 \
    hpc_submit_missing_single_stage.sh
done
```

---

## Reproducibility Notes

### Seeds and Random Number Generation

All randomness is controlled via fixed seeds:

1. **Data generation:** `base_seed = 12345` + scenario-specific offsets, with `doRNG::registerDoRNG()` for parallel reproducibility
2. **Bootstrap samples:** `bootstrap_seed = 20260417 + replicate_id`
3. **Cross-fitting folds:** Deterministic via `rsample::vfold_cv()` with fixed seed

**Result:** Same input data + same analysis code → byte-for-byte identical outputs (on the same R/OS version).

### Platform and Version Dependencies

- **R version:** 4.5.0 or later (locked versions in `renv.lock`)
- **Operating system:** Linux, macOS, Windows (all supported; minor numeric differences possible across OS due to BLAS libraries)
- **Package versions:** Frozen in `renv.lock`; `renv::restore()` ensures exact versions

To verify reproducibility on your system:

```r
# In R, after renv::restore():
sessionInfo()  # Check R version and attached packages
renv::status()  # Confirm all packages match renv.lock
```

### Expected Runtime Estimates

| Analysis | Replicates | Single-core | 40-core Parallel |
|----------|:----------:|:-----------:|:---------------:|
| Single complete scenario (CF) | 500 | ~45 min | ~2 min |
| All complete-data scenarios (CF) | 500 × 20 | ~15 hrs | ~30 min |
| Single missing-outcome setting | 500 × 20 × 4 methods | ~24 hrs | ~1.5 hrs |
| All missing-outcome analyses | 500 × 20 × 4 settings × 4 procedures | ~14 days | ~1 day |

---

<!--
## Troubleshooting and FAQ

### Q: My analysis stopped unexpectedly. How do I resume?

**A:** Simply re-run the same command. The checkpoint system automatically detects completed replicates and resumes from the next unfinished one. No manual intervention needed.

### Q: I'm running on a machine with limited RAM. What can I do?

**A:** Reduce `REPLICATES_TO_RUN` and/or disable parallelization:

```bash
REPLICATES_TO_RUN=50 USE_PARALLEL=false Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R
```

### Q: The analysis is very slow. How can I speed it up?

**A:** Enable parallelization (default) and increase `cpus-per-task` if on HPC:

```bash
USE_PARALLEL=true Rscript simulations/complete_data/scripts/cf/single_stage_cf_results.R
```

If using HPC, request more cores in the Slurm header.

### Q: I got an error about package conflicts. What should I do?

**A:** Reset and restore your environment:

```bash
Rscript -e "renv::clean()"  # Remove unused packages
Rscript -e "renv::restore()"  # Reinstall locked versions
```

### Q: Can I modify the data generation or analysis code?

**A:** Yes. The code is fully provided for transparency. If you modify any code that affects randomness (e.g., seeds, RNG), results may differ from the paper.

### Q: How do I verify that my results match the paper?

**A:** After all analyses complete, check that:

1. All checkpoints show `status = "done"`
2. Aggregated results files exist and have expected columns
3. Results tables have expected dimensions (see paper appendix)

For detailed verification:

```r
# Load final results
results <- read.csv("simulations/complete_data/complete_data_results/single_stage_cf/single_stage_cf_results.csv")
dim(results)  # Should match paper table dimensions
summary(results$mean_bias)  # Compare to paper
```

---

## Citation and License

If you use this repository, please cite the paper:

```bibtex
@article{[author_year],
  title={Machine learning for robust covariate adjustment: enhancing efficiency and mitigating bias in randomized trials with missing outcomes},
  author={[authors]},
  journal={[journal]},
  year={[year]},
  doi={[doi]}
}
```

This repository is licensed under the [LICENSE](LICENSE) file (e.g., MIT, Apache 2.0).

---

## Support and Questions

For bugs, questions, or suggestions:

- **Issues:** Open a GitHub issue with a minimal reproducible example.
- **Discussions:** Use GitHub Discussions for questions about reproducing analyses.
- **Email:** [contact information, if applicable]

---

**Last Updated:** May 2026  
**Repository Version:** [version_tag]  
**Paper Status:** [Submitted / Accepted / Published]
-->
