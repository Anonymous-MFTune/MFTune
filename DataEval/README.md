# DataEval

---

## Overview

`DataEval` serves as the **analysis companion** to the main [MFTune](https://github.com/yourusername/MFTune) repository. While the main project implements the tuning framework and evaluation pipeline, this repository focuses on **statistical analysis and figures/tables generation**.

It supports the investigation of the following **research questions (RQs)**:

| Research Question | Focus                          | Corresponding Files                                    |
|--------------------|--------------------------------|--------------------------------------------------------|
| **RQ1** | Benefits of Imperfect-Fidelity | `rq1_rq2_efficiency_info.py`, `rq1_efficiency_plot.py` |
| **RQ2** | Effectiveness and Efficiency   | `rq2_effectiveness_sk.py`, `rq2_efficiency_plot.py`    |
| **RQ3** | Sensitivity to α               | `rq3_sensitivity_sk.py`                                |
| **RQ4** | Ablation study                 | `rq4_ablation_sk.py`                                   |

---

## Directory Structure

```
MFTune-statistics/
│
├── config/                      # Configuration files 
│
├── discussion/                  # Supporting discussion figures and evolution processes
│   ├── fidelity_evo_process
│   └── mf_analysis
│
├── results/                     # Raw and processed results across all systems and tuners
│   ├── clang/
│   ├── gcc/
│   ├── httpd/
│   ├── mysql/
│   ├── postgresql/
│   └── tomcat/
│       ├── flash/ bohb/ dehb/ ga/ hebo/ smac/ hyperband/ priorband/ promise/ bestconfig # Compared algorithms (single- or multi-fidelity)
│       ├── MFTune-a1 ~ MFTune-a9  # Sensitivity results (α = 0.1, 0.3, …, 0.9)
│       ├── MFTune-I, MFTune-II    # Ablation variants
│
├── RQ1/                         # Figures and processed data for RQ1
├── RQ2/                         # Figures and processed data for RQ2
├── RQ3/                         # Figures and processed data for RQ3
├── RQ4/                         # Figures and processed data for RQ4
│
├── utils/                       # Utility functions for analysis and plotting
│
├── rq1_rq2_efficiency_info.py   # Data collection and aggregation for RQ1
├── rq1_efficiency_plot.py       # Efficiency visualization
├── rq2_effectiveness_sk.py      # Scott–Knott ESD analysis for effectiveness
├── rq2_efficiency_plot.py       # Efficiency visualization for RQ2
├── rq2_efficiency_post.py       # Efficiency visualization for RQ2 results
├── rq3_sensitivity_sk.py        # Sensitivity analysis of α
└── rq4_ablation_sk.py           # Ablation study of MFTune variants
```

---

## Data Description

All experimental results are organized under the `results/` folder, covering  
**six configurable systems** and **multiple tuning algorithms**:

- **Systems:**  
  `MySQL`, `PostgreSQL`, `Tomcat`, `HTTPD`, `GCC`, and `Clang`.

- **Tuners:**  
  - **SOTA:** `FLASH`, `BOHB`, `DEHB`, `SMAC`, `HEBO`, `GA`, `HyperBand`, `PriorBand`, `Promise`, and `BestConfig` 
  - **Proposed Variants:**  
    - `MFTune-a1` ~ `MFTune-a9`: sensitivity analysis with α = 0.1, 0.3, …, 0.9  
    - `MFTune-I` and `MFTune-II`: ablation experiments analyzing different design components

Each folder includes result files (e.g., `.csv`) representing performance evaluations, which are processed by the scripts in the `RQ*/` directories to generate the paper figures.

---

[//]: # (## Environment Setup)

[//]: # ()
[//]: # (```bash)

[//]: # (# Clone the repository)

[//]: # (git clone https://github.com/***.git)

[//]: # (cd MFTune-statistics)

[//]: # ()
[//]: # (# Create and activate a virtual environment)

[//]: # (python3 -m venv venv)

[//]: # (source venv/bin/activate       # &#40;Mac/Linux&#41;)

[//]: # (# venv\Scripts\activate      # &#40;Windows&#41;)

[//]: # ()
[//]: # (# Install dependencies)

[//]: # (pip install -r requirements.txt)

[//]: # (```)

### requirements.txt

```txt
matplotlib==3.5.3
numpy==1.24.4
pandas==2.2.3
rpy2==3.5.17
scikit_learn==1.5.1
scipy==1.10.1
```

---

## Usage

You can reproduce each figure or table in the paper by executing the corresponding script:

[//]: # (```bash)

[//]: # (# RQ1: Efficiency analysis)

[//]: # (python rq1_efficiency_info.py)

[//]: # (python rq1_efficiency_plot.py)

[//]: # ()
[//]: # (# RQ2: Effectiveness and statistical ranking)

[//]: # (python rq2_effectiveness_sk.py)

[//]: # (python rq2_efficiency_plot.py)

[//]: # ()
[//]: # (# RQ3: Sensitivity analysis)

[//]: # (python rq3_sensitivity_sk.py)

[//]: # ()
[//]: # (# RQ4: Ablation study)

[//]: # (python rq4_ablation_sk.py)

[//]: # (```)

The generated figures and summary tables will be saved in the corresponding `RQ*/` folder.

---

[//]: # (## Citation)

[//]: # ()
[//]: # (If you use this repository, please cite the corresponding paper:)

[//]: # ()
[//]: # (```)

[//]: # (@article{MFTune2025,)

[//]: # (  title={MFTune: Multi-Fidelity Configuration Tuning for Complex Software Systems},)

[//]: # (  author={...},)

[//]: # (  year={2025},)

[//]: # (  journal={...})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (---)

## License

[//]: # (This project is released under the [MIT License]&#40;LICENSE&#41;.)
