# MFTune-compiler: Compiler Tuning Module

This module implements the MFTune framework for **GCC** and **Clang** compilers.  

---

## Documentation

```
MFTune-compiler/
├── config/                 # Configuration space (GCC/Clang flags)
├── fidelity_factors/       # Fidelity space
├── logs/                   # Runtime logs and tuning outputs
├── params_setup/           # General parameter setup
├── systems/                # GCC/Clang system controller classes
├── tuner/                  # Tuning algorithms (GA, FLASH, SMAC, etc.)
├── utils/                  # Utility tools
│
├── auto_runner.py          # Main entry for full tuning experiments
├── docker-compose.yml      # Compiler + app_tuning containers
├── dockerfile              # Image for compiler tuning environment
├── main.py                 # main
│
├── run_gcc.sh              # Run GCC tuning
├── run_clang.sh            # Run Clang tuning
├── run_sampler.sh          # Run sampling experiments
│
├── clang_help.txt          # Reference for Clang flags
└── requirements.txt        # Python dependencies
```

---

## Prerequisites

### **Python 3.9+**

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### **Docker & Docker Compose**

```bash
sudo apt-get install docker.io docker-compose -y
```

---

### **tmux (Recommended)**

```bash
sudo apt-get install tmux -y
tmux new -s mftune
```

---

# Running GCC Tuning

First of all, you need to download the following images before running:


- Gcc: [Zenodo-gcc](https://zenodo.org/records/17802608)

- Clang: [Zenodo-clang](https://zenodo.org/records/17802608)

Then, load the image locally by:

```bash
docker load -i gcc.tar
docker load -i clang.tar
```

Run:

```bash
sudo bash run_gcc.sh
```

This script:

1. Starts GCC + tuning container  
2. Generates workloads via Csmith  
3. Runs full tuning pipeline  
4. Stores logs in `experimental_results/` and `logs/`

---

## 🔎 Image Build Options

Inside `run_gcc.sh`:

```bash
docker-compose up --build -d app_tuning $compiler_service
# docker-compose up -d app_tuning $compiler_service
```

### ✔ Build Locally

```bash
docker-compose up --build -d app_tuning gcc
```

### ✔ Use Pre-built Image (Recommended)

The pre-built image can be accessed via:

- app_tuning: [Zenodo-app_tuning](https://zenodo.org/records/17802608)

Then, load the image locally by:

```bash
docker load -i app_tuning.tar
```

---

# Running Clang Tuning

```bash
sudo bash run_clang.sh
```

---

