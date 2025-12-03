# MFTune-db: Database Tuning Module

This module implements the MFTune framework for **MySQL** and **PostgreSQL**.  

---

## Documentation

```
MFTune-db/
├── config/                 # Configuration space
├── fidelity_factors/       # Fidelity space
├── lua/                    # LUA scripts for Sysbench (custom workloads)
├── params_setup/           # General parameters settings
├── systems/                # MySQL/PostgreSQL system controller classes
├── tuner/                  # Tuning algorithms (GA, FLASH, SMAC, etc.)
├── utils/                  # Tools
├── workload/               # Sysbench / OLTPBench workload controllers
│
├── auto_runner.py          # Main entry for full tuning experiments
├── auto_runner_local_test.py # Local quick-test runner
├── docker-compose.yml      # Database + App container setup
├── dockerfile              # Image for the tuning app
├── main.py                 # main entry
│
├── run_mysql.sh            # Script to launch MySQL tuning
├── run_postgresql.sh       # Script to launch PostgreSQL tuning
├── run_sampler.sh          # Script for fidelity sampling evaluation
├── run_analyser.sh         # Script for analyzing stored data
│
└── requirements.txt        # Python dependencies
```

---

## Prerequisites

### **1. Python**

This project requires:

```
Python 3.9+
```

Create virtual environment and install dependencies:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### **2. Docker & Docker Compose**

Install:

```bash
sudo apt-get install docker.io docker-compose -y
```

---

### **3. tmux (Strongly Recommended)**

To avoid interruption due to SSH disconnection:

```bash
sudo apt-get install tmux -y
tmux new -s mftune
```

---

# Running MySQL Tuning

Inside tmux:

```bash
sudo bash run_mysql.sh
```

This script will:

1. Initialize logs  
2. Start MySQL + app_tuning containers  
3. Run full tuning pipeline  
4. Store results/logs in `experimental_results/` and `logs/`  

---

## 🔎 Note About Image Build Options

Inside `run_mysql.sh`, you will find:

```bash
docker-compose up --build -d app_tuning $db_service >> "$run_log" 2>&1
# docker-compose up -d app_tuning $db_service >> "$run_log" 2>&1
```

### ✔ Option A: **Build locally **

```
docker-compose up --build -d app_tuning mysql
```

Uses the included `dockerfile` to set up Python, Sysbench, OLTPBench.

### ✔ Option B: **Use our pre-built image (strongly recommended)**

```
docker-compose up -d app_tuning mysql
```

If you use Option B, you need to download the following images before running:

```
image for tuning logic: 
image for mysql:
image for postgresql:
```

Then, load the image locally by:

```bash
docker load -i app_tuning.tar
docker load -i mysql.tar
docker load -i postgresql.tar
```

---

# Running PostgreSQL Tuning

Same as MySQL, but run:

```bash
sudo bash run_postgresql.sh
```

---



