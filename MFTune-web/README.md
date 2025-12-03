# MFTune-web: Web Server Tuning Module

This module implements the MFTune framework for **Tomcat** and **HTTPD**.

---

## Documentation

```
MFTune-web/
├── config/                 # Configuration space for Tomcat/HTTPD
├── fidelity_factors/       # Fidelity space
├── logs/                   # Runtime logs and container outputs
├── params_setup/           # General parameter setting
├── systems/                # Tomcat/HTTPD system controller classes
├── tempfiles/              # Temporary files for modifying configuration
├── tuner/                  # Tuning algorithms (GA, FLASH, SMAC, etc.)
├── utils/                  # Utility functions
│
├── auto_runner.py          # Main entry for full tuning experiments
├── docker-compose.yml      # Web server + App container setup
├── dockerfile              # Image for the tuning app
├── main.py                 # Main entry for configuration tuning
│
├── run_tomcat.sh           # Script to launch Tomcat tuning
├── run_httpd.sh            # Script to launch HTTPD tuning
├── run_sampler.sh          # Script for fidelity sampling evaluation
│
└── requirements.txt        # Python dependencies
```

---

## Prerequisites

### **1. Python**

Requires:

```
Python 3.9+
```

Install dependencies:

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

To avoid termination during SSH disconnection:

```bash
sudo apt-get install tmux -y
tmux new -s mftune
```

---

# Running Tomcat Tuning

Before running tuning, download and load the images:

- Tomcat workload: [Zenodo-tomcat](https://zenodo.org/records/17802608)  
- HTTPD workload: [Zenodo-httpd](https://zenodo.org/records/17802608)  

```bash
docker load -i tomcat_sampling_10.1.34.tar
docker load -i httpd-sampling_2.4.63.tar
```

Inside tmux:

```bash
sudo bash run_tomcat.sh
```

This script will:

1. Initialize logs  
2. Start Tomcat + app_tuning containers  
3. Launch full tuning pipeline  
4. Store results in `experimental_results/` and `logs/`

---

## 🔎 Note About Image Build Options

Inside `run_tomcat.sh`:

```bash
docker-compose up --build -d app_tuning $web_service >> "$run_log" 2>&1
# docker-compose up -d app_tuning $web_service >> "$run_log" 2>&1
```

### ✔ Option A: **Build locally**

```bash
docker-compose up --build -d app_tuning tomcat
```

### ✔ Option B: **Use pre-built image (recommended)**

```bash
docker-compose up -d app_tuning tomcat
```

The pre-built image can be accessed via:

- app_tuning: [Zenodo-app_tuning](https://zenodo.org/uploads/17802608)

Then, load the image locally by:

```bash
docker load -i app_tuning.tar
```

---

# Running HTTPD Tuning

```bash
sudo bash run_httpd.sh
```

---



