# Less is More: Tuning Configurable Systems with Imperfect Fidelity

This repository contains the full implementation of MFTune, a multi-fidelity configuration tuning framework evaluated across diverse real-world software systems. The repository includes all system controllers, fidelity generators, tuning algorithms, experimental data, and analysis scripts used in the paper.


## Subject Systems

| System        | Domain     | Performance        | 
|---------------|------------|--------------------|
| MySQL         | Database   | Throughput (tps)   | 
| PostgreSQL    | Database   | Throughput (tps)   | 
| Tomcat        | Web server | Throughput (rps)   |
| HTTPD         | Web server | Throughput (rps)   | 
| GCC           | Compiler   | Runtime (s)        | 
| Clang         | Compiler   | Runtime (s)        | 

## Compared Tuning Algorithms

| Tuner        | Fidelity        | Strategy      | Domain        | Year |
|--------------|-----------------|---------------|---------------|------|
| PromiseTune  | Single-fidelity | Model-based   | Configuration | 2026 |
| HEBO         | Single-fidelity | Model-based   | General       | 2022 |
| GA           | Single-fidelity | Model-free    | General       | 2020 |
| FLASH        | Single-fidelity | Model-based   | Configuration | 2018 |
| BestConfig   | Single-fidelity | Model-free    | Configuration | 2017 |
| SMAC         | Single-fidelity | Model-free    | General       | 2011 |
| PriorBand    | Multi-fidelity  | Model-free    | General       | 2023 |
| DEHB         | Multi-fidelity  | Model-free    | General       | 2021 |
| BOHB         | Multi-fidelity  | Model-based   | General       | 2018 |
| Hyperband    | Multi-fidelity  | Model-free    | General       | 2018 |

## Repository Structure

- DataEval/: raw data and analysis scripts  
- MFTune-compiler/: GCC/Clang tuning  
- MFTune-db/: MySQL/PostgreSQL tuning  
- MFTune-web/: Tomcat/HTTPD tuning  

## Module Documentation

Each module provides detailed setup and usage instructions:

- MFTune-db/README.md  
- MFTune-web/README.md  
- MFTune-compiler/README.md  
- DataEval/README.md  

## Citation


## License

