# VE-BRB Fault Diagnosis

This repository contains the MATLAB implementation of the **Vibration-Enhanced Belief Rule Base (VE-BRB)** model proposed in the paper:  
*“An Interpretable Vibration-Enhanced BRB Model for Rolling Bearing Fault Diagnosis” (PLOS ONE, 2025).*

## 📘 Overview
The VE-BRB model integrates vibration-based dual-frequency feature extraction with a rule-based reasoning framework, optimized via P-CMA-ES.  
It is designed for interpretable and robust fault diagnosis of rolling bearings.

## 🧩 File Structure
- `main.m`: Main execution script for running VE-BRB.
- `wavelet.m`: Wavelet-based feature extraction function.
- `generator.m`: Data preprocessing and dataset construction.
- `fun_test.m`: Model testing and evaluation.
- `example_data/`: (Optional) Example vibration signal data.

## 🧠 Datasets
The experiments use publicly available datasets:
- Case Western Reserve University (CWRU) Bearing Dataset  
- Huazhong University of Science and Technology (HUST) Bearing Dataset  

## ▶️ Usage
Run the following in MATLAB:
```matlab
main
