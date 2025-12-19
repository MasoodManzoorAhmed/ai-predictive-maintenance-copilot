# 🛠️ AI Predictive Maintenance Copilot  
**NASA CMAPSS (FD001–FD004) | Classical ML + Deep Learning + GenAI (RAG) | Production-Grade MLOps**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Google Cloud Run](https://img.shields.io/badge/Deployed-Google%20Cloud%20Run-green)
![RAG](https://img.shields.io/badge/GenAI-RAG%20Copilot-purple)

---

## 📌 Project Overview

This project is a **full-scale, production-style AI Predictive Maintenance system** built on the **NASA CMAPSS turbofan engine datasets (FD001–FD004)**.

It goes **far beyond notebooks**, delivering:
- Robust **baseline ML benchmarking**
- Advanced **deep learning sequence models**
- A **unified inference engine**
- A **FastAPI backend**
- A **Streamlit decision-support dashboard**
- A **GenAI-powered Maintenance Copilot (RAG)**
- **Dockerized CI/CD deployment on Google Cloud Run**

This is designed to reflect **real-world industrial AI systems**, not academic demos.

---

## 🎯 Business Problem

Unexpected engine failures lead to:
- Costly downtime
- Safety risks
- Inefficient maintenance schedules

### Objective
1. **Predict Remaining Useful Life (RUL)** accurately
2. **Assess operational risk** (early / mid / critical)
3. **Assist engineers & managers** with explainable, knowledge-backed guidance via AI Copilot

---

## 📊 Datasets Used

NASA CMAPSS:
- **FD001** – Single operating condition, single fault
- **FD002** – Multiple operating conditions, single fault
- **FD003** – Single operating condition, multiple faults
- **FD004** – Multiple operating conditions, multiple faults (most complex)

Each dataset has its **own optimized pipeline**, but inference is **fully unified**.

---

## 🧠 Modeling Strategy (What Was Actually Built)

### 1️⃣ Classical Baseline Models (Tabular)
Used to establish **strong, interpretable benchmarks** before deep learning.

- **RandomForest Regressor**
- **XGBoost Regressor (lightweight, optional)**

**Purpose:**
- Sanity-check feature engineering
- Measure how much value deep learning truly adds

---

### 2️⃣ Deep Learning Models (Sequence-Based)

#### Baseline Architectures
- **Baseline LSTM**
- **Baseline GRU**

Trained on:
- `seq_len = 30` (short-term degradation)
- `seq_len = 100` (long-term degradation)

---

#### Advanced / Tuned Architectures
- **Deep LSTM (stacked, dropout-regularized)**
- **Deep GRU**
- **Tuned LSTM with scaled RUL target**

**Key enhancements:**
- Longer temporal context
- Dropout regularization
- Learning-rate scheduling
- Early stopping
- RUL normalization + inverse scaling

---

### 3️⃣ Feature Engineering (Production-Grade)

Applied consistently across all FD datasets:

- Removal of near-constant sensors
- **Rolling statistics** (mean & std over 3, 5 cycles)
- **Delta features** (cycle-to-cycle change)
- NaN / Inf cleaning
- MinMax scaling
- Leakage-safe **engine-wise train/validation split**

---

## 🧠 Unified Inference Engine

A **single production inference pipeline** supports **FD001–FD004**:

- Loads correct model + scalers via FD config
- Applies identical feature engineering
- Builds final sequences
- Predicts RUL
- Applies NASA-style calibration
- Outputs:
  - RUL
  - Risk band
  - Risk score

This ensures:
- 🔁 Reproducibility
- 🧪 Consistency between training & deployment
- 🏭 Real-world readiness

---

## 🏗️ System Architecture

```text
User (Browser / Mobile)
        │
        ▼
┌──────────────────────────────┐
│ Streamlit Frontend (Cloud Run)│
│ • RUL Prediction              │
│ • Analytics                   │
│ • Maintenance Copilot         │
└───────────────▲──────────────┘
                │ REST
                ▼
┌──────────────────────────────┐
│ FastAPI Backend (Cloud Run)   │
│ • /predict/{fd}               │
│ • /single/predict              │
│ • /copilot/query               │
└───────────────▲──────────────┘
                │
     ┌──────────┴──────────┐
     ▼                     ▼
┌───────────────┐   ┌──────────────────┐
│ Inference      │   │ RAG Copilot       │
│ Engine         │   │ (FAISS + LLM)     │
└───────────────┘   └──────────────────┘

```
## 🤖 GenAI Maintenance Copilot (RAG)

User Question
     │
     ▼
Streamlit Copilot UI
     │
     ▼
FastAPI Copilot Endpoint
     │
     ▼
FAISS Vector Search (PDF Manuals)
     │
     ▼
LLM (OpenRouter)
     │
     ▼
Context-Aware Answer + Sources

## ☁️ Deployment

Dockerized services
Google Cloud Build for CI
Google Cloud Run for serverless deployment
Auto-scaling, HTTPS, IAM-ready

## 🧪 What This Project Demonstrates

✅ Classical ML benchmarking
✅ Deep learning for time-series RUL
✅ Robust feature engineering
✅ Unified inference design
✅ Production API + UI
✅ GenAI (RAG) integration
✅ Cloud-native deployment


## 👤 Author

Masood Manzoor Ahmed
Machine Learning / AI Engineer
MSc Data Science — University of Greenwich

## 📄 License

MIT License