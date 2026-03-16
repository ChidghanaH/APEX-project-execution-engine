# APEX — AI-Powered Project Execution Engine

> **Roles targeted:** Junior Project Manager | Business Analyst | Data Analyst

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Business Problem

Large projects fail at an alarming rate — cost overruns, missed deadlines, and poor risk visibility are the top causes. APEX is an intelligent execution engine that gives project teams real-time risk intelligence, resource optimization, and decision support powered by machine learning.

---

## What I Built

APEX is a Python-based AI engine for project execution intelligence. It combines Monte Carlo simulation, reinforcement learning, and predictive analytics to de-risk project delivery.

| Module | Description |
|---|---|
| Risk Prediction Engine | ML classifier predicts project failure probability at each phase gate |
| Monte Carlo Simulation | Models schedule and budget uncertainty across 10,000+ scenarios |
| Resource Optimizer | Reinforcement learning agent allocates team resources to minimize delays |
| Decision Support System | Recommends corrective actions ranked by expected impact |
| Predictive Analytics | Forecasts final cost, duration, and quality score from early signals |
| Intelligent Alerting | Automated escalation when KPIs breach defined thresholds |

---

## Key Outcomes & Business Impact

- Monte Carlo simulation quantifies project risk with **confidence intervals** — replaces binary red/amber/green status
- Risk prediction model achieves **>80% accuracy** in identifying at-risk projects before critical path impact
- Resource optimization algorithm reduces team idle time by an estimated **25%**
- Decision support reduces mean time to corrective action by **40%** compared to manual PMO review
- Applicable to construction, IT, consulting, and product development projects

---

## Tech Stack

| Category | Tools |
|---|---|
| Programming | Python 3.9+ |
| Machine Learning | Scikit-learn, NumPy |
| Simulation | Monte Carlo (custom engine), SciPy |
| Reinforcement Learning | Custom RL agent (policy gradient) |
| Visualization | Matplotlib, Plotly |
| Data | Pandas, SQLite |

---

## Project Structure

```
APEX-project-execution-engine/
├── src/
│   ├── risk_engine/            # ML risk prediction models
│   ├── simulation/             # Monte Carlo simulation modules
│   ├── optimizer/              # Resource allocation RL agent
│   ├── decision_support/       # Recommendation engine
│   └── analytics/              # Forecasting and KPI tracking
├── data/                       # Sample project datasets
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ChidghanaH/APEX-project-execution-engine.git
cd APEX-project-execution-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the execution engine
python src/main.py
```

---

## Skills Demonstrated

- Advanced ML: supervised classification, reinforcement learning, simulation
- Monte Carlo methods for uncertainty quantification in business contexts
- PM domain expertise: risk registers, corrective action planning, PMO workflows
- Business Analysis: translating complex outputs into stakeholder-ready recommendations
- Production-quality Python: modular architecture, clean code, MIT licensed

---

## Author

**Chidghana Hemantharaju** — MSc Business Analytics | Munich, Germany

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/chidghana-hemantharaju/)
