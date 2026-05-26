# 🎓 Intern Performance Prediction

> A production-ready Machine Learning system to predict intern performance, flag at-risk individuals, and deliver actionable insights — all through a clean Streamlit dashboard.

---

> The app will be live at `https://intern-performance-checker.streamlit.app/`

## Overview

This project uses **Random Forest** and **XGBoost** to predict intern performance scores (0–100) based on real behavioral and task data. The model is trained once, saved to disk, and serves instant predictions whenever a new CSV is uploaded — no retraining required.

**Trained on:** 6,000 real intern records  
**Best model:** XGBoost — R² = 0.8764, MAE = 4.4 pts  
**Prediction time:** < 1 second per CSV upload

---

## Features

| Feature | Description |
|---|---|
| ⚡ Instant Predictions | Pre-trained model loads in milliseconds — no waiting |
| 📊 Interactive Overview | Score distributions, correlations, tier breakdowns |
| 🚨 Risk Flagging | Automatically classifies every intern into 4 risk tiers with alert reasons |
| 🔮 Live Predictor | Adjust sliders to predict any intern's score in real time |
| 📥 Export Reports | Download full or filtered intern reports as CSV |

---

## Project Structure

```
intern_project/
│
├── app.py                   # Streamlit dashboard (4 pages)
├── pipeline.py              # Core ML engine — training, engineering, flagging, prediction
├── pretrain.py              # One-time training script — creates pretrained_model.pkl
├── pretrained_model.pkl     # Saved trained model (already included — ready to use)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

> ✅ `pretrained_model.pkl` is already included in this project.  
> You do **not** need to run `pretrain.py` unless you want to retrain on new data.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the app

```bash
streamlit run app.py
```

### 3. Upload your CSV

Use the **sidebar file uploader** to upload your intern dataset. Results appear instantly.

---

## How It Works

```
Your CSV
   │
   ▼
pipeline.py — Feature Engineering (adds 5 composite features)
   │
   ▼
pretrained_model.pkl — XGBoost scores every intern (< 1 second)
   │
   ▼
Risk Tier assigned → Alerts generated → Dashboard rendered
```

**Want to retrain on new data?**

```bash
python pretrain.py --data your_new_data.csv
# Saves a new pretrained_model.pkl
# Then relaunch: streamlit run app.py
```

---

## Dashboard Pages

### 📊 Overview
- KPI cards: total interns, average score, count per tier
- Score distribution histogram with threshold lines
- Risk tier pie chart
- Feature importance bar chart (from trained model)
- Correlation chart — which features drive performance most
- Engineered feature scatter plot with per-category trendlines

### 🚨 Risk Flagging
- Full intern table color-coded by risk tier
- Filters: tier, name search, score range slider
- Alert reasons per intern (missed deadlines, high stress, low attendance, etc.)
- Alert frequency bar chart
- Score distribution box plot by tier

### 🔮 Predict Intern
- Input sliders for all 11 features
- Instant predicted score with visual tier ring
- Breakdown of 5 composite scores (productivity, reliability, engagement, burnout risk, quality)
- Actionable recommendation per tier

### 📥 Export
- Download full risk report (all interns)
- Download critical interns only
- Download at-risk interns (Critical + At Risk combined)
- Download excelling interns
- Model summary as JSON

---

## Risk Tiers

| Tier | Score | Recommended Action |
|---|---|---|
| 🔴 **Critical** | < 35 | Immediate 1-on-1 with manager. Review workload and support needs urgently. |
| 🟠 **At Risk** | 35 – 45 | Bi-weekly check-ins. Address top alert reasons. Set short-term goals. |
| 🟡 **On Track** | 45 – 65 | Monitor progress. Encourage engagement. Reduce stress if flagged. |
| 🟢 **Excelling** | > 65 | Assign stretch projects. Consider for mentorship or leadership track. |

---

## Model Performance

Trained with `RandomizedSearchCV` (20 iterations, 5-fold cross-validation):

| Model | MAE | RMSE | R² | CV R² |
|---|---|---|---|---|
| Random Forest | 5.861 | 7.351 | 0.7651 | 0.7587 |
| **XGBoost ✅** | **4.407** | **5.543** | **0.8764** | **0.8636** |
| Ensemble | 4.883 | 6.114 | 0.8375 | — |

**XGBoost selected** as the best model. Predictions are accurate within ~4.4 points on a 0–100 scale.

### Top predictive features (XGBoost)

1. `avg_task_completion_time` — 18.7%
2. `deadlines_missed` — 17.2%
3. `work_hours_per_day` — 15.2%
4. `feedback_rating` — 10.7%
5. `task_completion_rate` — 9.9%

---

## Dataset Format

Your CSV must contain these columns:

| Column | Type | Description |
|---|---|---|
| `intern_id` | int | Unique intern identifier |
| `intern_name` | str | Full name |
| `task_completion_rate` | int | % of assigned tasks completed (0–100) |
| `avg_task_completion_time` | float | Average hours to complete a task |
| `attendance_percentage` | int | % attendance (0–100) |
| `feedback_rating` | float | Mentor/manager rating (1.0–5.0) |
| `deadlines_missed` | int | Number of missed deadlines |
| `github_commits` | int | Total code commits |
| `communication_score` | int | Communication rating (1–10) |
| `learning_speed` | int | Learning speed rating (1–10) |
| `meeting_participation` | int | Meeting participation rating (1–10) |
| `work_hours_per_day` | float | Average daily work hours |
| `stress_level` | int | Self-reported stress level (1–10) |
| `final_performance_score` | float | Ground truth score (0–100) |
| `performance_category` | str | `Excellent` / `Average` / `Needs Improvement` |

---
