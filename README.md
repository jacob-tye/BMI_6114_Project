# BMI_6114_Project
## 🧬 Predicting Survival Time in Lung Cancer (BMI 6114 Final Project)
This project explores the application of traditional machine learning and deep learning models to predict survival time in lung cancer patients. Using clinical and genomic data from TCGA, we implemented multiple pipelines, including feature preprocessing, dimensionality reduction with autoencoders, and regression modeling using Random Forests, Gradient Boosting, and deep neural networks.

## 🗂 Project Structure
```
├── README.md
└── analysis/
    ├── 00_data_prep.ipynb              # Preprocessing and feature engineering
    ├── 01_ml_analysis_gbm.ipynb        # Gradient Boosting Models (unscaled and scaled)
    ├── 01_ml_analysis_rf.ipynb         # Random Forest Regressor Models (unscaled)
    ├── 01_ml_analysis_rf_scaled.ipynb         # Random Forest Regressor Models (scaled)
    ├── 02_autoencoder.ipynb            # Autoencoder for dimensionality reduction
    ├── 03_dl_analysis.ipynb            # Deep learning with autoencoder (unscaled)
    ├── 04_dl_analysis_no_ae.ipynb      # Deep learning without autoencoder (unscaled)
    ├── 05_dl_analysis_scaled.ipynb     # Deep learning with autoencoder (log-scaled target)
    ├── 06_dl_analysis_no_ae_scaled.ipynb # Deep learning without autoencoder (log-scaled)
    ├── results/                        # Best models and paremeters for use in later steps
    ├── optuna_logs/                    # Optuna logs and checkpoints for hyperparameter tuning
    └── logs/                           # Lightning logs and checkpoints for all model trials
```
## 🧪 Methods Overview
### Data Source
TCGA LUAD and LUSC samples

Downloaded and filtered via [cBioPortal](https://www.cbioportal.org/study/summary?id=nsclc_tcga_broad_2016)

Features Used
Smoking history and pack years (clinical)

Mutation and copy number variation (genomic)

One-hot encoded + standardized features

### Modeling Approaches

Category	Models Implemented	Notes
ML	Random Forest, Gradient Boosting	Hyperparameter tuning via Optuna
Dimensionality	Autoencoder	Latent compression to reduce feature space
DL	Feedforward Neural Networks	With/without autoencoder; raw vs log targets
Experiment Logs	PyTorch Lightning logging to logs/	Models saved with metrics.csv and hparams.yaml per run

### 📊 Results Snapshot
🟢 Best ML Model: Random Forest on unscaled targets (R² = 0.416)

🔴 Deep Learning Models: All had R² < 0 on test set due to overfitting and dataset size

🔁 Overfitting Test: Scaled NN models showed learnability on training set with R ≈ 0.78

📄 Report
A full write-up of our methods, results, limitations, and visualizations is available in final_report.docx.
