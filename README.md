# Applied Machine Learning for Insurance Cost Prediction and Student Segmentation

## Project overview

This project contains two independent machine learning studies implemented in executed Jupyter notebooks. The first predicts medical insurance charges with supervised regression. The second explores student groups with unsupervised K-Means clustering. The methods are intended to be clear, reproducible, and suitable for an undergraduate artificial intelligence project.

## Objectives

- Prepare and inspect two real public datasets.
- Compare a Decision Tree Regressor with Ridge Regression using leakage-safe pipelines and five-fold cross-validation.
- Evaluate regression performance with MAE, RMSE, and R².
- Segment students with K-Means using non-grade academic-context, demographic, and behavioral features.
- compare cluster counts with the elbow method and silhouette score.
- Interpret clusters carefully without treating them as confirmed student categories.

## Project structure

```text
Applied-ML-Regression-Clustering/
├── data/
├── 01_insurance_regression.ipynb
├── 02_student_clustering.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Datasets

### Medical Cost Personal Dataset

- File: `data/insurance.csv`
- Records used after cleaning: 1,337
- Variables: age, sex, BMI, children, smoking status, region, and charges
- Target: `charges`
- Original dataset listing: [Medical Cost Personal Datasets on Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Downloaded public copy: [j-lee-26/us_medical_insurance_costs](https://github.com/j-lee-26/us_medical_insurance_costs/blob/main/insurance.csv)
- Direct file URL: [insurance.csv](https://raw.githubusercontent.com/j-lee-26/us_medical_insurance_costs/main/insurance.csv)
- Repository license: [MIT License](https://github.com/j-lee-26/us_medical_insurance_costs/blob/main/LICENSE)

The raw file had no missing values and one exact duplicate, which was removed before modeling. High charge observations identified by the IQR rule were retained because they were plausible values, were strongly associated with smoking, and were part of the outcome the model must predict.

### UCI Student Performance Dataset

- File used: `data/student-performance-uci/student-mat.csv`
- Records: 395 mathematics students
- Source: [UCI Machine Learning Repository: Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- DOI: [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T)
- Download URL: [UCI archive](https://archive.ics.uci.edu/static/public/320/student+performance.zip)
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Citation: Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository.

The dataset had no missing values or duplicate rows. The mathematics course file was used. `G1`, `G2`, and `G3` were excluded from clustering so academic grades did not define the groups. `G3` was used only afterward as an external interpretation variable.

## Technologies

- Python 3.11
- Jupyter and nbconvert
- pandas and NumPy
- scikit-learn
- Matplotlib and Seaborn

## Regression methodology

The insurance notebook performs data inspection, duplicate removal, exploratory plots, and an outlier review. It uses an 80/20 train/test split with `random_state=42`.

Numeric variables are median-imputed and standardized. Categorical variables are mode-imputed and one-hot encoded. These operations are contained in scikit-learn pipelines, so preprocessing is fitted separately inside each training fold. GridSearchCV uses five-fold cross-validation and negative RMSE to tune compact parameter grids.

### Regression results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Decision Tree Regressor | $2,613.67 | $4,303.54 | 0.8992 |
| Ridge Regression | $4,185.41 | $5,964.28 | 0.8064 |

The tuned Decision Tree was the best model on the held-out test set. Its selected parameters were `max_depth=4`, `min_samples_leaf=5`, and `min_samples_split=2`. The tree performed better because it could represent non-linear relationships and interactions, particularly the large difference associated with smoking status. Ridge selected `alpha=1.0`.

## Clustering methodology

The clustering notebook uses selected background, study, support, social, health, and attendance variables. Numeric variables are median-imputed and standardized. Categorical variables are mode-imputed, one-hot encoded, and scaled. K-Means models for K values from 2 through 8 use `random_state=42` and 20 initializations.

### Elbow and silhouette results

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 12,687.1 | 0.1695 |
| 3 | 11,898.5 | 0.1241 |
| 4 | 11,083.9 | 0.0938 |
| 5 | 10,438.3 | 0.0972 |
| 6 | 9,844.9 | 0.0997 |
| 7 | 9,446.7 | 0.0928 |
| 8 | 9,238.5 | 0.0765 |

K=2 was selected because it had the highest silhouette score among the tested values. The score is modest, so the groups overlap and should be treated as exploratory patterns.

### Student group profiles

- Cluster 0 contained 82 students. Relative to the complete sample, it showed lower study time, more past failures, higher weekend alcohol use, older average age, longer travel time, and lower rates of internet access, family support, and higher-education intention. Its mean final grade was 8.93, used only for interpretation.
- Cluster 1 contained 313 students. It showed higher study time, fewer past failures, lower weekend alcohol use, stronger access and support indicators, and a higher rate of higher-education intention. Its mean final grade was 10.81, used only for interpretation.

These labels describe differences within this dataset. They do not establish fixed, causal, or real-world student categories.

## Key findings

- The Decision Tree produced lower errors and a higher R² than Ridge Regression on the same held-out test set.
- Smoking status was visually associated with the largest charge differences, while age and BMI also showed useful relationships.
- The strongest tested clustering solution used two groups, but its silhouette score of 0.1695 indicates limited separation.
- Student grades were not used to construct the clusters, preventing performance from directly defining the segmentation.

## Installation and execution

1. Create and activate a Python virtual environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open and run `01_insurance_regression.ipynb`, then `02_student_clustering.ipynb`.

To verify both notebooks non-interactively from the project directory:

```bash
jupyter nbconvert --to notebook --execute 01_insurance_regression.ipynb --output 01_insurance_regression.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute 02_student_clustering.ipynb --output 02_student_clustering.ipynb --ExecutePreprocessor.timeout=600
```

Both notebooks in this repository are saved with their actual execution outputs and plots.
