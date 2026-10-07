# 🚢 Titanic Survival Prediction

> An end-to-end Machine Learning classification project that predicts whether a passenger survived the Titanic disaster using Logistic Regression.

## 📌 Project Overview

The **Titanic Survival Prediction** project is an end-to-end Machine Learning classification project built using the famous Titanic dataset.

The goal of this project is to analyze passenger information such as age, gender, passenger class, fare, family size, and embarkation point, and use these features to predict whether a passenger survived the Titanic disaster.

This project follows a complete Machine Learning workflow, starting from data exploration and preprocessing to model training, evaluation, and cross-validation.

### Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Cross Validation
```

---

## 🎯 Project Objective

The main objective of this project is to build a Machine Learning classification model that can predict whether a Titanic passenger survived.

### Target Variable

The target variable is:

```text
Survived
```

Where:

```text
0 → Did not survive
1 → Survived
```

### Input Features

The model uses passenger-related features such as:

- Passenger Class
- Sex
- Age
- Number of Siblings/Spouses
- Number of Parents/Children
- Fare
- Embarkation Port
- Family Size
- Is Alone

---

## 🧠 Problem Statement

The Titanic dataset contains information about passengers who travelled on the RMS Titanic.

Using this historical passenger data, the objective is to train a classification model that learns patterns associated with passenger survival.

The project demonstrates how Machine Learning can be used to:

- Understand real-world datasets
- Clean incomplete data
- Engineer useful features
- Convert categorical data into numerical form
- Scale numerical features
- Train a classification model
- Evaluate model performance
- Validate model consistency

---

# 🛠️ Technologies & Tools

| Technology / Tool | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |
| Jupyter Notebook | Development Environment |
| Google Colab | Model Development |
| VS Code | Project Organization |
| Git | Version Control |
| GitHub | Project Hosting |

---

# 📊 Dataset

This project uses the **Titanic Dataset** from Kaggle.

The dataset contains passenger information including demographic details, ticket information, passenger class, fare, family information, and survival status.

## Dataset Features

| Column | Description |
|---|---|
| PassengerId | Unique passenger identifier |
| Survived | Survival status |
| Pclass | Passenger class |
| Name | Passenger name |
| Sex | Passenger gender |
| Age | Passenger age |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation |

---

# 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed before model training to understand the structure, quality, and patterns within the dataset.

## Dataset Inspection

The following operations were performed:

- Dataset shape
- Column names
- Data types
- Statistical summary
- Missing value analysis
- Duplicate record analysis

Example:

```python
df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

---

# 📈 Data Visualization

Several visualizations were created to understand the relationship between passenger characteristics and survival.

### Visualizations Included

- Survival Distribution
- Gender vs Survival
- Passenger Class vs Survival
- Age Distribution
- Fare Distribution
- Fare vs Survival Box Plot

These visualizations helped identify patterns and relationships before applying Machine Learning algorithms.

---

# 🧹 Data Preprocessing

The Titanic dataset contains missing values and columns that are not directly required for prediction.

Therefore, preprocessing was performed before model training.

## Missing Value Handling

### Age

The `Age` column contained missing values.

Missing values were replaced using the median age.

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
```

Median imputation was selected because it is less sensitive to extreme values than the mean.

### Embarked

The `Embarked` column contained missing values.

Missing values were replaced using the most frequently occurring value.

```python
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
```

## Cabin

The `Cabin` column contained a large amount of missing information.

Therefore, it was removed from the dataset.

```python
df = df.drop(columns=["Cabin"])
```

## Removing Unnecessary Columns

The following columns were removed:

```text
PassengerId
Name
Ticket
```

These columns were not directly required for the selected Machine Learning approach.

```python
df = df.drop(
    columns=["PassengerId", "Name", "Ticket"]
)
```

---

# ⚙️ Feature Engineering

Feature engineering was performed to create additional information that could help the model.

Two new features were created:

```text
FamilySize
IsAlone
```

## 👨‍👩‍👧 Family Size

Family size was calculated using the number of siblings/spouses and parents/children travelling with the passenger.

```python
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
```

The `+1` represents the passenger themselves.

### Formula

```text
FamilySize = SibSp + Parch + 1
```

## 🚶 Is Alone

A binary feature was created to identify whether a passenger was travelling alone.

```python
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
```

The values represent:

```text
0 → Passenger travelled with family
1 → Passenger travelled alone
```

---

# 🔤 Categorical Encoding

Machine Learning models generally require numerical input.

The categorical columns:

```text
Sex
Embarked
```

were converted into numerical representations using `OneHotEncoder`.

The encoder was configured as:

```python
OneHotEncoder(
    handle_unknown="ignore",
    drop="first"
)
```

### `handle_unknown="ignore"`

This prevents errors if an unseen category appears during prediction.

### `drop="first"`

This removes one category from each encoded feature to avoid unnecessary redundancy.

---

# 📏 Feature Scaling

Numerical features were standardized using `StandardScaler`.

The numerical features used were:

```text
Pclass
Age
SibSp
Parch
Fare
FamilySize
IsAlone
```

Scaling was implemented using:

```python
StandardScaler()
```

Feature scaling ensures that numerical features are placed on a comparable scale, which is useful for Logistic Regression.

---

# ✂️ Train/Test Split

The dataset was divided into training and testing datasets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Configuration

| Parameter | Value |
|---|---|
| Training Data | 80% |
| Testing Data | 20% |
| Random State | 42 |
| Stratification | Enabled |

### Why Stratification?

Stratification helps maintain a similar class distribution between the training and testing datasets.

---

# 🤖 Machine Learning Model

## Logistic Regression

The primary Machine Learning algorithm used in this project is **Logistic Regression**.

Logistic Regression is a supervised Machine Learning algorithm commonly used for binary classification problems.

In this project, the two possible classes are:

```text
0 → Did not survive
1 → Survived
```

The model learns the relationship between passenger features and survival outcomes.

---

# 🔗 Machine Learning Pipeline

A Scikit-learn Pipeline and ColumnTransformer were used to organize the preprocessing and model training process.

The overall process is:

```text
Raw Data
   ↓
Categorical Encoding
   ↓
Numerical Scaling
   ↓
Logistic Regression
   ↓
Prediction
```

Using a pipeline makes the Machine Learning workflow cleaner and more reproducible.

It also ensures that preprocessing operations are applied consistently during both training and prediction.

---

# 🧪 Model Training

The complete preprocessing and model training workflow was combined into a single pipeline.

The model was trained using:

```python
pipeline.fit(X_train, y_train)
```

The model learns patterns from the training dataset.

---

# 🔮 Model Prediction

After training, predictions were generated using:

```python
y_pred = pipeline.predict(X_test)
```

The model's predicted survival probabilities were also calculated:

```python
y_probability = pipeline.predict_proba(X_test)[:, 1]
```

The probability represents the model's estimated probability that a passenger belongs to the survival class.

---

# 📈 Model Evaluation

The trained model was evaluated using multiple classification metrics.

## Performance Results

| Metric | Score |
|---|---:|
| Accuracy | 80.45% |
| Precision | 78.33% |
| Recall | 68.12% |
| F1 Score | 72.87% |
| ROC-AUC | 85.09% |

---

# 📌 Evaluation Metrics Explained

## Accuracy

Accuracy represents the percentage of total predictions that were correct.

```text
Accuracy =
Correct Predictions / Total Predictions
```

The model achieved **80.45% Accuracy**.

## Precision

Precision measures how many passengers predicted as survivors actually survived.

```text
Precision =
True Positives /
(True Positives + False Positives)
```

Model Precision: **78.33%**

## Recall

Recall measures how many of the actual survivors were correctly identified by the model.

```text
Recall =
True Positives /
(True Positives + False Negatives)
```

Model Recall: **68.12%**

## F1 Score

F1 Score is the harmonic mean of Precision and Recall.

```text
F1 Score =
2 × (Precision × Recall) /
(Precision + Recall)
```

Model F1 Score: **72.87%**

## ROC-AUC

ROC-AUC measures how effectively the model can distinguish between the two classes across different classification thresholds.

The model achieved **85.09% ROC-AUC**.

A higher ROC-AUC generally indicates better class separation.

---

# 🧩 Confusion Matrix

A confusion matrix was used to analyze the classification results.

It contains four important values:

```text
True Positive
True Negative
False Positive
False Negative
```

The confusion matrix helps identify which types of predictions the model gets correct and where it makes mistakes.

---

# 📉 ROC Curve

A ROC Curve was generated using the predicted probabilities.

The ROC Curve compares:

```text
True Positive Rate
        vs
False Positive Rate
```

The area under the ROC curve was **85.09%**.

This indicates that the model has a reasonably strong ability to distinguish between survivors and non-survivors.

---

# 🔁 5-Fold Cross Validation

To evaluate model consistency, 5-Fold Cross Validation was performed.

The dataset was divided into five folds.

The model was trained and validated five times, with each fold being used as the validation set once.

## Cross Validation Scores

```text
Fold 1 → 78.21%
Fold 2 → 76.97%
Fold 3 → 79.78%
Fold 4 → 78.09%
Fold 5 → 83.15%
```

### Mean Cross Validation Accuracy

**79.24%**

### Cross Validation Standard Deviation

**2.15%**

The relatively small variation between folds indicates reasonably consistent model performance across the different validation splits.

---

# ⚠️ Overfitting Analysis

Training and testing accuracy were compared to identify potential overfitting.

```text
Training Accuracy → 80.20%
Testing Accuracy  → 80.45%
```

The training and testing accuracy values are very close.

This indicates that the model does not show a significant overfitting issue on the selected train/test split.

---

# 📊 Visualizations

The project contains visualizations covering both dataset analysis and model evaluation.

## Exploratory Data Analysis

```text
✓ Survival Distribution
✓ Gender vs Survival
✓ Passenger Class vs Survival
✓ Age Distribution
✓ Fare Distribution
✓ Fare vs Survival Box Plot
```

## Model Evaluation

```text
✓ Confusion Matrix
✓ ROC Curve
```

These visualizations provide a better understanding of the dataset and the model's performance.

---

# 📁 Project Structure

```text
titanic-survival-prediction/
│
├── Titanic_Survival_Prediction.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

> `train.csv` is kept locally and excluded from the GitHub repository using `.gitignore`.

---

# 📄 File Description

| File | Description |
|---|---|
| `Titanic_Survival_Prediction.ipynb` | Complete Machine Learning notebook |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from Git |
| `LICENSE` | MIT License |

---

# ▶️ How to Run the Project

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Rohithvasanthan/titanic-survival-prediction.git
```

## Step 2 — Navigate to the Project Directory

```bash
cd titanic-survival-prediction
```

## Step 3 — Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scriptsctivate
```

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 5 — Add the Dataset

Download the Titanic `train.csv` dataset from Kaggle and place it inside the project directory.

The structure should look like:

```text
titanic-survival-prediction/
│
├── train.csv
├── Titanic_Survival_Prediction.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Step 6 — Open the Notebook

Open:

```text
Titanic_Survival_Prediction.ipynb
```

using Jupyter Notebook, JupyterLab, VS Code, or Google Colab.

## Step 7 — Run the Notebook

Run the notebook cells sequentially from top to bottom.

---

# 📦 Requirements

The project uses the following Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# 🔐 Data Privacy

The original Titanic dataset is not included in the GitHub repository.

The local dataset file:

```text
train.csv
```

is excluded through `.gitignore`.

This keeps the repository focused on the project code, notebook, documentation, and configuration files.

---

# 🧠 Key Learnings

Through this project, I practiced the complete Machine Learning workflow.

### Data Handling

- Loading datasets using Pandas
- Inspecting dataset structure
- Understanding data types
- Handling missing values
- Detecting duplicate records

### Exploratory Data Analysis

- Statistical analysis
- Distribution analysis
- Relationship analysis
- Data visualization

### Data Preprocessing

- Missing value imputation
- Removing unnecessary features
- Categorical encoding
- Feature scaling

### Feature Engineering

- Creating Family Size
- Creating Is Alone feature

### Machine Learning

- Supervised Learning
- Binary Classification
- Logistic Regression
- Train/Test Split
- Machine Learning Pipelines

### Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Cross Validation
- Overfitting Analysis

---

# 💡 Key Takeaway

This project helped me understand how a Machine Learning model moves from raw data to a complete prediction workflow.

The core workflow learned from this project is:

```text
Understand the Data
        ↓
Clean the Data
        ↓
Explore the Data
        ↓
Engineer Useful Features
        ↓
Preprocess the Features
        ↓
Split the Dataset
        ↓
Train the Model
        ↓
Evaluate the Model
        ↓
Validate the Model
        ↓
Analyze the Results
```

This workflow provides a strong foundation for building more advanced Machine Learning applications.

---

# 🚀 Future Improvements

The current project uses Logistic Regression as the primary model.

Future improvements could include experimenting with additional algorithms.

### Alternative Models

- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine
- K-Nearest Neighbors

### Model Optimization

- Hyperparameter Tuning
- Feature Selection
- Cross Validation
- Ensemble Learning

### Explainability

Future versions could include model explainability techniques to understand which features have the strongest influence on predictions.

### Deployment

The trained model could also be converted into a real-world application using:

```text
Streamlit
FastAPI
Docker
Cloud Deployment
```

---

# 📚 Concepts Demonstrated

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Data Loading
Data Cleaning
Missing Value Handling
Exploratory Data Analysis
Data Visualization
Feature Engineering
Categorical Encoding
OneHotEncoder
Feature Scaling
StandardScaler
Train/Test Split
Stratification
Logistic Regression
Machine Learning Pipeline
ColumnTransformer
Classification
Prediction
Confusion Matrix
Accuracy
Precision
Recall
F1 Score
ROC-AUC
ROC Curve
5-Fold Cross Validation
Overfitting Analysis
```

---

# 📌 Project Highlights

```text
✓ End-to-End Machine Learning Project
✓ Real-World Dataset
✓ Exploratory Data Analysis
✓ Data Cleaning
✓ Feature Engineering
✓ Categorical Encoding
✓ Feature Scaling
✓ Logistic Regression
✓ Scikit-learn Pipeline
✓ Model Evaluation
✓ Confusion Matrix
✓ ROC Curve
✓ ROC-AUC
✓ 5-Fold Cross Validation
✓ Overfitting Analysis
✓ GitHub Documentation
```

---

# ✅ Project Status

```text
✅ Dataset Loaded
✅ Exploratory Data Analysis
✅ Data Cleaning
✅ Missing Value Handling
✅ Feature Engineering
✅ Categorical Encoding
✅ Feature Scaling
✅ Train/Test Split
✅ Logistic Regression
✅ Machine Learning Pipeline
✅ Model Training
✅ Model Prediction
✅ Model Evaluation
✅ Confusion Matrix
✅ ROC Curve
✅ 5-Fold Cross Validation
✅ Overfitting Analysis
✅ README Documentation
✅ GitHub Repository
```

---

# 🏆 Project Outcome

The project successfully demonstrates an end-to-end binary classification workflow using the Titanic dataset.

The Logistic Regression model achieved:

```text
Accuracy  → 80.45%
Precision → 78.33%
Recall    → 68.12%
F1 Score  → 72.87%
ROC-AUC   → 85.09%
```

The project also achieved:

```text
5-Fold CV Accuracy → 79.24%
```

This project strengthened practical understanding of data preprocessing, feature engineering, classification, model evaluation, and validation.

---

# 👨‍💻 Author

## Rohith Vasanthan

BE Computer Science and Engineering Student

Aspiring AI Engineer

### GitHub

https://github.com/Rohithvasanthan

---

# 📜 License

This project is licensed under the MIT License.

See the `LICENSE` file for more information.

---

# ⭐ If You Found This Project Useful

Feel free to explore the repository, review the notebook, and use the project as a reference for learning Machine Learning workflows.

⭐ Star the repository if you find it useful!
