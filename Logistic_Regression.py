# ============================================================
# TITANIC SURVIVAL PREDICTION - Logistic Regression Assignment
# ============================================================
# Run: python titanic_logistic_regression.py
# Requirements: pip install pandas scikit-learn matplotlib seaborn joblib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

train_df = pd.read_csv("Titanic_train.csv")
test_df  = pd.read_csv("Titanic_test.csv")

print("=" * 50)
print("STEP 1: DATA EXPLORATION")
print("=" * 50)

print(f"\nTraining set shape : {train_df.shape}")
print(f"Test set shape     : {test_df.shape}")

print("\nFirst 5 rows:")
print(train_df.head())

print("\nColumn data types:")
print(train_df.dtypes)

print("\nSummary statistics:")
print(train_df.describe())

print("\nMissing values in training set:")
print(train_df.isnull().sum())


# ============================================================
# STEP 1c: VISUALIZATIONS (EDA)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic EDA", fontsize=16)

# 1. Survival count
sns.countplot(x='Survived', data=train_df, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title("Survival Count (0=No, 1=Yes)")

# 2. Age distribution
train_df['Age'].dropna().hist(ax=axes[0, 1], bins=30, color='steelblue', edgecolor='black')
axes[0, 1].set_title("Age Distribution")
axes[0, 1].set_xlabel("Age")

# 3. Survival by gender
sns.countplot(x='Sex', hue='Survived', data=train_df, ax=axes[0, 2], palette='Set1')
axes[0, 2].set_title("Survival by Gender")

# 4. Survival by passenger class
sns.countplot(x='Pclass', hue='Survived', data=train_df, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title("Survival by Passenger Class")

# 5. Fare box plot by survival
sns.boxplot(x='Survived', y='Fare', data=train_df, ax=axes[1, 1], palette='coolwarm')
axes[1, 1].set_title("Fare vs Survival")

# 6. Correlation heatmap (numeric columns only)
corr = train_df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[1, 2])
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=100)
plt.show()
print("\n[Saved] eda_plots.png")


# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================

print("\n" + "=" * 50)
print("STEP 2: DATA PREPROCESSING")
print("=" * 50)

def preprocess(df):
    df = df.copy()

    # Fill missing Age with median age
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fill missing Embarked with most frequent value (mode)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Fill missing Fare (test set has 1 missing) with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Encode Sex: male=0, female=1
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

    # Encode Embarked: C=0, Q=1, S=2 (alphabetical order)
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'].astype(str))

    # Drop columns that are not useful for prediction
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    return df

train_clean = preprocess(train_df)
test_clean  = preprocess(test_df)

print("\nCleaned training data (first 3 rows):")
print(train_clean.head(3))

print("\nMissing values after preprocessing:")
print(train_clean.isnull().sum())


# ============================================================
# STEP 3: MODEL BUILDING
# ============================================================

print("\n" + "=" * 50)
print("STEP 3: MODEL BUILDING")
print("=" * 50)

# Features and target
X = train_clean.drop('Survived', axis=1)
y = train_clean['Survived']

# Split: 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples  : {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

# Build and train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Save the model for Streamlit app
joblib.dump(model, "logistic_model.pkl")
print("[Saved] logistic_model.pkl")


# ============================================================
# STEP 4: MODEL EVALUATION
# ============================================================

print("\n" + "=" * 50)
print("STEP 4: MODEL EVALUATION")
print("=" * 50)

y_pred      = model.predict(X_val)
y_pred_prob = model.predict_proba(X_val)[:, 1]  # probability of class 1

# Core metrics
acc       = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall    = recall_score(y_val, y_pred)
f1        = f1_score(y_val, y_pred)
roc_auc   = roc_auc_score(y_val, y_pred_prob)

print(f"\nAccuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nFull Classification Report:")
print(classification_report(y_val, y_pred, target_names=['Not Survived', 'Survived']))

# Cross-validation (5-fold) for robustness
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-Val Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- Plot: ROC Curve ---
fpr, tpr, _ = roc_curve(y_val, y_pred_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=100)
plt.show()
print("[Saved] roc_curve.png")

# --- Plot: Confusion Matrix ---
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.show()
print("[Saved] confusion_matrix.png")


# ============================================================
# STEP 5: INTERPRETATION — Logistic Regression Coefficients
# ============================================================

print("\n" + "=" * 50)
print("STEP 5: INTERPRETATION")
print("=" * 50)

feature_names = X.columns.tolist()
coefficients  = model.coef_[0]

coef_df = pd.DataFrame({
    'Feature'    : feature_names,
    'Coefficient': coefficients
}).sort_values('Coefficient', ascending=False)

print("\nLogistic Regression Coefficients (higher = more impact on survival):")
print(coef_df.to_string(index=False))

# Plot coefficients
plt.figure(figsize=(8, 5))
colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel("Coefficient Value")
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.tight_layout()
plt.savefig("coefficients.png", dpi=100)
plt.show()
print("[Saved] coefficients.png")

print("""
INTERPRETATION NOTES:
- Positive coefficient → feature increases survival probability
- Negative coefficient → feature decreases survival probability
- Sex (female=1): likely strong positive → females had higher survival
- Pclass: likely negative → higher class number (3rd class) = lower survival
- Age: slight negative → older passengers slightly less likely to survive
- Fare: positive → higher fare = better class = higher survival
""")

print("\n[DONE] All outputs saved. Run streamlit_app.py for deployment.")