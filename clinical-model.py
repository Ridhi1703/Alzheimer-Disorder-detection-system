# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for clean output

# Step 1: Load the Dataset
file_path = 'alzheimers_disease_data.csv'  # Path to the file
df = pd.read_csv(file_path)

# Display the first few rows
print("Dataset Head:")
print(df.head())

# Step 2: Data Preprocessing - Rounding Numerical Columns
columns_to_round = ['EducationLevel', 'AlcoholConsumption', 'PhysicalActivity',
                    'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
                    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
                    'CholesterolTriglycerides', 'MMSE']

for col in columns_to_round:
    df[col] = df[col].round().astype(int)

# Step 3: Drop Unnecessary Columns
columns_to_drop = ['PatientID', 'Ethnicity', 'ADL', 'DoctorInCharge', 'FunctionalAssessment']
data = df.drop(columns=columns_to_drop)

# Display the dataset after dropping columns
print("\nDataset after Dropping Unnecessary Columns:")
print(data.head())

# Step 4: Define Features and Target Variable
X = data.drop(columns=['Diagnosis'])  # Features
y = data['Diagnosis']  # Target variable

# Step 5: Scale Numerical Features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Step 6: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Print dataset shapes
print("\nTraining and Testing Set Shapes:")
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Step 7: Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Print balanced dataset shapes and class distribution
print("\nBalanced Training Set Shapes:")
print("Resampled training set shape:", X_resampled.shape, y_resampled.shape)
print("Class distribution in Resampled Training Set:\n", y_resampled.value_counts())

# Step 8: Feature Importance with Random Forest
print("\nTraining RandomForest for Feature Selection...")
rfc = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
rfc.fit(X_resampled, y_resampled)

# Identify top features based on importance
importances = rfc.feature_importances_
important_features = pd.Series(importances, index=X.columns).sort_values(ascending=False)
top_features = important_features[important_features > 0.01].index  # Threshold for feature selection

print("\nTop Features Selected:")
print(top_features)

# Step 9: Subset the Data with Top Features
X_train_top = X_resampled[top_features]
X_test_top = X_test[top_features]

# Step 10: Ensemble Voting Classifier with Optimized Models
print("\nTraining Ensemble Voting Classifier...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, max_depth=5, random_state=42)
gbc = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
rfc_opt = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('RandomForest', rfc_opt),
    ('XGBoost', xgb),
    ('GradientBoosting', gbc)
], voting='soft')  # Soft voting for probability averaging

# Train the Voting Classifier
voting_clf.fit(X_train_top, y_resampled)

# Step 11: Model Evaluation
y_pred = voting_clf.predict(X_test_top)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 12: Save the Model
joblib.dump(voting_clf, 'ensemble_model.pkl')
print("\nEnsemble model saved as 'ensemble_model.pkl'.")