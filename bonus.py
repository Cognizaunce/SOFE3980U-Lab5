import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from cleanlab.classification import CleanLearning

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

np.random.seed(42)
num_errors = 5
error_indices = np.random.choice(len(y), num_errors, replace=False)
y[error_indices] = np.random.choice([0, 1, 2], num_errors, replace=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Use Cleanlab's CleanLearning
cleaner = CleanLearning(clf, low_memory=True)
cleaner.fit(X_train, y_train)

# Get label issues (higher scores mean more likely mislabeled)
label_issues = cleaner.find_label_issues(X=X_train, labels=y_train)

# Display suspected mislabeled data
mislabeled_indices = np.where(label_issues["is_label_issue"])[0]
print("Suspected label errors at indices:", mislabeled_indices)

# Assuming cleaner, X_train, y_train, and mislabeled_indices are defined earlier
# Your existing code:
predicted_labels = cleaner.predict(X_train)

suspect_dfs = []
for idx in mislabeled_indices:
    df_suspect = pd.DataFrame([X_train[idx]], columns=iris.feature_names)
    df_suspect.insert(0, "Index", idx)
    df_suspect["True Label"] = y_train[idx]
    df_suspect["Previously Assigned Label"] = predicted_labels[idx]
    suspect_dfs.append(df_suspect)

df_all_suspects = pd.concat(suspect_dfs, ignore_index=True)

print("\n                                           Suspected Mislabeled Data Points")
print("-----------------------------------------------------------------------------------------------------------------------")
print(df_all_suspects.to_string(index=False))

# Load iris data
iris = load_iris(as_frame=True)
df = iris.data
df['target'] = iris.target

# Calculate category statistics
means = df.groupby('target').mean()
stds = df.groupby('target').std()

# Corrected analysis for suspect data
print("\nAnalysis of Suspected Mislabeled Points:")
print("----------------------------------------")

for idx, suspect_row in df_all_suspects.iterrows():
    # Get only the feature columns (excluding Index and Label columns)
    feature_cols = iris.feature_names
    suspect_features = suspect_row[feature_cols]
    
    # Calculate differences from each category mean
    differences = means - suspect_features
    
    # Calculate normalized differences (z-scores)
    z_scores = (means - suspect_features) / stds
    
    print(f"\nSuspect at Index {suspect_row['Index']}:")
    print("True Label:", suspect_row["True Label"])
    print("Previously Assigned Label:", suspect_row["Previously Assigned Label"])
    print("\nRaw Differences from Category Means:")
    print(differences)
    print("\nNormalized Differences (z-scores) from Category Means:")
    print(z_scores)
    
    # Find closest category based on sum of absolute differences
    raw_closest = np.abs(differences).sum(axis=1).idxmin()
    norm_closest = np.abs(z_scores).sum(axis=1).idxmin()
    print(f"\nClosest category (raw differences): {raw_closest}")
    print(f"Closest category (normalized): {norm_closest}")

    