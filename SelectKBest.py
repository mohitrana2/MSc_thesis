import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

# Load the data
df = pd.read_csv('/data4/msc23104470/version1_data-matrix-TTD-w-HUGO-symbols-ICD-11.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

# Separate features and target
X = df.iloc[:, 1:-1]  # Exclude 'rid' and 'ICD-11' columns
y = df.iloc[:, -1]    # The 'ICD-11' column

# Initialize variables to store results
num_features = range(10, X.shape[1], 10)  # Test different numbers of features
mean_scores = []

# Function to evaluate model performance for a given number of features
def evaluate_features(k):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    clf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(clf, X_new, y, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
    mean_score = scores.mean()
    return k, mean_score, selector.scores_

# Evaluate model performance with different numbers of features in parallel
results = Parallel(n_jobs=-1)(delayed(evaluate_features)(k) for k in num_features)

# Extract the results
num_features_selected = [result[0] for result in results]
mean_scores = [result[1] for result in results]
feature_scores_list = [result[2] for result in results]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(num_features_selected, mean_scores, marker='o')
plt.title('Model Performance vs. Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validated Accuracy')
plt.grid(True)
plt.savefig('model_performance_vs_num_features.png')
plt.show()

# Find the optimal number of features
optimal_num_features = num_features_selected[mean_scores.index(max(mean_scores))]
print(f'Optimal number of features: {optimal_num_features}')

# Feature scores for optimal number of features
optimal_feature_scores = feature_scores_list[mean_scores.index(max(mean_scores))]

# Plot feature scores
plt.figure(figsize=(14, 8))
sns.barplot(x=np.arange(len(optimal_feature_scores)), y=optimal_feature_scores)
plt.title('Feature Scores for Optimal Number of Features')
plt.xlabel('Feature Index')
plt.ylabel('Score')
plt.savefig('optimal_feature_scores.png')
plt.show()

# Save detailed results to a CSV file
results_df = pd.DataFrame({
    'num_features': num_features_selected,
    'mean_scores': mean_scores
})
results_df.to_csv('feature_selection_results.csv', index=False)

# Save optimal features scores to a CSV file
optimal_features_df = pd.DataFrame({
    'feature_index': np.arange(len(optimal_feature_scores)),
    'score': optimal_feature_scores
})
optimal_features_df.to_csv('optimal_feature_scores.csv', index=False)

