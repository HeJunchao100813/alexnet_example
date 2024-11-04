from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Dummy dataset for example
X = np.array([[i] for i in range(8)])  # Features
y = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # Labels
sensitive_features = np.array(['male', 'female', 'female', 'male', 'male', 'female', 'female', 'male'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train a simple logistic regression model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Apply threshold optimizer for post-processing
threshold_optimizer = ThresholdOptimizer(
    estimator=model,
    constraints="demographic_parity",  # Can use "equalized_odds" as well
    prefit=True
)

# Fit and predict
threshold_optimizer.fit(X_train, y_train, sensitive_features=sensitive_features[:4])
y_pred_postprocessed = threshold_optimizer.predict(X_test, sensitive_features=sensitive_features[4:])

print("Predictions after bias mitigation:", y_pred_postprocessed)