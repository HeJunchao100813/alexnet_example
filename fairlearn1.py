import numpy as np
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score

def metrics_fun():
    # Dummy data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # True labels
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])  # Predicted labels
    sensitive_features = np.array(['male', 'female', 'female', 'male', 'male', 'female', 'female', 'male'])

    # Create MetricFrame for accuracy, grouped by gender
    metric_frame = MetricFrame(
        metrics=accuracy_score,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

if __name__ == '__main__':
    # Show overall accuracy and per-group accuracy
    print("Overall Accuracy:", metric_frame.overall)
    print("Accuracy by Group:", metric_frame.by_group)