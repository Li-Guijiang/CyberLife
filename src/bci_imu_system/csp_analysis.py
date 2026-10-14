"""
Common Spatial Pattern (CSP) Filtering + SVM Classification
Reference: sktime, braindecode BCI toolbox
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class CSPSpatialFilter:
    """
    Common Spatial Pattern filter for EEG feature enhancement.
    Maximizes variance difference between two motor imagery classes.
    """
    def __init__(self, n_components: int = 6):
        self.n_components = n_components
        self.filters_ = None
        
    def fit(self, epochs: np.ndarray, labels: np.ndarray):
        """
        Compute CSP filters from labeled epochs.
        epochs: (n_samples, n_channels, n_times)
        labels: (n_samples,) binary labels
        """
        class_0 = epochs[labels == 0]
        class_1 = epochs[labels == 1]
        
        # Compute average covariance matrices
        cov_0 = np.mean([np.cov(e) for e in class_0], axis=0)
        cov_1 = np.mean([np.cov(e) for e in class_1], axis=0)
        
        # Solve generalized eigenvalue problem
        eig_vals, eig_vecs = np.linalg.eig(
            np.linalg.inv(cov_0 + cov_1) @ cov_1
        )
        
        # Sort by eigenvalue distance from 0.5
        sorted_idx = np.argsort(np.abs(eig_vals.real - 0.5))
        self.filters_ = eig_vecs[:, sorted_idx[:self.n_components]].real
        
    def transform(self, epochs: np.ndarray) -> np.ndarray:
        """
        Project epochs onto CSP subspace and extract log-variance features.
        Returns: (n_samples, n_components) feature matrix
        """
        projected = np.array([self.filters_.T @ e for e in epochs])
        features = np.log(np.var(projected, axis=2))
        return features


class BCIClassifier:
    """
    Combined CSP + SVM classifier for motor imagery BCI.
    """
    def __init__(self, n_components: int = 6):
        self.csp = CSPSpatialFilter(n_components)
        self.clf = SVC(kernel='rbf', C=1.0)
        
    def fit(self, epochs: np.ndarray, labels: np.ndarray):
        features = self.csp.fit_transform(epochs, labels)
        self.clf.fit(features, labels)
        
    def predict(self, epochs: np.ndarray) -> np.ndarray:
        features = self.csp.transform(epochs)
        return self.clf.predict(features)
