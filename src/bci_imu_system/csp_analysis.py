# CSP共空间模式 + SVM分类
# 参考了braindecode的CSP实现，改了简化版
# 脑电信号特征提取常用这个方法，能把两类信号分开
import numpy as np
from sklearn.svm import SVC

class CSPFilter:
    """
    共空间模式滤波
    就是找一组空间滤波器，让两类运动想象的信号方差差最大
    这样后面分类就好分了
    """
    def __init__(self, n_comp=6):
        self.n_comp = n_comp
        self.filters = None
        
    def fit(self, epochs, labels):
        """
        训练CSP滤波器
        epochs形状是(样本数, 通道数, 时间点)
        labels是对应的标签，0和1两类
        """
        class0 = epochs[labels == 0]
        class1 = epochs[labels == 1]
        
        # 算两类的平均协方差矩阵
        cov0 = np.mean([np.cov(e) for e in class0], axis=0)
        cov1 = np.mean([np.cov(e) for e in class1], axis=0)
        
        # 解广义特征值问题
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(cov0 + cov1) @ cov1)
        
        # 挑特征值离0.5最远的那几个，区分度最大
        idx = np.argsort(np.abs(eigvals.real - 0.5))
        self.filters = eigvecs[:, idx[:self.n_comp]].real
        
    def transform(self, epochs):
        """
        把信号投影到CSP空间，提取对数方差特征
        返回形状是(样本数, 特征数)
        """
        projected = np.array([self.filters.T @ e for e in epochs])
        features = np.log(np.var(projected, axis=2))
        return features


class BCIClassifier:
    """
    CSP + SVM 组合分类器
    先CSP提特征，再SVM分类
    """
    def __init__(self, n_comp=6):
        self.csp = CSPFilter(n_comp)
        self.svm = SVC(kernel='rbf', C=1.0)
        
    def fit(self, epochs, labels):
        features = self.csp.fit_transform(epochs, labels)
        self.svm.fit(features, labels)
        
    def predict(self, epochs):
        features = self.csp.transform(epochs)
        return self.svm.predict(features)
