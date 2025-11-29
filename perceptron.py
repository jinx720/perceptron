import numpy as np

class Perceptron:
    def __init__(self, iters = 10, alpha = 0.01):
        self.iters = iters
        self.alpha = alpha

    def fit(self,X,y):
        rows, cols = X.shape
        self.w = np.zeros(cols)
        self.b = 0
        for iter in range(self.iters):
            err = 0
            for xi,yi in zip(X,y):
                output = np.dot(xi,self.w) + self.b
                y_pred = 1 if output >= 0 else 0
                update = self.alpha * (yi - y_pred)
                if update != 0:
                    self.w += update * xi
                    self.b += update
                    err += 1
            print("Errors:", err)
    
    def predict(self,X):
        output = np.dot(X, self.w) + self.b
        return np.where(output >= 0, 1, 0)
                
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 0, 0, 1])
p = Perceptron(iters=20, alpha=0.1)
p.fit(X, y)
print(p.predict(np.array([1, 0])))

