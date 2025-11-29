# Perceptron From Scratch

A minimal, educational implementation of a single-layer perceptron written in Python using NumPy.  
This project is intentionally basic and focuses on helping beginners understand how linear classifiers learn through weight updates.

## Overview

- Pure NumPy implementation  
- Perceptron learning rule (`fit`) and binary prediction (`predict`)  
- Optional verbose mode for inspecting training steps  
- Works on linearly separable problems (AND, OR, NAND)  
- Demonstrates failure on non-linearly separable tasks (XOR)  

This is not a production-grade ML model — it is a foundational learning project that will be expanded over time.

## Example Usage

```python
import numpy as np
from perceptron import Perceptron

# AND gate dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,0,0,1])

p = Perceptron(iters=20, alpha=0.1, verbose=False)
p.fit(X, y)

print("Predictions:", p.predict(X))
