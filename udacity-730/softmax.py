"""Softmax."""

import numpy as np

# scores = [3.0, 1.0, 0.2]
scores = [1.0, 2.0, 3.0]
scores2 = np.array([[1, 2, 3, 6],
                    [2, 4, 5, 6],
                    [3, 8, 7, 6]])
scores3 = np.array([[1/10, 2/10, 3, 6],
                    [2/10, 4/10, 5, 6],
                    [3/10, 8/10, 7, 6]])

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print "softmax(scores2)"
print softmax(scores2)

print "\nsoftmax(scores3)"
print softmax(scores3)

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.ylabel('softmax')
plt.xlabel('x')
plt.show()
