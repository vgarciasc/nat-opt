import pdb
import matplotlib.pyplot as plt
import numpy as np

def animate_evolution(i, metrics):
    min_fitnesses = metrics
    plt.plot(min_fitnesses)

def plot_decision_surface(X, y, model):
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1

    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    
    xx, yy = np.meshgrid(x1grid, x2grid)
    
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    
    grid = np.hstack((r1,r2))
    yhat = np.array([model.predict(row) for row in grid])

    zz = yhat.reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap='Paired')
    
    for class_value in range(2):
        row_ix = np.where(y == class_value)
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
    
    plt.show()