import matplotlib.pyplot as plt

def animate_evolution(i, metrics):
    min_fitnesses = metrics
    plt.plot(min_fitnesses)