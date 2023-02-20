import matplotlib.pyplot as plt
import numpy as np

#Logistic Equation: x(n+1) = r * x(n) * (1 - x(n-1))
def logistic(x0, r, t_end):
    x = np.zeros(t_end)
    x[0] = x0
    for i in range(1, t_end):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

if __name__=="__main__":
    # Set the parameters for the simulation
    r_vals = [2, 3.9] # growth rate
    x0_vals = [0.5, 0.5 + 1e-6] # initial population value; change between 0.5 and 0.50000001
    t_end = 100 # number of time steps

    # Simulate the logistic map


    for r in r_vals:
        fig, ax = plt.subplots(figsize=(7.16, 3))
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.set_title(f"Logistic Map | r={r}")
        for x0 in x0_vals:
            ax.plot(logistic(x0, r, t_end), label=f"x0={x0}")
        plt.legend()
        plt.savefig(f"logistic_r{r}.svg")
    plt.show()
