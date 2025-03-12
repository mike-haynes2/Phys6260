# an example of steepest (gradient) descent minimization

import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x0, x1, a, b):
    return (a - x0)**2 + b*(x1 - x0**2)**2

def drosdx(x, a, b):
    x0 = x[0]
    x1 = x[1]
    return np.array([-2.0*(a - x0) - 4.0*b*x0*(x1 - x0**2),
                     2.0*b*(x1 - x0**2)])

def inside(x, bounds):
    return ((x > bounds[:,0]) & (x < bounds[:,1])).all()

def main():

    xmin = -2.0
    xmax = 2.0
    ymin = -1.0
    ymax = 3.0

    a = 1.0
    b = 100.0

    N = 256
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)

    x2d, y2d = np.meshgrid(x, y, indexing="ij")

    plt.imshow(np.log10(np.transpose(rosenbrock(x2d, y2d, a, b))), 
               origin="lower",
               extent=[xmin, xmax, ymin, ymax])

    plt.colorbar()
    plt.tight_layout()

    plt.savefig("min_2d_start.png", dpi=150)


    # do descent
    xp = np.array([-1.0, 1.5])
    xp_old = 1000*xp

    eps = 1.e-5

    eta = 0.002
    iter = 0
    bounds = np.array([[xmin, xmax], [ymin, ymax]])
    while np.linalg.norm(xp - xp_old) > eps and inside(xp, bounds):
        xp_old[:] = xp[:]

        ### In-class problem: Complete your implementation of steepest descent here

        iter += 1
        if iter % 20 == 0:
            print(f"Iteration {iter}: Completed")
    
        plt.plot([xp_old[0], xp[0]], [xp_old[1], xp[1]], color="C1")

    plt.scatter([xp[0]], [xp[1]], marker="o", color="C1")    
    if not inside(xp, bounds):
        print(f"Exited because point {xp} exited the allocated bounds {bounds}")
        plt.title("Failed")
    else:
        print(f"Found minimum {xp} in {iter} iterations")
        plt.title("Success")

    plt.savefig("min_2d_descent.png", dpi=150)

if __name__ == "__main__":
    main()
