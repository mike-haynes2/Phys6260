import numpy as np
import matplotlib.pyplot as plt
import time
import copy

class ModelData(object):
    '''
    Input/output data for the neural network because the example NN code shown in class had the same structure
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

class BurgerSolver(object):
    def __init__(self, n=1000, nu=0.1, tend=1, cfl=0.01):
        """
        Initialization of all parameters, solution arrays, and set the initial conditions
        Inputs: n :: Number of x points
                nu :: diffusion coefficient
                tend :: stop time
                cfl :: Courant factor controlling the timestep
        """
        self.L = 2.0
        self.n = n
        self.dx = self.L / self.n
        self.x = np.linspace(0, self.L, self.n) - self.L / 2  # Centered on zero
        self.nu = nu
        self.t = 0.0
        self.tend = tend
        self.dt = cfl * self.dx
        print(self.dt)
        self.Nt = int(self.tend / self.dt)
        self.u = np.empty((self.n, self.Nt + 1))
        self.dudx = np.zeros((self.n, self.Nt + 1))
        self.iter = 0
        self.seed = None

        # Initial conditions
        self.u[:, 0] = -np.sin(np.pi * self.x)

    def plot_one(self, iter=None, show=True, save=False):
        """
        Plot one solution at Iteration `iter`.  You can either `show` it or `save` it.
        Default is to plot the final solution.
        """
        if iter == None:
            iter = self.iter  # last iteration
        fig, ax = plt.subplots()
        ax.plot(self.x, self.u[:, iter])
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"Time = {self.t:.4g} :: Iteration {iter}")
        if save:
            plt.savefig(f"burger-iter{iter:05d}.png")
        if show:
            plt.show()

    def plot_evo(self, show=True, save=False):
        """
        Plot all solutions as a space-time diagram.  You can either `show` it or `save` it.
        """
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(
            self.u,
            origin="lower",
            cmap="RdBu",
            extent=[0, self.tend, -self.L / 2, self.L / 2],
            aspect="auto",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("x")
        fig.colorbar(im, label="u")
        ax.set_title(f"Burger's Equation evolution")
        if save:
            plt.savefig(f"burger-iter{self.iter:05d}.png")
        if show:
            plt.show()

    def solve(self, pbar=True, verbose=False):
        """
        Numeric solver for the Burger equation.
        """
        # Pre-compute some quantities
        dx_inv = 1 / self.dx
        dtdx = self.dt * dx_inv
        dtdx2 = self.dt * dx_inv * dx_inv
        du = np.empty(self.n)
        if pbar and not verbose:
            print(
                "0|"
                + "-" * 25
                + "|"
                + "-" * 25
                + "|"
                + "-" * 25
                + "|"
                + "-" * 25
                + "|100"
            )
        while self.t < self.tend:
            u = self.u[:, self.iter]  # Just for conciseness
            unew = u.copy()

            ### COMPLETE CODE HERE
            # FTUS solver.
            # self.u holds the solutions for all times.
            # u is the current solution.
            # unew is the future solution, which you need to compute with finite differencing.

            # Determine filters for forward (u > 0) and backward (u < 0) propagation 
            # for the FTUS method.
            direction = np.sign(u)
            direction[0] = 0
            direction[-1] = 0
            # Forward and backward indices
            fidx = np.where(direction > 0)[0]
            bidx = np.where(direction < 0)[0]
            du[:] = 0.0
            du[fidx] = (u[fidx] - u[fidx-1])
            du[bidx] = (u[bidx+1] - u[bidx])
            #du[1:-1] = 0.25*du[:-2] + 0.5*du[1:-1] + 0.25*du[2:]
            unew[1:-1] = u[1:-1] - dtdx * u[1:-1] * du[1:-1] + \
                self.nu * dtdx2 * (u[2:] - 2*u[1:-1] + u[:-2])
            self.u[:,self.iter+1] = unew

            # Use center differencing to compute du/dx from new solution.
            # du/dx boundaries are already zero, so only modify [1:-1]
            self.dudx[1:-1, self.iter + 1] = (unew[2:] - unew[:-2]) * dx_inv
            self.iter += 1
            self.t += self.dt
            if verbose:
                print(
                    f"Time = {self.t} seconds, mean/min/max = {unew.mean()} / {unew.min()} / {unew.max()}"
                )
            elif pbar:
                print("  " + "=" * int(self.iter / self.Nt * 100) + ">", end="\r")
            if self.iter == self.Nt:
                break

    def random_sample(self, N, seed=None):
        """
        Returns a ModelData object with random samples of u(x,t), x, t from our solution.
        Used in this notebook to train a neural network below.
        """
        if self.iter == 0:
            print("Calculation not performed yet. Returning nothing.")
            return None
        # Initialize RNG if not done already
        if self.seed == None:
            if seed != None:
                self.seed = seed
            else:
                self.seed = int(time.time())
            np.random.seed(self.seed)
        randx = np.random.randint(1, self.n - 1, size=N)
        randt = np.random.randint(0, self.Nt, size=N)
        input = np.array([self.x[randx], self.dt * randt])
        output = np.atleast_2d(self.u[randx, randt])
        obj = ModelData(input, output)
        return obj
    
class NeuralNetwork(object):
    '''
    A neural network class with a single hidden layer.

    '''

    def __init__(
        self,
        solver,
        num_training_unique=1000,
        n_epochs=10,
        learning_rate=0.1,
        regularization_rate=0.05,
        hidden_layer_size=2,
    ):
        """
        Initialization routine for parameters, training data, and NN weights (matrices)
        Inputs:
        * solver :: PDE solver with a `random_sample` routine that returns a ModelData object.
                    In HW7, it'd be a BurgerSolver object after you solve the system.
        * num_training_unique :: Training dataset size
        * n_epochs :: Number of training epochs, each randomly selecting `num_training_unique` inputs
        * learning_rate :: Factor that softens the gradient descent during minimization
        * regularization_rate :: Factor that controls how much the PDE residual is weighted (if used) in the loss function
        * hidden_layer_size :: Number of nodes in the single hidden layer
        """

        self.solver = solver
        self.num_training_unique = num_training_unique
        self.n_epochs = n_epochs

        self.train_set = self.solver.random_sample(self.num_training_unique)
        self.normalize(set_limits=True)
        self.set_boundary_points()
        # learning rate
        self.eta = learning_rate
        self.lam = regularization_rate

        # we get the size of the layers from the length of the input
        # and output
        model = self.train_set

        # the number of nodes/neurons on the output layer
        self.m = model.y.shape[0]

        # the number of nodes/neurons on the input layer
        self.n = model.x.shape[0]

        # the number of nodes/neurons on the hidden layer
        self.k = hidden_layer_size

        # we will initialize the weights with Gaussian normal random
        # numbers centered on 0 with a width of 1/sqrt(n), where n is
        # the length of the input state

        # A is the set of weights between the hidden layer and output layer
        self.A = np.random.normal(0.0, 1.0 / np.sqrt(self.k), (self.m, self.k))

        # B is the set of weights between the input layer and hidden layer
        self.B = np.random.normal(0.0, 1.0 / np.sqrt(self.n), (self.k, self.n))

    def normalize(self, model=None, set_limits=False):
        """
        Training NNs typically work best when the inputs and outputs are normalized.
        """
        if set_limits:
            self.xmin = self.train_set.x.min(1)
            self.xmax = self.train_set.x.max(1)
            self.ymin = self.train_set.y.min()
            self.ymax = self.train_set.y.max()
        if model == None:
            self.train_set.x = (self.train_set.x - self.xmin[:, None]) / (
                self.xmax[:, None] - self.xmin[:, None]
            )
            self.train_set.y = (self.train_set.y - self.ymin) / (self.ymax - self.ymin)
        else:
            _model = copy.deepcopy(model)
            _model.x = (_model.x - self.xmin[:, None]) / (
                self.xmax[:, None] - self.xmin[:, None]
            )
            _model.y = (_model.y - self.ymin) / (self.ymax - self.ymin)
            return _model

    def denormalize(self, data):
        """
        This function is used when we need to remove the normalization factors when providing predictions.
        """
        data = data * (self.ymax - self.ymin) + self.ymin
        return data

    def set_activation(self, gtype="sigmoid"):
        if gtype not in ["sigmoid", "relu", "tanh", "leaky_relu"]:
            raise RuntimeError(f"Activation function {gtype} unknown.")
        self.gtype = gtype

    def g(self, p, type="sigmoid"):
        """
        our activation function that operates on the hidden layer.
        NOTE: AFAIK in this code, sigmoid is the only function that works in Burger's Equation for 1 hidden layer.
        """
        if self.gtype == "sigmoid":
            return 1.0 / (1.0 + np.exp(-p))
        elif self.gtype == "relu":
            return np.maximum(0, p)
        elif self.gtype == "leaky_relu":
            return np.maximum(0.1 * p, p)
        elif self.gtype == "tanh":
            return np.tanh(p)
        else:
            raise RuntimeError(f"Activation function {self.gtype} unknown.")

    def set_boundary_points(self, N=50):
        """
        Sets points for the initial and boundary conditions so we can use them in the loss function.
        """
        self.bc_x = np.linspace(self.xmin[0], self.xmax[0], N)
        self.bc_t = np.linspace(self.xmin[1], self.xmax[1], N)

        ###
        ### COMPLETE HERE.  Set the analytic functions that describe the initial and boundary conditions.
        ###

        self.bc_x = np.linspace(self.xmin[0],self.xmax[0],N)
        self.bc_t = np.linspace(self.xmin[1],self.xmax[1],N)
        self.initial_fn = lambda x: -np.sin(np.pi * x)
        self.boundary_fn = lambda x: 0.0

    def return_ic_loss(self):
        """
        Returns the mean squared error between the predictions and analytic initial condition.
        """
        ###
        ### COMPLETE HERE.
        ###

        nn = self.bc_x.size
        input_data = np.array([self.bc_x, np.zeros(nn)])
        initial_model = ModelData(input_data, np.zeros(nn))  # 2nd argument unused
        pred_t0 = self.predict(initial_model)
        return np.linalg.norm(pred_t0[0] - self.initial_fn(self.bc_x), 2) / nn

    def return_bc_loss(self):
        """
        Returns the mean squared error between the predictions and analytic boundary condition.
        """
        ###
        ### COMPLETE HERE.
        ###

        nn = self.bc_t.size
        input_data = np.array([np.ones(nn), self.bc_t])
        input_data[0,::2] *= -1  # Alternate between +/- 1 for x-boundary
        bc_model = ModelData(input_data, np.zeros(nn))  # 2nd argument unused
        pred_x0 = self.predict(bc_model)
        return np.linalg.norm(pred_x0[0] - self.boundary_fn(self.bc_t), 2) / nn / nn

    def return_deriv(self, x0, t0, dx=1e-3, dt=1e-3):
        """
        Given a (x,t) value, this function retuns the following partial derivatives so we can compute the residual from Burger's Equation.
        Inputs:
        * x0 :: position to evaluate derivatives
        * t0 :: time to evaluate derivatives
        * dx :: distance between adjacent points when computing spatial derivatives
        * dt :: time between adjacent points when computing time derivatives

        Returns:
        * u_t :: du/dt
        * u_x :: du/dx
        * u_xx :: d^2 u/dx^2
        """
        ###
        ### COMPLETE HERE
        ###
        # Use the NN to predict the u-values at nearby points and use central differencing to compute all the needed derivatives

        # Use central differencing
        xx = [x0-0.5*dx, x0, x0+0.5*dx] # for du/dx
        xt = [x0, x0, x0]  # for du/dt
        tx = [t0, t0, t0] # for du/dx
        tt = [t0-0.5*dt, t0, t0+0.5*dt]  # for du/dt
        y_empty = np.zeros((1,3))
        x_stencil = ModelData(np.array([xx, tx]).reshape(2,3), y_empty)
        t_stencil = ModelData(np.array([xt, tt]).reshape(2,3), y_empty)
        ux = self.predict(x_stencil)
        ut = self.predict(t_stencil)
        u_t = (ut[0,2] - ut[0,0]) / dt
        u_x = (ux[0,2] - ux[0,0]) / dx
        u_xx = (ux[0,2] - 2*ux[0,1] + ux[0,0]) / dx**2
        return u_t, u_x, u_xx

    def return_res_loss(self, u, x, t):
        """
        Returns residual from Burger's equation: u_t + u*u_x - nu*u_xx = 0
        """
        u_t, u_x, u_xx = self.return_deriv(x, t)
        res = u_t + u * u_x - self.eta * u_xx
        return np.atleast_2d(res)

    def set_loss_function(self, fn=None, method="exact"):
        if fn is not None:
            self.loss = fn
        else:
            if method == "exact":
                self.loss = lambda x,y,z: z-y
            # Add your methods here or assign a function
            elif method == "exact+initial":
                self.loss = lambda x,y,z: z - y + self.return_ic_loss()
            elif method == "exact+initial+boundary":
                self.loss = lambda x,y,z: z - y + self.return_ic_loss() + self.return_bc_loss()
            elif method == "res+initial+boundary":
                self.loss = lambda x,y,z: z - y + self.lam*self.return_res_loss(y[0],x[0],x[1]) + \
                    self.return_ic_loss() + self.return_bc_loss()
            else:
                raise RuntimeError(f"Method {method} not recognized.")

    def train(self):
        """
        Train the neural network by doing gradient descent with back
        propagation to set the matrix elements in B (the weights
        between the input and hidden layer) and A (the weights between
        the hidden layer and output layer)
        """
        all_loss = []
        print(
            "0|" + "-" * 25 + "|" + "-" * 25 + "|" + "-" * 25 + "|" + "-" * 25 + "|100"
        )
        for iepoch in range(self.n_epochs):
            # print(f"epoch {i+1} of {self.n_epochs}")
            loss = 0.0
            for _ in range(self.num_training_unique):
                ii = np.random.randint(0, self.num_training_unique)

                # Convert into 1D
                x = self.train_set.x[:, ii].reshape(self.n, 1)
                y = self.train_set.y[:, ii].reshape(self.m, 1)

                z_tilde = self.g(self.B @ x)
                z = self.g(self.A @ z_tilde)

                e = self.loss(x, y, z)
                loss += (e**2).sum()
                if np.isinf(loss):
                    raise RuntimeError(
                        f"Infinite loss function. Epoch {iepoch}, e = {e}, x,y = {x},{y}"
                    )
                e_tilde = self.A.T @ e

                dA = -2 * self.eta * e * z * (1 - z) @ z_tilde.T
                dB = -2 * self.eta * e_tilde * z_tilde * (1 - z_tilde) @ x.T

                self.A[:, :] += dA
                self.B[:, :] += dB

            print("  " + "=" * int(iepoch / self.n_epochs * 100) + ">", end="\r")
            all_loss.append(loss)
        return np.array(all_loss) / self.num_training_unique

    def predict(self, model):
        """predict the outcome using our trained matrix A"""
        nmodel = self.normalize(model=model)
        y = self.g(self.A @ (self.g(self.B @ nmodel.x)))
        return self.denormalize(y)