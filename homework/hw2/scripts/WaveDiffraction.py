# Setup Gaussian Quadrature params
from gaussxw import gaussxw


def gQuad(N, a, b, f):
    xg, wg = gaussxw(N)
    xp = 0.5*(b-a)*xg + 0.5*(b+a)
    wp = 0.5*(b-a)*wg
    gq = (wp * f(xp)).sum()
    return gq

## define I/I_0 (C_u, S_u) in terms of the integral functions (treated here as variables
def calcI_I0(Cu,Su):
    p1 = 2. * Cu + 1.
    p2 = 2. * Su + 1.
    ret = (1./8.) * ( (p1 ** 2.) + (p2 ** 2.) )
    return ret

## define C(u), S(u) function integrands. define u(x,z,\lambda)

def C(t):
    return np.cos(m.pi* t ** 2. / 2.)

def S(t):
    return np.sin(m.pi* t ** 2. / 2.)

def calcu(x,z,lam):
    return (x*m.sqrt(2./(lam * z)))


N = 50
a = 0.
# wavelength \lambda set to 1 m
lam = 1.
# z set to 3 meters
z = 3.
# set stepsize
dx = 0.05



x_arr = np.arange(-5.,5.,dx)
I_I0_arr = np.zeros_like(x_arr)

x_list = x_arr.tolist()

for i, x in enumerate(x_list):
    # assign u coordinate using x,z and wavelength
    u = calcu(x,z,lam)
    # set upper limit of integration to u
    b = u
    # evaluate integrals
    C_u = gQuad(N,a,b,C)
    S_u = gQuad(N,a,b,S)
    I_I0_arr[i] = calcI_I0(C_u,S_u)
    
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6,6))

plt.plot(x_arr, I_I0_arr, linewidth=2.,color='navy')
ax.set_xlabel(r'$x$ [m]')
ax.set_ylabel(r'$\frac{I}{I_0}$', fontsize=16)
fig.suptitle(r"Relative Intensity $I/I_0$", fontsize=24)
ax.grid()
# reset step size
dx = 0.1

x_arr = np.arange(-5.,5.,dx)
z_arr = np.arange(0.,5.,dx)

mx, mz = np.meshgrid(x_arr,z_arr)

I_I0_arr = np.zeros_like((mx,mz))

x_list = x_arr.tolist()
z_list = z_arr.tolist()
k=0
print(len(x_list)*len(z_list))
print(np.shape(I_I0_arr))
for i, x in enumerate(x_list):
    for j, z in enumerate(z_list):
        if z == 0.:
            continue
        k += 1
        # assign u coordinate using x,z and wavelength
        u = calcu(x,z,lam)
        # set upper limit of integration to u
        b = u
        # evaluate integrals
        C_u = gQuad(N,a,b,C)
        S_u = gQuad(N,a,b,S)
        I_I0_arr[0,j,i] = calcI_I0(C_u,S_u)

fig, ax = plt.subplots(figsize=(6,8))
pos1 = ax.imshow(I_I0_arr[0,:,:],
           extent=[z_arr.min(),z_arr.max(),x_arr.min(),x_arr.max()],
           cmap='viridis',interpolation='catrom',norm='log',
                vmin=1.e-02,vmax=1.e+01)
fig.colorbar(pos1, ax=ax)

ax.set_xlabel(r'$z$ [m]')
ax.set_ylabel(r'$x$ [m]')

fig.suptitle(r'Spatial distribution of wave intensity $I/I_0$')

plt.show()


fig, ax = plt.subplots(figsize=(6,8))
pos1 = ax.imshow(I_I0_arr[0,:,:],
           extent=[z_arr.min(),z_arr.max(),x_arr.min(),x_arr.max()],
           cmap='turbo',interpolation='catrom',
                vmin=0., vmax=10.)
fig.colorbar(pos1, ax=ax)

ax.set_xlabel(r'$z$ [m]')
ax.set_ylabel(r'$x$ [m]')
fig.suptitle(r'Spatial distribution of wave intensity $I/I_0$')

plt.show()
