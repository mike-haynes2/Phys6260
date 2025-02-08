import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import constants
import math as m
import warnings
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('dark_background')
warnings.filterwarnings( "ignore")



############################################################### Problem 19 ###############################################################


def Thomas(d,a,b,y,x):                                                                                  
    for i in range(len(d)):
        a[i] = a[i]/(d[i] - b[i]*a[i-1])    #take advantage of the fact that a[n] = 0 so a[-1] = 0
        y[i] = (y[i] - b[i]*y[i-1])/(d[i] - b[i]*a[i-1]) if i!= 0 else y[i]/d[i]
    for i in range(len(d)-1,-1,-1):
        x[i] = y[i] if i == (len(d)-1) else y[i] - a[i]*x[i+1]
    return x





## part a

## initial condition
def h0(x, L, Hmax):
    length = len(x)
    h_arr = np.empty(length)
    for x_el in range(len(x)):
        if (x[x_el] <= L) and (x[x_el] >= 0):
            h = (Hmax*x[x_el])/L
        elif (x[x_el] > L) and (x[x_el] <= 2*L):
            h = Hmax - (Hmax/L)*(x[x_el] - L)
        else:
            h = 0.
        h_arr[x_el] = h

    return h_arr


x_IC = np.arange(start=0,stop=20.1,step=0.1)

h_IC = h0(x_IC, 10., 20.)


## plot of IC
fig, ax = plt.subplots(figsize=(60,100))
plt.plot(x_IC,h_IC,color='green',label=r'hill profile at $t\,=\,0$')

#plt.ylim(-100,100)
plt.legend()
plt.grid()
plt.xlabel("position $x$")
plt.ylabel("$h(x,0)$")
plt.title("Altitude Profile of Initial Condition")
#plt.show()
plt.close()




## part b, c

dx = 0.1
x_coords = np.arange(start=0,stop=20,step=dx)
dt = 0.033

k1 = 0.1
k2 = 1.0
k3 = 1.5


L = 10.
Hmax = 20.
Hlim = Hmax / 2.



hnew = np.empty(len(x_coords))      # x in Thomas algorithm nomenclature
hold = np.empty(len(x_coords))      # y in Thomas algorithm nomenclature

#kap_arr = [k1,k2,k3]

kap_arr = [0.1, 0.3, 0.66]


k_ind = -1
print("Output diffusion number for each simulation\n")

for k in kap_arr:
    k_ind += 1

    for j in range(200):
        hold[j] = h_IC[j]
        hnew = np.zeros(len(x_coords)) 
    
    

    D = (k*dt)/(dx ** 2)
    print("\nDiffusion number D = "+str(D))
    if (abs(D) >= 1.):
        print("Warning, diffusion number exceeds unity. While numerical instability will not be incurred through solver evolution because this method is implicit,\nit may result in over-driving the diffusive process.")
    
    d = 2*D + 1

    upper = np.full(len(x_coords), -D)   # a in Thomas algorithm nomenclature
    lower = np.full(len(x_coords), -D)   # b in Thomas algorithm nomenclature
    diag =  np.full(len(x_coords), d)   # d in Thomas algorithm nomenclature
    upper[-1] = 0.
    lower[0] = 0.

    half_ind = len(x_coords) // 2
    hcur = h_IC[half_ind]

    H_t = [hold.copy()]    # initialize storage array where H[i][j] is defined and appended to such that i is the timestep index, and j is the spatial index

    t = 0
    counter = 0

    if k_ind == 0:
        while hcur >= Hlim:

            t += dt
            counter += 1

            ## pass copies or else the solver will fail for some reason
            hnew = Thomas(diag.copy(),upper.copy(),lower.copy(), hold.copy(), hnew.copy())
            hold = hnew.copy()
            H_t.append(hold.copy())
            hcur = hold[half_ind]

        total_steps = counter
        # assert(all(np.array(H_t[-1]) == np.array(H_t[10])))

        numsteps = int(t / dt)
        #print(numsteps)
        increment = numsteps // 4


        h_p0 = H_t[0][:]
        h_p1 = H_t[increment][:]
        h_p2 = H_t[2*increment][:]
        h_p3 = H_t[3*increment][:]
        h_p4 = H_t[numsteps][:]




        x_grid = np.arange(start=0,stop=20.1,step=dx)

        ## plot of solution
        fig, ax = plt.subplots(figsize=(60,100))



        plt.plot(x_coords,h_p0,color='green',label=r'hill profile at $t\,=\,0$')
        plt.plot(x_coords,h_p1,color='yellow',label=r'hill profile at $t\,=\,T/4$')
        plt.plot(x_coords,h_p2,color='orange',label=r'hill profile at $t\,=\,T/2$')
        plt.plot(x_coords,h_p3,color='red',label=r'hill profile at $t\,=\,3T/4$')
        plt.plot(x_coords,h_p4,color='magenta',label=r'hill profile at $t\,=\,T$')

        plt.xlim(0,21)
        plt.ylim(-0.5,20.5)
        plt.legend()
        plt.grid()
        plt.xlabel("position $x$")
        plt.ylabel("$h(x,t)$")
        plt.title("Altitude Evolution of Eroded Hill: $\kappa = $"+str(k))
        plt.show()

    elif k_ind > 0:
        counter = 0
        while total_steps >= counter:

            t += dt
            counter += 1

            ## pass copies or else the solver will fail for some reason
            hnew = Thomas(diag.copy(),upper.copy(),lower.copy(), hold.copy(), hnew.copy())
            hold = hnew.copy()
            H_t.append(hold.copy())
            hcur = hold[half_ind]


        # assert(all(np.array(H_t[-1]) == np.array(H_t[10])))

        numsteps = int(t / dt)
        #print(numsteps)
        increment = numsteps // 4


        h_p0 = H_t[0][:]
        h_p1 = H_t[increment][:]
        h_p2 = H_t[2*increment][:]
        h_p3 = H_t[3*increment][:]
        h_p4 = H_t[numsteps][:]




        x_grid = np.arange(start=0,stop=20.1,step=dx)

        ## plot of solution
        fig, ax = plt.subplots(figsize=(60,100))



        plt.plot(x_coords,h_p0,color='green',label=r'hill profile at $t\,=\,0$')
        plt.plot(x_coords,h_p1,color='yellow',label=r'hill profile at $t\,=\,T/4$')
        plt.plot(x_coords,h_p2,color='orange',label=r'hill profile at $t\,=\,T/2$')
        plt.plot(x_coords,h_p3,color='red',label=r'hill profile at $t\,=\,3T/4$')
        plt.plot(x_coords,h_p4,color='magenta',label=r'hill profile at $t\,=\,T$')

        plt.xlim(0,21)
        plt.ylim(-0.5,20.5)
        plt.legend()
        plt.grid()
        plt.xlabel("position $x$")
        plt.ylabel("$h(x,t)$")
        plt.title("Altitude Evolution of Eroded Hill: $\kappa = "+str(k))
        plt.show()

    







    ############################################################### Problem 20 ###############################################################


tau = 365   # days / year
eta = 0.1   # m^2 / day

## Modulating Parameters ##
A = 10.     # deg Celsius
B = 12.     # deg Celsius
BC_inner = 11.  # deg Celsius
dz = 0.1    # meters
#############################



## calculate timestep to fulfill stability condition, namely     (eta * dt / dx^2) < 1/2
# Set this parameter to choose the threshold below stability to modulate timestep,  **ranges (1,infty)**
# effectively the factor by which dt would need to be multiplied to make the stability condition equation an equality
timestep_threshold = 1.01
# calculate timestep
dt = (dz ** 2)/(2.*eta)
dt /= timestep_threshold
## this gives timestep in DAYS
print("\n\nTimestep in days: "+str(dt)+"\n")
# convert to years
dt /= tau

## compute D
D_coef = (eta*dt*tau) / (dz ** 2)



## initialize coordinates
depth_arr = np.arange(start=0,stop=20.,step=dz)
num_spatial_steps = len(depth_arr)

time_arr = np.arange(start=0.,stop=10.,step=dt)
num_timesteps = len(time_arr)



## Time dependent surface boundary condition
def BC_t(t):
    return (A + B*np.sin((2*m.pi)*t))

# plt.plot(time_arr,BC_t(time_arr))
# plt.show()

## initialize IC
depth_IC = np.full(num_spatial_steps,10.)   # set all values to 10 deg Celsius
depth_IC[0] = BC_t(0.)
depth_IC[-1]= 11.

Temp_profile = depth_IC.copy()
newTemp_profile = np.zeros(num_spatial_steps)
timeDependent_Temp_profile = [Temp_profile.copy()]

## START SOLVER


for n in range(num_timesteps):
    for j in range(num_spatial_steps):
        if j == 0:
            newTemp_profile[j] = BC_t(n*dt)
        elif j == num_spatial_steps-1 :
            newTemp_profile[j] = BC_inner
        else:
            newTemp_profile[j] = Temp_profile[j] + D_coef*(Temp_profile[j+1] - 2*Temp_profile[j] + Temp_profile[j-1])
        

    timeDependent_Temp_profile.append(newTemp_profile.copy())
    Temp_profile = newTemp_profile.copy()




## plotting and visualization

dt_per_year = 1 // dt
dt_inc = dt_per_year // 4

#print(dt_per_year)


plot_1 = timeDependent_Temp_profile[num_timesteps-1][:]
plot_2 = timeDependent_Temp_profile[int(num_timesteps-dt_inc)][:]
plot_3 = timeDependent_Temp_profile[int(num_timesteps-(2*dt_inc))][:]
plot_4 = timeDependent_Temp_profile[int(num_timesteps-(3*dt_inc))][:]




x_grid = np.arange(start=0,stop=20.1,step=dz)

## plot of solution
fig, ax = plt.subplots(figsize=(60,100))

plt.plot(depth_arr, plot_4,label='Summer',color='cyan')
plt.plot(depth_arr, plot_3,label='Fall', color='yellow')
plt.plot(depth_arr, plot_2,label='Winter',color='magenta')
plt.plot(depth_arr, plot_1,label='Spring',color='limegreen')


plt.xlim(0,21)
plt.ylim(0,20)

plt.legend()
plt.grid()
plt.xlabel("depth $z$")
plt.ylabel("$T(z,t=t_{fixed})$")
plt.title("Spatial evolution of temperature propogation through Earth's crust (4 seasons during the final simulated year)")
plt.show()



#print(np.shape(timeDependent_Temp_profile))

plt.close()
plt.figure()
ax = plt.gca()

#fig1 = plt.imshow(timeDependent_Temp_profile, extent = [0,360, -90,90], origin = 'lower', cmap = 'viridis',interpolation='catrom', vmin = 0, vmax = maxel+1e2)
fig1 = plt.imshow(np.transpose(timeDependent_Temp_profile), origin = 'lower', cmap = 'viridis', vmin = -3, vmax = 23, aspect = 'auto')


divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

ax.set_ylabel('Depth in dz (z/'+str(dz)+')')
ax.set_xlabel('Timestep (T*'+str(dt_per_year)+')')

ax.set_title('Temperature evolution through Earth\'s outer crust over 10 years')
cbar = plt.colorbar(fig1,cax = cax, extend='both')
cbar.set_label('Temperature',y=0.3,rotation=270)
plt.show()

