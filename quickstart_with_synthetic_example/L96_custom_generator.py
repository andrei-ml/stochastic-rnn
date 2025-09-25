import numpy as np
from sdeint import itoint
import matplotlib.pyplot as plt
import pickle as pkl
import os
Generator=np.random.default_rng(seed=1) ## Generator to use in itoint for reproducability

# These are our constants
F1, F2 =6,3
#F1, F2 =6,6#If we are going to consider stationary case
sigma=0.3
F_ampl=2.
F_period=10.
D = 5  # Number of variables
dt=0.01 #Integration time step
tstep=10 # Sampling step (number of dt)
tobs=200 #Corresponding interval duration in model units
tskip=300 #Initial relaxation interval duration in model units

# We put a linear parameter F (slow trend) and add its periodic disturbance. 
# The period of disturbance is 10 time units, while its phase is dependent periodically on the coordinate on the Lorenz96 circle.
def Fslow(t):
    """Slow Forcing"""
    return F1+(F2-F1)/tobs*t

def Fperiodic(t):
    """Periodic disturbanceForcing"""
    return np.stack([F_ampl*np.sin(2*np.pi*t/F_period), F_ampl*np.cos(2.*np.pi*t/F_period)],axis=-1)

def L96(x, t):
    """Lorenz 96 model with forcing F(t)"""
    d = np.zeros(D)
    periodic=Fperiodic(t)
    for i in range(D):
        d[i] = (x[(i + 1) % D] - x[i - 2]) * x[i - 1] - x[i] + Fslow(t)+periodic[...,0]*np.sin(2.*np.pi*i/D)+periodic[...,1]*np.cos(2.*np.pi*i/D)
    return d

def G(x,t):
    """Ito Stochastic part of the system"""
    d=np.eye(D)*sigma
    return d

#if __name__=='__main__':
def generate_datasets(N=2000,ensemble_size=1):
    x0 = F1 * np.ones(D)  # Initial state (equilibrium)
    x0[0] += 0.01  # Add small perturbation to the first variable

    t = np.arange(-tskip, 0+dt*1e-5, dt)
    #x = odeint(L96, x0, t)
    x=itoint(L96, G, x0, t,Generator)

    x0=x[-1,:]
    #t = np.arange(0., tobs, dt)
    t = np.arange(0., N/tstep, dt)
    #x=itoint(L96, G, x0, t,Generator)

    # generate many time series withthe same external and initial conditions
    # take each tstep and first 2 variables as observed time series
    x=[itoint(L96, G, x0, t,Generator)[::tstep,:2] for i in range(ensemble_size)]
    raw_data=np.stack(x,axis=0)
    raw_time=t[::tstep,None] 
    raw_forcing=Fperiodic(t[::tstep])

    return raw_data.squeeze(), raw_time, raw_forcing

    ensemble=[]
#x = odeint(L96, x[-1,:], t)
    for i in range(ensemble_size):
        x=itoint(L96, G, x0, t,Generator)
        ensemble.append(x)#[::tstep,:2])
    ensemble=np.stack(ensemble,axis=0)
    #print(ensemble.shape)


    raw_time=t[::tstep,None] 
    raw_data=ensemble[:,::tstep,:2]
    raw_forcing=Fperiodic(t[::tstep])

    return raw_data, raw_time, raw_forcing
    
    # generate also test dataset by starting new random sequence with the same forcing from the same initial condition x0
    t = np.arange(0., tobs, dt)
    x1=itoint(L96, G, x0, t,Generator)
    raw_test_data=x1[::tstep,:2]   
    
    
    return raw_train_data, raw_forcing, raw_time, raw_test_data
    print('N=',len(raw_time))
 #   with open(os.path.join('data','raw_data.pkl3'),'wb') as f: pkl.dump(raw_data,f)
 #   with open(os.path.join('data','raw_forcing.pkl3'),'wb') as f: pkl.dump(raw_forcing,f)
 #   with open(os.path.join('data','raw_time.pkl3'),'wb') as f: pkl.dump(raw_time,f)
    #with open('raw_time.pkl3','wb') as f: pkl.dump(raw_time[:,0:0],f) #If we are going to consider stationary case

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(raw_time[:,0], raw_data[:, 0],'.-')
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x_1$")
    plt.show()

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(raw_time[:1000,0], raw_data[:1000, 0],'.-')
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x_1$")
    plt.show()

    # Plot the first three variables
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(raw_data[:1000, 0], raw_data[:1000, 1])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    plt.show()