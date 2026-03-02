import cupy as cp

def expon(tau, tau_c):
    return cp.exp(-tau / tau_c)

def gaussian(tau, tau_c):
    arg = tau / tau_c
    return cp.exp(-arg * arg)