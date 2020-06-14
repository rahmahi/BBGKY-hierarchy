#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mrahaman@scholar.buruniv.ac.in
"""


import numpy as np
from scipy.sparse import dia_matrix
from math import *
from itertools import combinations
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import time as tm


#Pauli matrices
sig_x, sig_y, sig_z = \
  np.array([[0j, 1.0+0j], [1.0+0j, 0j]]), \
    np.array([[0j, -1j], [1j, 0j]]), \
      np.array([[1.0+0j, 0j], [0j, -1+0j]])
sig_plus = (sig_x + sig_y*1j)/2.0
sig_minus = (sig_x - sig_y*1j)/2.0

# Required parameters
N = 4      
t = np.linspace(0.0,1.0,20)
d_t = (1.0 - 0.0)/20.0
alpha = 0.2
omega = 0.0

# Initial values: ACCORDING TO MORI'S PAPER i.e. ZZX model
jx, jy, jz = 0.0, 0.0, -1.0 
hx, hy, hz = -1.0, 0.0, 0.0
sx, sy, sz = 0.0, 0.0, 1.0

# Periodic Boundary Condition
def jmat(N, alpha):
    J = dia_matrix((N, N))
    mid_diag = np.floor(N/2).astype(int)
    for i in np.arange(1,mid_diag+1):
        elem = pow(i, -alpha)
        J.setdiag(elem, k=i)
        J.setdiag(elem, k=-i)
    for i in np.arange(mid_diag+1, N):
        elem = pow(N-i, -alpha)
        J.setdiag(elem, k=i)
        J.setdiag(elem, k=-i)
    return J.toarray()
J = jmat(N, alpha)

gmat = np.zeros((N,3,3))
wmat = np.array([[0, 2 * (sz + hz), 0], [- 2 * (sz + hz), 0, 2 *hx],\
                  [0, - 2 * hx, 0]])
v_vec = np.array([[2 * sy], [- 2 * sx], [0.0]])
f_vec = np.array([[- sx * sz], [- sy * sz], [1 - sz * sz]])

def fsx(sx,sy,sz,N):
    dsx = 2.0 * sy * sz + 2 * hz * sy +\
                2 * np.sum(J[i,j] * gmat[j,1,2] \
                           for i in np.arange(N) for j in np.arange(1,N))
    return dsx
                                           
def fsy(sx,sy,sz,N):
    dsy  = -2.0 * sx * sz - 2 * hz * sx +\
    2 * hx * sz - 2 * (np.sum(J[i,j] *\
                                       gmat[j,0,2] for i in \
                                       np.arange(N) for j in np.arange(1,N)))
    return dsy

def fsz(sx,sy,sz,N):
    dsz = - 2.0 * hx * sy
    return dsz

def gcorr(k, a, b):
    left = np.sum(np.dot(wmat[a,c], gmat[k,c,b]) + np.dot(wmat[b,c], gmat[k,a,c]) for c in np.arange(3))
    mdl1 = v_vec[a] * np.sum(np.dot(J[0, j], gmat[np.absolute(j-k) + 1, 2, b]) for j in np.arange(1,k))
    mdl2 = v_vec[b] * np.sum(np.dot(J[k, j], gmat[j,a,2]) for j in np.arange(1,k))
    rght = J[0, k] * (np.dot(v_vec[a], f_vec[b]) + np.dot(v_vec[b], f_vec[a]))
    gc   = left + mdl1 + mdl2 + rght
    return gc

def g_t():
    gstate = gcorr(k,i,j)
    G = odeint(gcorr, gstate, d_t, args=())
    return G

def zzx_model(ddt, t):
    sx, sy, sz = ddt[0], ddt[1], ddt[2]
    v_vec = np.array([[2 * sy], [- 2 * sx], [0.0]])
    f_vec = np.array([[- sx * sz], [- sy * sz], [1 - sz * sz]])
    
    for p in np.arange(k):
        for q in np.arange(3):
            for kr in np.arange(3):
                gmat[p,q,r] = gcorr(p,q,r)
    
    dsx = 2.0 * sy * sz + 2 * hz * sy + \
                2 * np.sum(J[0,j] * gmat[j,1,2] \
                           for i in np.arange(N) for j in np.arange(1,N))
    dsy = -2.0 * sx * sz - 2 * hz * sx + \
                2 * hx * sz - 2 * (np.sum(J[0,j] *\
                                          gmat[j,0,2] for i in np.arange(N)\
                                          for j in np.arange(1,N)))
    dsz = - 2.0 * hx * sy
    ddt = [dsx, dsy, dsz]   
    
    return ddt

z0 = [0.0, 0.0, 1.0]
n = 500
t = np.linspace(0.0,1.0,20)

#vv = zzx_model()

print("code is okay!")