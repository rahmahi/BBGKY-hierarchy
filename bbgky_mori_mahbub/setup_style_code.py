#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:47:48 2020

        BBGKY heirachy for quantum
        spins and transverse fields with time-periodic drive
        * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        * Copyright (c) 2020 Mahbub Rahaman (mrahaman@scholar.buruniv.ac.in)
        *
        *This is free software: you can redistribute it and/or modify
        *it under the terms of version 2 of the GNU Lesser General
        *Public License as published by the Free Software Foundation.
        * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

"""

from __future__ import division, print_function

__version__   = '0.1'
__author__    = 'Mahbub Rahaman'
__credits__   = 'Department of Physics, The University of Burdwan'

import numpy as np
from pprint import pprint
from itertools import combinations
from scipy.integrate import odeint
from tempfile import mkdtemp
import os.path as path

desc = """ BBGKY hierachy Dynamics of Curie-Weiss model with long range interactions
       and periodic drive"""

try:
    # Python 2 forward compatibility
    range = xrange
except NameError:
    pass

#Pauli matrices
sig_x, sig_y, sig_z = \
  np.array([[0j, 1.0+0j], [1.0+0j, 0j]]), \
    np.array([[0j, -1j], [1j, 0j]]), \
      np.array([[1.0+0j, 0j], [0j, -1+0j]])
sig_plus = (sig_x + sig_y*1j)/2.0
sig_minus = (sig_x - sig_y*1j)/2.0


class ParamData:
    description = """Class to store parameters and hopping matrix"""
    def __init__(self, hopmat = np.eye(11), lattice_size=11, omega=0.0, \
                                      times = np.linspace(0.0, 10.0,100),\
                                       hx=0.0, hy=0.0, hz=0.0,\
                                       sigx=0.0, sigy=0.0, sigz=0.0,\
                                        jx=0.0, jy=0.0, jz=1.0, hdc=0.0,\
                                        memmap=False, verbose=False):
        self.lattice_size = lattice_size
        self.omega = omega
        self.times = times
        self.jx, self.jy, self.jz = jx, jy, jz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.verbose = verbose
        self.jmat = hopmat
        self.memmap = memmap
        self.norm = 1.0 #Change to kac norm later


class equation:
    description = """ Precalcuates dynamics of the terms"""
    
    def g_term():
        def wmat(self, param):
            wmat = np.array([[0, 2 *(sigz + hz), 0], [-2 * (sigz + hz), 0 ,\
                              2 * hx], [0, -2 * hx, 0]])
            return wmat
