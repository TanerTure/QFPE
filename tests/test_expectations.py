'''
Tests for functions in expectations.py file
'''

import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),"..", "QFPE"))

import expectations as expect
import SHO

def test_trace():
    rho = np.eye(5) * 1/5
    assert np.allclose(expect.trace(rho), 1)

def test_x():
    rho = 1/2 * np.eye(2) 
    assert np.allclose(expect.x(rho), 0)
    rho = 1/2 * np.ones((2,2))
    assert np.allclose(expect.x(rho), np.sqrt(2)/2)

def test_p():
    rho = 1/2 * np.eye(2)
    assert np.allclose(expect.p(rho), 0)
    rho = 1/2 * np.ones((2,2))
    assert np.allclose(expect.p(rho), 0)
    rho = np.zeros((2,2), dtype=np.complex128)
    rho[0,0] = 0.25
    rho[0,1] = 0.25j
    rho[1,0] = -0.25j
    rho[1,1] = .75
    print(SHO.p(2), "is SHO.p")
    assert np.allclose(expect.p(rho), -np.sqrt(2)/4)


    
