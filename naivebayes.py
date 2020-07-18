#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY


def naivebayes(x, y, x1):
    # =============================================================================
    # function logratio = naivebayes(x,y,x1);
    #
    # Computation of log P(Y|X=x1) using Bayes Rule
    # Input:
    # x : n input vectors of d dimensions (dxn)
    # y : n labels (-1 or +1)
    # x1: input vector of d dimensions (dx1)
    #
    # Output:
    # logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))
    # =============================================================================

    # Convertng input matrix x and x1 into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)
    X1 = np.matrix(x1)

    # Pre-configuring the size of matrix X
    d, n = X.shape

    # =============================================================================
    # fill in code here
    X1conj = 1 - X1
    Xpos, Xneg = naivebayesPXY(X, y)
    Xpos0 = 1 - Xpos
    Xneg0 = 1 - Xneg

    t1 = [a for a in range(X1.size) if X1[a,0]==1]
    t2 = [a for a in range(X1conj.size) if X1conj[a,0]==1]
    # test1  = Xpos[np.nonzero(X1)]
    # test2 = Xpos0[np.nonzero(X1conj)]
    X1pos = np.prod(Xpos[t1]) * np.prod(Xpos0[t2])
    X1neg = np.prod(Xneg[t1]) * np.prod(Xneg0[t2])

    # TX1pos = X1.T.dot(Xpos)
    # TX1neg = X1.T.dot(Xneg)

    Ypos, Yneg = naivebayesPY(X, y)

    logratio = np.log((X1pos * Ypos) / (X1neg * Yneg))
    # Tlogratio = np.log((TX1pos * Ypos) / (TX1neg * Yneg))
    return logratio
# =============================================================================