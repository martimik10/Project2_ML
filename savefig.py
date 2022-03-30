# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 18:43:13 2022

@author: Marti
"""
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate


plt.savefig('images/regress_Q2_V4.pdf',bbox_inches = 'tight')
