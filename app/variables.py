import numpy as np
# from numpy import float32, array


###-----------------------------------------------------------------------------
# VARIABLES
# CLEAR_BACKGROUND = 1
# LEAVE_BACKGROUND = 0
MAX_AREA_OF_SINGLE_CELL = 0
MIN_AREA_OF_SINGLE_CELL = 0
FILTER_BLACK = 1
FILTER_WHITE = 0
mask_x = np.array([[-1],
                   [0],
                   [1]])
mask_y = np.array([[-1, 0, 1]])


## for testing purposes
exampleArray = np.array([[25, 100, 75, 49, 130],
                         [50, 80, 0, 70, 100],
                         [5, 10, 20, 30, 0],
                         [60, 50, 12, 24, 32],
                         [37, 53, 55, 21, 90],
                         [140, 17, 0, 23, 222]])
MALAexample = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
exampleKernel = np.array([[1, 0, 1],
                          [0, 1, 0],
                          [0, 0, 1]])


## may be useful
XSobelKernel = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]], np.float32)
YSobelKernel = np.array([[1, 2, 1], 
                         [0, 0, 0], 
                         [-1, -2, -1]], np.float32)

edgeDetection = np.array([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]])


