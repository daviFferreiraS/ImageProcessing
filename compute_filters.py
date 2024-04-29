import numpy as np
import imageio
import matplotlib.pyplot as plt

def conv_point(f, w, x, y, debug=False):
    n,m = w.shape
    a = int((n-1)/2)
    b = int((m-1)/2)
    # gets submatrix of pixel neighbourhood
    sub_f = f[ x-a : x+a+1 , y-b:y+b+1 ]

    # flips the filter
    w_flip = np.flip( np.flip(w, 0) , 1)
    
    # conditional for debugging the function by showing the arrays
    if (debug==True):
        print("sub-image f:\n" + str(sub_f))
        print("\nflipped filter w:\n" + str(w_flip))
    
    # performs convolution (without converting to int)
    value = np.sum( np.multiply(sub_f, w_flip))
    return value

image = np.matrix( [ [25, 50, 1, 4], [255, 52, 2, 5], [255,100,0,3], [225, 100,3,120] ])

w1 = np.matrix( [ [1,1,1], [1, -8, 1], [1,1,1] ] )
w2 = np.matrix( [ [0,1/10,0], [1/10, 6/10, 1/10], [0,1/10,0] ] )

print(conv_point(image, w1, 1, 1))
print(conv_point(image, w1, 2, 2))
print(conv_point(image, w2, 1, 2))


