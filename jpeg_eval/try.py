import numpy as np
from scipy import signal

# Inputs
a = np.array([[1,2,3],[3,4,5],[5,6,7],[4,8,9]])
p = 0.6
mutate_index = np.random.choice(a=[False, True], size=(4,3), p=[p, 1-p])
arr = np.asarray(a,float)    
kernel = np.array([[0,1,0],
               [1,1,1],
               [0,1,0]]) 
arr = signal.convolve2d(arr, kernel, boundary='wrap', mode='same')/kernel.sum()
qtable = np.array(a)
qtable[mutate_index] = np.asarray(arr[mutate_index],int)

print(a)
print(arr)
print(qtable)
