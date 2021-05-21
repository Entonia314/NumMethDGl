import numpy as np

a = np.array([4,5])
b = np.array([1,2])
c = a-b

d = np.zeros((3,10,2))

"""print('Whole d: ', d)
print('First Slice: ', d[0,:,:])
print('Second Slice: ', d[:,0,:])
print('Third Slice: ', d[:,:,0])"""


[x0, y0], [v0, w0] = [1, 2], [3, 4]
[x1, y1], [v1, w1] = [2, 2], [0.5, -0.5]
[x2, y2], [v2, w2] = [-3, -3], [1, -1]
init = np.array([[[x0, y0], [v0, w0]], [[x1, y1], [v1, w1]], [[x2, y2], [v2, w2]]])

print(init[:,0,:])
