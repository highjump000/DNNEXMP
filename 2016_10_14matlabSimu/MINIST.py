import numpy as np

a = []

a.append([1.0,2.0,3.0])
a.append([21.0,31.0,41.0])

a = np.asarray(a)

b = []
for i in xange(3):
    b.append(a[:,i] -1)

b = np.asarray(b)
print b

