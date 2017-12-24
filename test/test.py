# per = 4
# li = list(range(0,10,per))
# for n,idx in enumerate(li):
#     if n == len(li)-1:
#         end = 10
#     else:
#         end = li[n+1]
#     print(n,idx, (idx,end))

import numpy as np

# types = [np.double, np.float32, np.float16, np.uint8, np.uint16]
# for t in types:
#     d = np.dtype(t)
#     print(d.name)

a = np.zeros((10,32,32,3))
s = np.zeros((1,4))
s[:] = a.shape
print(s)
ss = s.transpose((0, 2, 3, 1))
print(ss)