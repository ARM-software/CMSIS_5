import numpy as np
from cmsisdsp.sdf.nodes.simu import *

a=np.zeros(10)
f=FIFO(10,a)

f.dump()

nb = 1 
for i in range(4):
    w=f.getWriteBuffer(2)
    w[0:2]=nb*np.ones(2)
    nb = nb + 1 
    f.dump()

print(a)

for i in range(4):
    w=f.getReadBuffer(2)
    print(w)