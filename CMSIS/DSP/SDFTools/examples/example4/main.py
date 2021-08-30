import sched as s 
import matplotlib.pyplot as plt
from custom import *

# Only ONE FileSink can be used since the data will be dumped
# into this global buffer for display with Matplotlib
# It will have to be cleaned and reworked in future to use better
# mechanism of communication with the main code
DISPBUF = np.zeros(16000)

print("Start")
nb,error = s.scheduler(DISPBUF)
print("Nb sched = %d" % nb)

plt.figure()
plt.plot(DISPBUF)
plt.show()