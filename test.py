import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

x=np.linspace(-5,5,100)
y=(((x-5)/10)**2+(x-5)/10)

plt.plot(x,y)
plt.show()