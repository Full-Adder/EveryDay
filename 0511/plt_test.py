import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,100)
x_1 = np.linspace(0,10,20)

plt.plot(x,np.sin(x))
plt.plot(x_1,np.sin(x_1),color = 'y')
plt.ylim(0,1)
plt.show()