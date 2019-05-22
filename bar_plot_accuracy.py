import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('d', 'k-NN', 'ANN 1 layer', 'ANN 2 layer')
y_pos = np.arange(len(objects))
accuracy = [112.38, 54.41, 23.27, 12.46]

plt.bar(y_pos, accuracy, align='center', color=['blue', 'green', 'red', 'magenta'], width=1.0)
plt.xticks(y_pos, objects)
plt.ylabel('Time(s)')
plt.title('Computation time comparison')

plt.show()
