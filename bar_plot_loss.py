import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('Linear regression', 'k-NN', 'ANN 1 layer', 'ANN 2 layer', 'SVM')
y_pos = np.arange(len(objects))
loss = [0.4048, 0.4708, 0.4591, 0.4178, 1.3081]

plt.bar(y_pos, loss, align='center', color=['black', 'red', 'green', 'blue', 'pink'])
plt.xticks(y_pos, objects)
plt.ylabel('MSE Loss')
plt.title('Comparision of machine learning models')

plt.show()
