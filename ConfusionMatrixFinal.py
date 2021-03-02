from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import numpy as np


binary = np.array([[41653, 11651],
                   [1836, 6296]])

fig, ax = plot_confusion_matrix(conf_mat=binary,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()
print("Hey")

# TN, FP, FN, TP
