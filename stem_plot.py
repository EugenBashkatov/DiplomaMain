"""
=========
Stem Plot
=========

`~.pyplot.stem` plots vertical lines from a baseline to the y-coordinate and
places a marker at the tip.
"""
import matplotlib.pyplot as plt
import numpy as np

def stemplot(x,y):
    plt.stem(x, y, use_line_collection=True)
    plt.show()
    return True

x = np.linspace(0.1, 2 * np.pi, 100)
y = np.exp(np.sin(x))
stemplot(x,y)
#############################################################################
#
# The position of the baseline can be adapted using *bottom*.
# The parameters *linefmt*, *markerfmt*, and *basefmt* control basic format
# properties of the plot. However, in contrast to `~.pyplot.plot` not all
# properties are configurable via keyword arguments. For more advanced
# control adapt the line objects returned by `~.pyplot`.
exit
markerline, stemlines, baseline = plt.stem(
    x, y, linefmt='grey', markerfmt='D', bottom=1.1, use_line_collection=True)
markerline.set_markerfacecolor('green')
plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

import matplotlib
matplotlib.pyplot.stem
matplotlib.axes.Axes.stem
