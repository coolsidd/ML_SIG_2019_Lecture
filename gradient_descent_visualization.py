#!/usr/bin/env python3

# %matplotlib notebook
​
"""Taken from - Shathra/gradient-descent-demonstration/blob/master/gradient_descent_local_minima.py"""
​
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
​
p = np.poly1d([1, 0, 1])
​
# Function to minimize
def f(x):
    global p
    return np.polyval(p, x)
​
# Derivative of the function
def fd(x):
    global p
    return np.polyval(np.polyder(p), x)
​
x_min = -30
x_max = 30
x = np.linspace(x_min, x_max, 200)
y = f(x)
​
r = 0.01  # Learning rate
x_est = 25  # Starting point
​
y_est = f(x_est)
​
# Visualization Stuff
fig, ax = plt.subplots()
ax.set_xlim([x_min, x_max])
ax.set_ylim([-5, 500])
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
plt.title("Gradient Descent Local Minima")
line, = ax.plot([], [])
scat = ax.scatter([], [], c="red")
text = ax.text(-25,450,"")
​
##r = Slider(ax, "Learning Rate", 0, 1, 0.01)
​
def animate(i):
    global x_est
    global y_est
​
    # Gradient descent
    x_est = x_est - fd(x_est) * r
    y_est = f(x_est)
​
    # Update the plot
    scat.set_offsets([[x_est,y_est]])
    text.set_text("Value : %.2f" % y_est)
    line.set_data(x, y)
    return line, scat, text
​
def init():
    line.set_data([], [])
    return line,
​
try:
    ani = animation.FuncAnimation(fig, animate, 40, init_func=init, interval=100, blit=True)
except:
    print("Out of bounds.")
​
plt.show()
