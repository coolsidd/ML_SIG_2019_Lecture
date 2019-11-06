#!/usr/bin/env python3

"""Taken from - Shathra/gradient-descent-demonstration/blob/master/gradient_descent_local_minima.py"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Function to minimize
def f(x):
    global p
    return np.polyval(p, x)


# Derivative of the function
def fd(x):
    global p
    return np.polyval(np.polyder(p), x)


##r = Slider(ax, "Learning Rate", 0, 1, 0.01)
##
def animate(i):
    global x_est
    global y_est

    # Gradient descent
    x_est = x_est - fd(x_est) * r
    y_est = f(x_est)

    # Update the plot
    scat.set_offsets([[x_est, y_est]])
    text.set_text("Value : %.2f" % y_est)
    line.set_data(x, y)
    return line, scat, text


def init():
    line.set_data([], [])
    return (line,)


def gradient_descent_animated(
    poly=[1, 0, 1], starting_point=25, learning_rate=0.01, range_of_vals=(-30, 30)
):
    global x_min, x_max, x, y, x_est, y_est, fig, ax, line, text, scat, p, r
    poly = list(reversed(poly))
    p = np.poly1d(poly)
    x_min, x_max = range_of_vals
    x = np.linspace(x_min, x_max, 200)
    y = f(x)
    r = learning_rate  # Learning rate
    x_est = starting_point  # Starting point
    y_est = f(x_est)
    # Visualization Stuff
    fig, ax = plt.subplots()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-5, 500])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    plt.title("Gradient Descent Local Minima")
    line, = ax.plot([], [])
    scat = ax.scatter([], [], c="red")
    text = ax.text(-25, 450, "")

    try:
        ani = animation.FuncAnimation(
            fig, animate, None, init_func=init, interval=100, blit=True
        )
    except:
        print("Out of bounds.")
    return HTML(ani.to_html5_video())
