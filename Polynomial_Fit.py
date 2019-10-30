#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets

NUM_POINTS = 200
func = None


def get_noisy_func(x, variance):

    y = func(x)
    noise = np.random.normal(
        0, (np.max(y) - np.min(y)) * 0.2 * variance, int(NUM_POINTS)
    )
    y = y + noise
    return y


def get_polynomial_fit(x, y, degree):

    y_hat_coeff = np.polyfit(x, y, degree)
    y_hat_poly = lambda x: np.sum(
        [y_hat_coeff[-i - 1] * (x ** i) for i in range(y_hat_coeff.shape[0])], axis=0
    )
    y_new = y_hat_poly(x)
    return y_new


fig = ax = None
# fig, ax = plt.subplots()
# fig.tight_layout()
l1 = l2 = None
# plt.subplots_adjust(bottom=0.25)
# ax.margins(x=0)
x = y = y_hat = degree = None


def update_degree(val):
    global degree
    degree = val
    y_hat = get_polynomial_fit(x, y, degree)
    l2[0].set_ydata(y_hat)
    fig.canvas.draw_idle()


def update_noise(degree, val):
    noise = val
    y = get_noisy_func(x, noise)
    y_hat = get_polynomial_fit(x, y, degree)
    l1.set_offsets(np.c_[x, y])
    l2[0].set_ydata(y_hat)
    fig.canvas.draw_idle()


def reset(event):
    snoise.reset()
    sdegree.reset()


def poly_regression(func_to_fit=np.sqrt, x_range=(1, 50), num_points=None):
    global NUM_POINTS
    global l1, l2, x, y, y_hat, fig, ax, func
    func = func_to_fit
    if num_points is not None:
        NUM_POINTS = num_points
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots()
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    axcolor = "lightgoldenrodyellow"
    # noise = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    # degree = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    # snoise = Slider(noise, "Noise", 0, 1.0, valinit=0.2)
    # sdegree = Slider(degree, "Degree", 1, 30, valinit=3, valstep=1)

    x = np.linspace(x_range[0], x_range[1], NUM_POINTS)
    y = get_noisy_func(x, 0.2)
    y_hat = get_polynomial_fit(x, y, 3)
    l1 = ax.scatter(x, y, alpha=0.8)
    l2 = ax.plot(x, y_hat, c="r", lw=3)

    # snoise.on_changed(lambda val: update_noise(sdegree.val, val))
    # sdegree.on_changed(update_degree)

    interact(update_degree, val=widgets.IntSlider(value=2, min=1, max=30, step=1))
    widgets.interact(
        lambda noise: update_noise(degree, noise),
        noise=widgets.FloatSlider(value=0.3, min=0, max=1, step=0.001),
    )
    # resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    # button = Button(resetax, "Reset", color=axcolor, hovercolor="0.975")
    # button.on_clicked(reset)

    plt.show()
