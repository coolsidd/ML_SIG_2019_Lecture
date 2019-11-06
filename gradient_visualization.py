#!/usr/bin/env python3

import numpy as np
from matplotlib.pyplot import *
import Polynomial_Fit
from ipywidgets import interact, widgets

# y = mx + b
# m is slope, b is y-intercept


def compute_error(b, m, points):
    y = points[:, 1]
    x = points[:, 0]
    totalError = np.sum((y - (m * x + b)) ** 2)
    return totalError / points.shape[0]


def step_gradient(b_current, m_current, points, learningRate):
    N = points.shape[0]
    y = points[:, 1]
    x = points[:, 0]
    b_gradient = np.sum(-(2 / N) * (y - ((m_current * x) + b_current)))
    m_gradient = np.sum(-(2 / N) * x * (y - ((m_current * x) + b_current)))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(
    points, starting_b, starting_m, learning_rate, num_iterations, plotdata
):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
        if i % 10 == 0:
            plotdata.append(np.array([b, m, compute_error(b, m, points)]))
    return [b, m], plotdata


def update_plot(iterations, fig, plotdata, info, plots):
    #    p1, p2, p3 = plots
    p3 = plots
    in_b, in_m, lr, points = info

    [b, m], plotdata = gradient_descent_runner(
        points, in_b, in_m, lr, iterations, plotdata
    )

    np_plotdata = np.array(plotdata)

    #p1.set_ydata(np_plotdata[:iterations, 2])

    #p2.set_ydata(np_plotdata[:iterations, 2])

    p3.set_ydata(m * points[:, 0] + b)

    fig.canvas.draw_idle()


def get_line(
    m_slope=1, c_intercept=0, initial_b=0, initial_m=0, learning_rate=0.001, num_pts=150, num_itr=200
):
    num_pts = min(200, num_pts)
    fig = figure()
    plotdata = []
    x = np.linspace(1, 50, num_pts)
    points = np.array(
        [
            x,
            Polynomial_Fit.get_noisy_func(
                x, 1, lambda x: m_slope * x + c_intercept, num_pts=num_pts
            ),
        ]
    ).T
    #    initial_b = 0  # initial y-intercept guess
    #    initial_m = 0  # initial slope guess
    num_iterations = num_itr
    print(
        "Starting gradient descent at:\n b = {0}, m = {1}, error = {2}".format(
            initial_b, initial_m, compute_error(initial_b, initial_m, points)
        )
    )
    [b, m], plotdata = gradient_descent_runner(
        points, initial_b, initial_m, learning_rate, num_iterations, plotdata
    )
    print(
        "After {0} iterations:\n b = {1}, m = {2}, error = {3}".format(
            num_iterations, b, m, compute_error(b, m, points)
        )
    )
    fig.tight_layout()
    np_plotdata = np.array(plotdata)
    l1 = subplot2grid((2, 2), (0, 0),rowspan=2)
    xlabel("iterations (10x)")
    ylabel("cost")
    fig1 = plot(np_plotdata[:, 2])
#    l2 = subplot2grid((2, 2), (1, 0))
#    xlabel("iterations (10x)")
#    ylabel("cost")
#    fig2 = plot(np_plotdata[:, 2])
    l3 = subplot2grid((2, 2), (0, 1), rowspan=2)
    xlabel("x")
    ylabel("y")
    print(np_plotdata.shape)
    fig3 = scatter(points[:, 0], points[:, 1], alpha=0.6)
    fig4 = plot(points[:, 0], m * points[:, 0] + b, c="r")
    interact(
        #        lambda iterations: update_plot(iterations, fig, plotdata, [initial_b, initial_m, learning_rate,points], (fig1[0], fig2[0], fig4[0])),
        lambda iterations: update_plot(
            iterations,
            fig,
            plotdata,
            [initial_b, initial_m, learning_rate, points],
            (fig4[0]),
        ),
        iterations=widgets.IntSlider(
            value=num_iterations, min=0, max=num_iterations, step=1
        ),
    )
    show()
