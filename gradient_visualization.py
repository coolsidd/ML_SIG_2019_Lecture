#!/usr/bin/env python3

import numpy as np
from matplotlib.pyplot import *
import Polynomial_Fit

# y = mx + b
# m is slope, b is y-intercept
# def compute_error_for_line_given_points(b, m, points):
#     totalError = 0

#     for i in range(0, len(points)):
#         x = points[i, 0]
#         y = points[i, 1]
#         totalError += (y - (m * x + b)) ** 2
#     return totalError / float(len(points))
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


# def step_gradient(b_current, m_current, points, learningRate):
#     b_gradient = 0
#     m_gradient = 0
#     N = float(len(points))
#     for i in range(0, len(points)):
#         x = points[i, 0]
#         y = points[i, 1]
#         b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
#         m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
#     new_b = b_current - (learningRate * b_gradient)
#     new_m = m_current - (learningRate * m_gradient)
#     return [new_b, new_m]


def gradient_descent_runner(
    points, starting_b, starting_m, learning_rate, num_iterations, plotdata
):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
        if i % 10 == 0:
            plotdata.append(
                np.array([b, m, compute_error_for_line_given_points(b, m, points)])
            )
    return [b, m], plotdata


def update_plot(iterations, plotdata, plots):
    p1, p2, p3 = plots
    pass


def run():
    plotdata = []
    x = np.linspace(0, 10, 200)
    points = np.array([x, Polynomial_Fit.get_noisy_func(x, 1, lambda x: x)]).T
    learning_rate = 0.001
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess
    num_iterations = 10000
    print(
        "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(
            initial_b, initial_m, compute_error(initial_b, initial_m, points)
        )
    )
    print("Running...")
    [b, m], plotdata = gradient_descent_runner(
        points, initial_b, initial_m, learning_rate, num_iterations, plotdata
    )
    print(
        "After {0} iterations b = {1}, m = {2}, error = {3}".format(
            num_iterations, b, m, compute_error(b, m, points)
        )
    )
    plotdata = np.array(plotdata)
    l1 = subplot(1, 3, 1)
    xlabel("c")
    ylabel("cost")
    fig1 = plot(plotdata[:, 0], plotdata[:, 2])
    l2 = subplot(1, 3, 2)
    xlabel("m")
    ylabel("cost")
    fig2 = plot(plotdata[:, 1], plotdata[:, 2])
    subplot(1, 3, 3)
    xlabel("x")
    ylabel("y")
    fig3 = scatter(points[:, 0], points[:, 1])
    fig4 = plot(points[:, 0], m * points[:, 0] + b)
    print(fig1)
    interact(
        lambda iterations: update_plot(iterations, (fig1[0], fig2[0], fig4[0])),
        iterations=widgets.IntSlider(
            value=num_iterations, min=0, max=num_iterations, step=1
        ),
    )
    show()
    return plotdata


if __name__ == "__main__":
    run()
