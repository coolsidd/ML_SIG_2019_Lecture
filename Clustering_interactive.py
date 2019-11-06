#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import DBSCAN


index = MAX_POINTS = pts = text = l1 = fig = None


def cluster_pts(num_pts=30):
    global index, MAX_POINTS, pts, text, l1, fig
    MAX_POINTS = min(num_pts, 50)
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    text = ax.text(0, 0, "", va="bottom", ha="left")
    pts = np.random.random((MAX_POINTS, 2)) * 10
    l1 = ax.scatter(pts[:, 0], pts[:, 1], c=[-1 for i in range(MAX_POINTS)])
    index = 0
    cid = fig.canvas.mpl_connect("button_press_event", onclick)


def onclick(event):
    add_point((event.xdata, event.ydata))


def add_point(pt):
    global index, MAX_POINTS, pts
    # print("Add pt " + str(pt))
    pts[index % MAX_POINTS][0], pts[index % MAX_POINTS][1] = pt
    clustering = DBSCAN(min_samples=2).fit(pts)
    unique_labels = set(clustering.labels_)
    colors = {
        i: plt.cm.hsv(x) for i, x in enumerate(np.linspace(0, 1, len(unique_labels)))
    }
    colors[-1] = (0, 0, 0, 1)
    # print(colors)
    color_arr = [colors[x] for x in clustering.labels_]
    n_clusters = len(set(unique_labels)) - (1 if -1 in unique_labels else 0)
    text.set_text("Estimated number of clusters: {}".format(n_clusters))
    l1.set_offsets(pts)
    l1.set_facecolor(color_arr)
    fig.canvas.draw_idle()
    index += 1
