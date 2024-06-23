import cv2  # for finding contours to extract points of figure
import matplotlib.pyplot as plt  # for plotting and creating figures
import numpy as np  # for easy and fast number calculation
from math import tau  # tau is constant number = 2*PI
from scipy.integrate import quad_vec  # for calculating definite integral

import matplotlib.animation as animation  # for compiling animation and exporting video


# function to generate x+iy at given time t
def f(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j * y_list)


def create_path(points):
    len_points = points.shape[0]
    path = np.zeros((len_points, 2), dtype=np.int64)
    path[0] = points[0]
    added = 1
    not_added = points[1:].astype(np.int64)
    # vector_lengths = np.zeros((len_points,), dtype=np.float64)

    while added != len_points:
        dist = np.abs(path[added - 1] - not_added)
        r = 5
        empty_circle = True
        vector_len = np.zeros((dist.shape[0],), dtype=np.float64)
        while empty_circle:
            for i in range(dist.shape[0]):
                if dist[i, 0] <= r and dist[i, 1] <= r:
                    empty_circle = False
                    vector_len[i] = np.sqrt(dist[i, 0] ** 2 + dist[i, 1] ** 2)
                else:
                    vector_len[i] = 10000
            r += 5

        min_dist = np.argsort(vector_len)[0]#minimum value er index anbe
        path[added] = not_added[min_dist]
        # vector_lengths[added] = vector_len[min_dist]
        not_added = np.delete(not_added, min_dist, axis=0)
        added += 1

    return path


# reading the image and convert to greyscale mode
# ensure that you use image with black image with white background
img = cv2.imread("Lab4/hw/face.jpeg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img_gray, 100, 200)

edge_points = np.argwhere(edges == 255)#coordinates nibo shada pointgular

# Apply Canny edge detection
# edges = cv2.Canny(img_gray, 100, 200)

# Find the edge points
# edge_points = np.argwhere(edges == 255)  # Extract points where the edge is detected

# Create the path using the provided function
path = create_path(edge_points)

# Extract x and y coordinates from the path
x_list, y_list = path[:, 1], -path[:, 0]  # Invert y-coordinates to match the image coordinate system

# Visualize the detected edges
plt.plot(x_list, y_list)
plt.show()

# x_list, y_list = edge_points[:, 1], -edge_points[:, 0]

# center the contour to origin
x_list = x_list - np.mean(x_list)#plotke centre e anbo
y_list = y_list - np.mean(y_list)

# visualize the contour
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_list, y_list)

# later we will need these data to fix the size of figure
xlim_data = plt.xlim()
ylim_data = plt.ylim()

plt.show()

# time data from 0 to 2*PI as x,y is the function of time.
t_list = np.linspace(0, tau, len(x_list))  # now we can relate f(t) -> x,y

# Now find forier coefficient from -n to n circles
# ..., c-3, c-2, c-1, c0, c1, c2, c3, ...
order = 100  # -order to order i.e -100 to 100
# you can change the order to get proper figure
# too much is also not good, and too less will not produce good result

print("generating coefficients ")
# lets compute fourier coefficients from -order to order
c = []
# we need to calculate the coefficients from -order to order
for n in range(-order, order + 1):
    # calculate definite integration from 0 to 2*PI
    # formula is given in readme
    coef = 1 / tau * quad_vec(lambda t: f(t, t_list, x_list, y_list) * np.exp(-n * t * 1j), 0, tau, limit=100, full_output=1)[0]
    c.append(coef)
print("completed generating coefficients.")

# converting list into numpy array
c = np.array(c)
# this is to store the points of last circle of epicycle which draws the required figure
draw_x, draw_y = [], []

# make figure for animation
fig, ax = plt.subplots()

# different plots to make epicycle
# there are -order to order numbers of circles
circles = [ax.plot([], [], 'r-')[0] for i in range(-order, order + 1)]
# circle_lines are radius of each circles
circle_lines = [ax.plot([], [], 'b-')[0] for i in range(-order, order + 1)]
# drawing is plot of final drawing
drawing, = ax.plot([], [], 'k-', linewidth=2)

# original drawing
# orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)

# to fix the size of figure so that the figure does not get cropped/trimmed
ax.set_xlim(xlim_data[0] - 200, xlim_data[1] + 200)
ax.set_ylim(ylim_data[0] - 200, ylim_data[1] + 200)

# hide axes
ax.set_axis_off()

# to have symmetric axes
ax.set_aspect('equal')

# Set up formatting for the video file
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, metadata=dict(artist='Amrit Aryal'), bitrate=1800)

print("compiling animation ")
# set number of frames
frames = 300


# pbar = tqdm(total=frames)


# save the coefficients in order 0, 1, -1, 2, -2, ...
# it is necessary to make epicycles
def sort_coeff(coeffs):
    new_coeffs = []
    new_coeffs.append(coeffs[order])
    for i in range(1, order + 1):
        new_coeffs.extend([coeffs[order + i], coeffs[order - i]])
    return np.array(new_coeffs)


# make frame at time t
# t goes from 0 to 2*PI for complete cycle
def make_frame(i, time, coeffs):
    # global pbar
    # get t from time
    t = time[i]

    # exponential term to be multiplied with coefficient
    # this is responsible for making rotation of circle
    exp_term = np.array([np.exp(n * t * 1j) for n in range(-order, order + 1)])

    # sort the terms of fourier expression
    coeffs = sort_coeff(coeffs * exp_term)  # coeffs*exp_term makes the circle rotate.
    # coeffs itself gives only direction and size of circle

    # split into x and y coefficients
    x_coeffs = np.real(coeffs)
    y_coeffs = np.imag(coeffs)

    # center points for fisrt circle
    center_x, center_y = 0, 0

    # make all circles i.e epicycle
    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        # calculate radius of current circle
        r = np.linalg.norm([x_coeff, y_coeff])  # similar to magnitude: sqrt(x^2+y^2)

        # draw circle with given radius at given center points of circle
        # circumference points: x = center_x + r * cos(theta), y = center_y + r * sin(theta)
        theta = np.linspace(0, tau, num=50)  # theta should go from 0 to 2*PI to get all points of circle
        x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
        circles[i].set_data(x, y)

        # draw a line to indicate the direction of circle
        x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
        circle_lines[i].set_data(x, y)

        # calculate center for next circle
        center_x, center_y = center_x + x_coeff, center_y + y_coeff

    # center points now are points from last circle
    # these points are used as drawing points
    draw_x.append(center_x)
    draw_y.append(center_y)

    # draw the curve from last point
    drawing.set_data(draw_x, draw_y)

    # draw the real curve
    # orig_drawing.set_data(x_list, y_list)

    # update progress bar
    # pbar.update(1)


# make animation
# time is array from 0 to tau
time = np.linspace(0, tau, num=frames)
anim = animation.FuncAnimation(fig, make_frame, frames=frames, fargs=(time, c), interval=5)
plt.show()
# anim.save('epicycle.mp4', writer='ffmpeg', fps=30)
# anim.save('epicycle.gif', writer='pillow', fps=100)
# pbar.close()
# print("completed: epicycle.mp4")
