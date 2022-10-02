# python_live_plot.py

import random
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# plt.style.use('fivethirtyeight')

# x_values = []
# y_values = []

# index = count()


# def animate(i):
#     x_values.append(next(index))
#     y_values.append(random.randint(0, 5))
#     # plt.cla()
#     plt.plot( y_values)


# ani = FuncAnimation(plt.gcf(), animate, 1000)


# plt.tight_layout()

with open('mujoco_costs.txt', 'r') as f:
    lines = f.readlines()
    x = [float(line.split()[0]) for line in lines]
    y = [float(line.split()[3]) for line in lines]
plt.plot(x ,y)
plt.show()