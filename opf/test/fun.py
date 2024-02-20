import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
bus_data = pd.read_csv("bus_data.csv", header=0)
gen_data = pd.read_csv("gen_data.csv", header=0)
bus = bus_data.sort_values(by="id", ignore_index=True)
gen = gen_data.sort_values(by="id", ignore_index=True)
bus.index= bus.id-1
gen.index = gen.id - 1
squ_sa_max = gen["sa_max"]**2
print(squ_sa_max)
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6]
y1 = [2, 4, 6, 8, 10, 12]
y2 = [12, 10, 8, 6, 4, 2]
result=cp.multiply(y1,y2)
print(result)
# Plotting
# bar_width = 0.35
# fig, ax = plt.subplots()
#
# bar1 = ax.bar([val - bar_width/2 for val in x], y1, bar_width, label='y1')
# bar2 = ax.bar([val + bar_width/2 for val in x], y2, bar_width, label='y2')
#
# # Add labels and title
# ax.set_xlabel('X Values')
# ax.set_ylabel('Y Values')
# ax.set_title('Group Barplot for X Values, Y1, and Y2')
# ax.set_xticks(x)
# ax.legend()
#
# # Show the plot
# plt.show()

