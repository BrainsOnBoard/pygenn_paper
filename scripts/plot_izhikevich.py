import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plot_settings

from itertools import chain, groupby
from six import iterkeys, itervalues

# Names and algorithms - could extract them from CSV but it's a ball-ache
devices = ["GeForce\nGTX 1650\nLinux", "Jetson\nXavier NX\nLinux", "Titan\nRTX\nLinux", "Geforce\nGTX 1050 Ti\nWindows"]
algorithms = ["Python", "Python: GPU stim", "Python: GPU recording",
              "Python: GPU stim & recording", "C++: GPU stim & recording", ]

# Import data
# **NOTE** np.loadtxt doesn't handle empty entries
data = np.genfromtxt("izhikevich.csv", delimiter=",", skip_header=1)
assert data.shape[0] == (len(devices) * len(algorithms))

# List of time columns and associated legend text
time_column = 4

group_size = len(algorithms)
num_groups = len(devices)
num_bars = group_size * num_groups
bar_x = np.empty(num_bars)

bar_width = 0.8
group_width = 5.0
bar_pad = 0.1
group_pad = 0.75
start = 0.0

# Create figure
fig, axis = plt.subplots(figsize=(plot_settings.column_width, 
                                  90.0 * plot_settings.mm_to_inches))

# Loop through each group (device) of bars
group_x = []
for d in range(0, num_bars, group_size):
    end = start + ((bar_width + bar_pad) * group_size)
    bar_x[d:d + group_size] = np.arange(start, end, bar_width + bar_pad)

    group_x.append(start + ((end - bar_width - start) * 0.5))

    # Update start for next group
    start = end + group_pad

pal = sns.color_palette("deep")
legend_actors = []
for i, a in enumerate(algorithms):
    bars = axis.bar(bar_x[i::group_size], data[i::group_size,time_column] / 1000.0, bar_width, color=pal[i], linewidth=0)
    legend_actors.append(bars[0])

# Configure axis
axis.set_xticks(group_x)
axis.set_xticklabels(devices, ha="center")
axis.set_ylabel("Simulation time [s]")

# Remove axis junk
sns.despine(ax=axis)
axis.xaxis.grid(False)

# Show figure legend with devices beneath figure
fig.legend(legend_actors, algorithms, ncol=2,
           frameon=False, loc="lower center")

plt.tight_layout(pad=0, rect=[0.0, 0.175, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/izhikevich.eps")
plt.show()
