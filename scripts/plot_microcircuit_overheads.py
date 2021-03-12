import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plot_settings

from itertools import chain, groupby
from six import iterkeys, itervalues

def autolabel(axis, rects, labels):
    for rect, label in zip(rects, labels):
        axis.annotate(label,
                      xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height()),
                      xytext=(0, 6),
                      textcoords="offset points",
                      ha="center", va="bottom", rotation=90)
                    
# Names and algorithms - could extract them from CSV but it's a ball-ache
devices = ["GeForce\nGTX 1650", "Jetson\nXavier NX", "Titan\nRTX", "Geforce\nGTX 1050 Ti"]

languages = ["PyGeNN", "GeNN"]
recording_method = ["CPU", "GPU"]

# Import data
# **NOTE** np.loadtxt doesn't handle empty entries
data = np.genfromtxt("microcircuit_overheads.csv", delimiter=",", skip_header=1)
assert data.shape[0] == (len(devices) * len(languages) * len(recording_method))

# List of time columns and associated legend text
time_columns = [6, 7, 8]
time_labels = ["Neuron simulation", "Synapse simulation", "Overhead"]

group_size = len(languages) * len(recording_method)
num_groups = len(devices)
num_bars = group_size * num_groups
bar_x = np.empty(num_bars)
bar_offset = np.zeros(num_bars)

bar_width = 0.8
group_width = 5.0
bar_pad = 0.1
group_pad = 0.75
start = 0.0

# Create figure
fig, axis = plt.subplots(figsize=(plot_settings.column_width, 
                                  95.0 * plot_settings.mm_to_inches))

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
for i, t in enumerate(time_columns):
    bars = axis.bar(bar_x, data[:,t], bar_width, bar_offset, color=pal[i], linewidth=0)
    legend_actors.append(bars[0])
    bar_offset += data[:,t]
    
    if i == 2:
        autolabel(axis, bars, languages * len(recording_method) * num_groups)

# Add real-time line
axis.axhline(1000.0, color="black", linestyle="--")

# Add device labels
for x, t in zip(group_x, devices):
    axis.text(x, -7500 if plot_settings.presentation else -3200.0, t, ha="center",
              fontsize=15 if plot_settings.presentation else 9)

# Get x coordinate of middle of each recording method
recording_method_x = np.reshape(bar_x, (8, 2))
recording_method_x = np.average(recording_method_x, axis=1)

# Configure axis
axis.set_xticks(recording_method_x)
axis.set_xticklabels(recording_method * num_groups)
axis.set_xlabel("Recording method")
axis.set_ylabel("Simulation time [ms]")
axis.set_ylim((0, 12000))

# Remove axis junk
sns.despine(ax=axis)
axis.xaxis.grid(False)

# Show figure legend with devices beneath figure
fig.legend(legend_actors, time_labels, ncol=2 if plot_settings.presentation else len(time_labels),
           frameon=False, loc="lower center")

plt.tight_layout(pad=0, rect=[0.0, 0.075, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/microcircuit_overheads.eps")
plt.show()
