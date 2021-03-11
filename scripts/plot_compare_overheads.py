import numpy as np
from matplotlib import pyplot as plt
from six import iterkeys
import seaborn as sns

import plot_settings

# Names of devices
devices = ["GeForce\nGTX 1650\nLinux", "Jetson\nXavier NX\nLinux", "Titan\nRTX\nLinux", "Geforce\nGTX 1050 Ti\nWindows"]

# Import data
# **NOTE** np.loadtxt doesn't handle empty entries
data = np.genfromtxt("compare_overheads.csv", delimiter=",", skip_header=1)
assert data.shape[0] == len(devices)

uc_timesteps = 1000.0 / 0.1
izk_timesteps = 60.0 * 60.0 * 1000.0
num_repeats = 5

# Extract times
times = data[:,1:21]

# Split data into seperate matrices for each experiment
experiment_data = [times[:,list(range(i, num_repeats * 4, 4))] 
                   for i in range(4)]

# Convert all data into microseconds
experiment_data[0] *= 1E6
experiment_data[1] *= 1E3
experiment_data[2] *= 1E6
experiment_data[3] *= 1E3

# Divide by numbers of timesteps
experiment_data[0] /= uc_timesteps
experiment_data[1] /= uc_timesteps
experiment_data[2] /= izk_timesteps
experiment_data[3] /= izk_timesteps

# Calculate mean across all repeates
experiment_mean = np.vstack([np.mean(e, axis=1) for e in experiment_data])

experiment_differences = np.vstack((experiment_mean[1,:] - experiment_mean[0,:],
                                    experiment_mean[3,:] - experiment_mean[2,:]))

group_size = 2
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
uc_bars = axis.bar(bar_x[0::group_size], experiment_differences[0], bar_width, color=pal[0], linewidth=0)
izk_bars = axis.bar(bar_x[1::group_size], experiment_differences[1], bar_width, color=pal[1], linewidth=0)

# Configure axis
axis.set_xticks(group_x)
axis.set_xticklabels(devices, ha="center")
axis.set_ylabel(r"Python overhead per-timestep [$\mu$ s]")

# Remove axis junk
sns.despine(ax=axis)
axis.xaxis.grid(False)

# Show figure legend with devices beneath figure
fig.legend([uc_bars, izk_bars], ["Microcircuit", "Pavlovian conditioning"], ncol=2,
           frameon=False, loc="lower center")

plt.tight_layout(pad=0, rect=[0.0, 0.1, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/compare_overhead.eps")

plt.show()
