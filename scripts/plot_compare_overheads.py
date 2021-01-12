import numpy as np
from matplotlib import pyplot as plt
from six import iterkeys
import seaborn as sns

import plot_settings

# Names and algorithms - could extract them from CSV but it's a ball-ache
uc_devices = ["GeForce\nGTX 1650\nLinux", "Jetson\nXavier NX\nLinux", "Titan\nRTX\nLinux"]
uc_algorithms = ["Python", "C++", "Python recording", "C++ recording"]

# Import data
# **NOTE** np.loadtxt doesn't handle empty entries
uc_data = np.genfromtxt("microcircuit_overheads.csv", delimiter=",", skip_header=1)
assert uc_data.shape[0] == (len(uc_devices) * len(uc_algorithms))

# Names and algorithms - could extract them from CSV but it's a ball-ache
izk_devices = ["GeForce\nGTX 1650\nLinux", "Titan\nRTX\nLinux", "Geforce\nGTX 1050 Ti\nWindows"]
izk_algorithms = ["Python", "Python: GPU stim", "Python: recording",
                         "Python: GPU stim & recording", "C++: GPU stim & recording", ]

# Import data
# **NOTE** np.loadtxt doesn't handle empty entries
izk_data = np.genfromtxt("izhikevich.csv", delimiter=",", skip_header=1)
assert izk_data.shape[0] == (len(izk_devices) * len(izk_algorithms))

izk_cpp_row = 4
izk_python_row = 3
izk_time_column = 4

uc_cpp_row = 3
uc_python_row = 2
uc_time_column = 3

uc_timesteps = 1000.0 / 0.1
izk_timesteps = 60.0 * 60.0 * 1000.0


uc_python_overhead = {d: (uc_data[(i * len(uc_algorithms)) + uc_python_row,uc_time_column] 
                          - uc_data[(i * len(uc_algorithms)) + uc_cpp_row,uc_time_column]) * 1000.0 / uc_timesteps
                      for i, d in enumerate(uc_devices)}

izk_python_overhead = {d: (izk_data[(i * len(izk_algorithms)) + izk_python_row,izk_time_column] 
                           - izk_data[(i * len(izk_algorithms)) + izk_cpp_row,izk_time_column]) * 1000.0 / izk_timesteps
                       for i, d in enumerate(izk_devices)}

# Build list of all devices
devices = list(set(iterkeys(uc_python_overhead)).union(set(iterkeys(izk_python_overhead))))

# Use them to build complete lists of overheads with zeros where there is no data
uc_python_overhead = [uc_python_overhead[d] if d in uc_python_overhead else 0.0
                      for d in devices]
izk_python_overhead = [izk_python_overhead[d] if d in izk_python_overhead else 0.0
                       for d in devices]
                      

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
uc_bars = axis.bar(bar_x[0::group_size], uc_python_overhead, bar_width, color=pal[0], linewidth=0)
izk_bars = axis.bar(bar_x[1::group_size], izk_python_overhead, bar_width, color=pal[1], linewidth=0)

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

plt.tight_layout(pad=0, rect=[0.0, 0.175, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/compare_overhead.pdf")
plt.show()