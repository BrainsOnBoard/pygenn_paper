from matplotlib import gridspec as gs
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import sys
import seaborn as sns
import plot_settings

num_neurons = 1000
num_excitatory = int(0.8 * num_neurons)
num_inhibitory = int(0.2 * num_neurons)

duration_ms = 60 * 60 * 1000
bin_s = 0.05
display_time = 2000
raster_rate_ratio = 4
pal = sns.color_palette("deep")

def get_masks(times):
    return (np.where(times < 50000),
            np.where(times > (duration_ms - 50000)))

def plot_reward(axis, times, max_val):
    axis.vlines(times / 1000.0, 0, max_val, 
                linestyle="--", linewidth=2, color=pal[4])

def plot_stimuli(axis, times, ids):
    for t, i in zip(times, ids):
        colour = pal[2] if i == 0 else "black"
        axis.annotate("", xy=(t / 1000.0, num_neurons), xycoords="data", color=colour,
                      xytext=(0, 15.0), textcoords="offset points", annotation_clip=True, 
                      arrowprops=dict(facecolor=colour, edgecolor=colour, headlength=6.0))

def read_spikes(filename):
    return np.loadtxt(filename, delimiter=",", skiprows=1,
                      dtype={"names": ("time", "id"),
                             "formats": (np.float, np.int)})

def configure_axis_ticks(axis, min_time, max_time):
    axis.set_xticks(np.arange(np.floor(min_time), np.ceil(max_time), 0.5))
    axis.set_xlim((min_time, max_time))

# Read spikes
e_spikes = read_spikes("izhikevich_e_spikes.csv")
i_spikes = read_spikes("izhikevich_i_spikes.csv")

# Read stimuli
stimuli = np.loadtxt("izhikevich_stimulus_times.csv", delimiter=",",
                     dtype={"names": ("time", "id"),
                            "formats": (np.float, np.int)})

# Read rewards
reward_times = np.loadtxt("izhikevich_reward_times.csv", dtype=np.float)

# Get masks for events in first and last seconds
e_spike_first_second, e_spike_last_second = get_masks(e_spikes["time"])
i_spike_first_second, i_spike_last_second = get_masks(i_spikes["time"])
stimuli_first_second, stimuli_last_second = get_masks(stimuli["time"])
reward_times_first_second, reward_times_last_second = get_masks(reward_times)

# Find the earliest rewarded stimuli in first and last seconds
rewarded_stimuli_time_first_second = stimuli["time"][stimuli_first_second][np.where(stimuli["id"][stimuli_first_second] == 0)[0][0]]
rewarded_stimuli_time_last_second = stimuli["time"][stimuli_last_second][np.where(stimuli["id"][stimuli_last_second] == 0)[0][0]]

# Find the corresponding stimuli
corresponding_reward_first_second = reward_times[reward_times_first_second][np.where(reward_times[reward_times_first_second] > rewarded_stimuli_time_first_second)[0][0]]
corresponding_reward_last_second = reward_times[reward_times_last_second][np.where(reward_times[reward_times_last_second] > rewarded_stimuli_time_last_second)[0][0]]

padding_first_second = (display_time - (corresponding_reward_first_second - rewarded_stimuli_time_first_second)) / 2
padding_last_second = (display_time - (corresponding_reward_last_second - rewarded_stimuli_time_last_second)) / 2

min_first_second = (rewarded_stimuli_time_first_second - padding_first_second) / 1000.0
max_first_second = (corresponding_reward_first_second + padding_first_second) / 1000.0
min_last_second = (rewarded_stimuli_time_last_second - padding_last_second) / 1000.0
max_last_second = (corresponding_reward_last_second + padding_last_second) / 1000.0


# Create plot
figure = plt.figure(frameon=False, figsize=(plot_settings.column_width, 
                                            120.0 * plot_settings.mm_to_inches))

# Create outer gridspec with 2 rows
gsp = figure.add_gridspec(2, 1)

# Within each row, create inner gridspecs with zero spacing for raster/rate split
first_second_gs = gs.GridSpecFromSubplotSpec(raster_rate_ratio + 1, 1, subplot_spec=gsp[0], hspace=0.0)
last_second_gs = gs.GridSpecFromSubplotSpec(raster_rate_ratio + 1, 1, subplot_spec=gsp[1], hspace=0.0)

# Add raster and rate axes into inner gridspec
first_second_raster_axis = figure.add_subplot(first_second_gs[0:raster_rate_ratio])
first_second_rate_axis = figure.add_subplot(first_second_gs[raster_rate_ratio])
last_second_raster_axis = figure.add_subplot(last_second_gs[0:raster_rate_ratio])
last_second_rate_axis = figure.add_subplot(last_second_gs[raster_rate_ratio])

# Create list of all axes
axes = [first_second_raster_axis, first_second_rate_axis, last_second_raster_axis, last_second_rate_axis]

# Plot spikes that occur in first second
first_second_bins = np.arange(0.0, 50.0, bin_s)
first_second_spikes = np.concatenate((e_spikes["time"][e_spike_first_second] / 1000.0, i_spikes["time"][i_spike_first_second] / 1000.0))
first_second_rates = np.histogram(first_second_spikes, bins=first_second_bins)[0] * (1.0 / bin_s) * (1.0 / num_neurons)

first_second_raster_axis.scatter(e_spikes["time"][e_spike_first_second] / 1000.0, e_spikes["id"][e_spike_first_second], 
                                 s=1, edgecolors="none", color=pal[3], rasterized=True)
first_second_raster_axis.scatter(i_spikes["time"][i_spike_first_second] / 1000.0, i_spikes["id"][i_spike_first_second] + num_excitatory, 
                                 s=1, edgecolors="none", color=pal[0], rasterized=True)
first_second_rate_axis.bar(first_second_bins[:-1], first_second_rates, width=bin_s, align="edge")

# Plot spikes that occur in final second
last_second_bins = np.arange((duration_ms - 50000.0) / 1000.0, duration_ms / 1000.0, bin_s)
last_second_spikes = np.concatenate((e_spikes["time"][e_spike_last_second] / 1000.0, i_spikes["time"][i_spike_last_second] / 1000.0))
last_second_rates = np.histogram(last_second_spikes, bins=last_second_bins)[0] * (1.0 / bin_s) * (1.0 / num_neurons)

last_second_raster_axis.scatter(e_spikes["time"][e_spike_last_second] / 1000.0, e_spikes["id"][e_spike_last_second], 
                                s=1, edgecolors="none", color=pal[3], rasterized=True)
last_second_raster_axis.scatter(i_spikes["time"][i_spike_last_second] / 1000.0, i_spikes["id"][i_spike_last_second] + num_excitatory, 
                                s=1, edgecolors="none", color=pal[0], rasterized=True)
last_second_rate_axis.bar(last_second_bins[:-1], last_second_rates, width=bin_s, align="edge")

# Remove axis junk
for ax in axes:
    sns.despine(ax=ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

# Turn off axis ticks and labels for raster axes
first_second_raster_axis.tick_params(bottom=False, labelbottom=False)
last_second_raster_axis.tick_params(bottom=False, labelbottom=False)

# Configure axes
first_second_raster_axis.set_title("A", loc="left")
last_second_raster_axis.set_title("B", loc="left")
configure_axis_ticks(first_second_raster_axis, min_first_second, max_first_second)
configure_axis_ticks(first_second_rate_axis, min_first_second, max_first_second)
configure_axis_ticks(last_second_raster_axis, min_last_second, max_last_second)
configure_axis_ticks(last_second_rate_axis, min_last_second, max_last_second)

first_second_rate_axis.set_yticks([0, 10, 20])
first_second_rate_axis.set_ylim((0, 30))
last_second_rate_axis.set_yticks([0, 10, 20])
last_second_rate_axis.set_ylim((0, 30))

first_second_rate_axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
last_second_rate_axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
first_second_raster_axis.set_ylim((0, num_neurons))
last_second_raster_axis.set_ylim((0, num_neurons))
first_second_raster_axis.set_ylabel("Neuron number")
last_second_raster_axis.set_ylabel("Neuron number")
first_second_rate_axis.set_ylabel("Rate\n[spike/s]")
last_second_rate_axis.set_ylabel("Rate\n[spike/s]")
first_second_rate_axis.set_xlabel("Time [s]")
last_second_rate_axis.set_xlabel("Time [s]")

# Align y axes labels
figure.align_ylabels(axes)
                   
# Fit tight layout
# **NOTE** for whatever reason, this doesn't play nicely with  
# annotations so we need to do this first, manually making sure 
# there is enough space BEFORE adding annotation
figure.tight_layout(pad=0, rect=[0.0, 0.0, 0.99, 0.98], h_pad=1.0)

# Plot reward times and rewarded stimuli that occur in first second
plot_reward(first_second_raster_axis, reward_times[reward_times_first_second], num_neurons);
plot_reward(first_second_rate_axis, reward_times[reward_times_first_second], 30);
plot_stimuli(first_second_raster_axis, stimuli["time"][stimuli_first_second], stimuli["id"][stimuli_first_second])

# Plot reward times and rewarded stimuli that occur in final second
plot_reward(last_second_raster_axis, reward_times[reward_times_last_second], num_neurons);
plot_reward(last_second_rate_axis, reward_times[reward_times_last_second], 30);
plot_stimuli(last_second_raster_axis, stimuli["time"][stimuli_last_second], stimuli["id"][stimuli_last_second])

if not plot_settings.presentation:
    figure.savefig("../figures/izhikevich_spikes.pdf", dpi=600)
plt.show()

