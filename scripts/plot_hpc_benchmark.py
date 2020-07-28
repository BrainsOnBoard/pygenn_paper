import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plot_settings

# Import data
data = np.loadtxt("hpc_benchmark.csv", delimiter=",", skiprows=1,
                  dtype={"names": ["device", "scale", "num_neurons", "num_synapses", "num_plastic_synapses", "platform", "sim_time", "init_time",
                                   "sparse_init_time", "neuron_sim_time","presynaptic_update_time", "postsynaptic_update_time", "overhead"],
                         "formats": ["U64", np.float64, np.uint64, np.uint64, np.uint64, "U64", np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]})

# Get unique devices
devices = np.unique(data["device"])

# Create figure
fig, axis = plt.subplots(figsize=(plot_settings.column_width, 
                                  90.0 * plot_settings.mm_to_inches))
axis.axhline(0.35, color="black", linestyle="--")

# Loop through devices
legend_actors = []
legend_text = []
for d in devices:
    # Select device data
    device_data = data[data["device"] == d]

    legend_actors.append(axis.plot(device_data["num_plastic_synapses"], device_data["sim_time"] / 1000.0, marker="x")[0])
    
    
    legend_text.append(d)


axis.set_xlabel("Number of plastic synapses")
axis.set_ylabel("Simulation time [s]")

# Remove axis junk
sns.despine(ax=axis)
axis.xaxis.grid(False)
axis.set_xscale("log")
axis.set_yscale("log")

# Show figure legend with devices beneath figure
fig.legend(legend_actors, legend_text, ncol=len(devices), frameon=False, loc="lower center")

plt.tight_layout(pad=0, rect=[0.0, 0.075, 1.0, 1.0])
if not plot_settings.presentation:
    fig.savefig("../figures/hpc_benchmark.pdf")
plt.show()