import numpy as np 
import matplotlib.pyplot as plt 

from pygenn import genn_model, genn_wrapper
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from scipy.stats import norm
from six import iteritems, itervalues
from time import perf_counter

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
TIMESTEP_MS = 1.0

# Should we rebuild model
BUILD_MODEL = False

# Generate code for kernel timing
MEASURE_TIMING = False

# Use GeNN's built in spike recording system
USE_GENN_RECORDING = True

# Simulation duration
DURATION_MS = 60.0 * 60.0 * 1000.0

# How much of start and end of simulation to record
# **NOTE** we want to see at least one rewarded stimuli in each recording window
RECORD_TIME_MS = 50.0 * 1000.0

DISPLAY_TIME_MS = 2000.0

# How often should outgoing weights from each synapse be recorded
WEIGHT_RECORD_INTERVAL_MS = 10.0 * 1000.0

# STDP params
TAU_D = 200.0

# Scaling
SIZE_SCALE_FACTOR = 1
WEIGHT_SCALE_FACTOR = 1.0 / SIZE_SCALE_FACTOR

# Scaled number of cells
NUM_EXCITATORY = 800 * SIZE_SCALE_FACTOR
NUM_INHIBITORY = 200 * SIZE_SCALE_FACTOR
NUM_CELLS = NUM_EXCITATORY + NUM_INHIBITORY

# Weights
INH_WEIGHT = -1.0 * WEIGHT_SCALE_FACTOR
INIT_EXC_WEIGHT = 1.0 * WEIGHT_SCALE_FACTOR
MAX_EXC_WEIGHT = 4.0 * WEIGHT_SCALE_FACTOR
DOPAMINE_STRENGTH = 0.5 * WEIGHT_SCALE_FACTOR

# Connection probability
PROBABILITY_CONNECTION = 0.1

# Input sets
NUM_STIMULI_SETS = 100
STIMULI_SET_SIZE = 50
STIMULI_CURRENT = 40.0

# Regime
MIN_INTER_STIMULI_INTERVAL_MS = 100.0
MAX_INTER_STIMULI_INTERVAL_MS = 300.0

# Reward
REWARD_DELAY_MS = 1000.0

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
def convert_ms_timestep(ms):
    return int(round(ms / TIMESTEP_MS))

def get_start_end_stim(stim_counts):
    end_stimuli = np.cumsum(stim_counts)
    start_stimuli = np.empty_like(end_stimuli)
    start_stimuli[0] = 0
    start_stimuli[1:] = end_stimuli[0:-1]
    
    return start_stimuli, end_stimuli

def plot_reward(axis, times):
    for t in times:
        axis.annotate("reward",
                      xy=(t, 0), xycoords="data",
                      xytext=(0, -15.0), textcoords="offset points",
                      arrowprops=dict(facecolor="black", headlength=6.0),
                      annotation_clip=True, ha="center", va="top")

def plot_stimuli(axis, times):
    for t, i in times:
        colour = "green" if i == 0 else "black"
        axis.annotate("S%u" % i,
                      xy=(t, NUM_CELLS), xycoords="data",
                      xytext=(0, 15.0), textcoords="offset points",
                      arrowprops=dict(facecolor=colour, edgecolor=colour, headlength=6.0),
                      annotation_clip=True, ha="center", va="bottom", color=colour)

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
izhikevich_dopamine_model = genn_model.create_custom_neuron_class(
    "izhikevich_dopamine",

    param_names=["a", "b", "c", "d", "tauD", "dStrength"],
    var_name_types=[("V", "scalar"), ("U", "scalar"), ("D", "scalar")],
    extra_global_params=[("rewardTimesteps", "uint32_t*")],
    sim_code=
        """
        $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT; //at two times for numerical stability
        $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT;
        $(U)+=$(a)*($(b)*$(V)-$(U))*DT;
        const unsigned int timestep = (unsigned int)($(t) / DT);
        const bool injectDopamine = (($(rewardTimesteps)[timestep / 32] & (1 << (timestep % 32))) != 0);
        if(injectDopamine) {
           const scalar dopamineDT = $(t) - $(prev_seT);
           const scalar dopamineDecay = exp(-dopamineDT / $(tauD));
           $(D) = ($(D) * dopamineDecay) + $(dStrength);
        }
        """,
    threshold_condition_code="$(V) >= 30.0",
    reset_code=
        """
        $(V)=$(c);
        $(U)+=$(d);
        """)

stim_noise_model = genn_model.create_custom_current_source_class(
    "stim_noise",

    param_names=["n", "stimMagnitude"],
    var_name_types=[("startStim", "unsigned int"), ("endStim", "unsigned int", VarAccess_READ_ONLY)],
    extra_global_params=[("stimTimes", "scalar*")],
    injection_code=
        """
        scalar current = ($(gennrand_uniform) * $(n) * 2.0) - $(n);
        if($(startStim) != $(endStim) && $(t) >= $(stimTimes)[$(startStim)]) {
           current += $(stimMagnitude);
           $(startStim)++;
        }
        $(injectCurrent, current);
        """)

izhikevich_stdp_tag_update_code="""
    // Calculate how much tag has decayed since last update
    const scalar tagDT = $(t) - tc;
    const scalar tagDecay = exp(-tagDT / $(tauC));
    // Calculate how much dopamine has decayed since last update
    const scalar dopamineDT = $(t) - $(seT_pre);
    const scalar dopamineDecay = exp(-dopamineDT / $(tauD));
    // Calculate offset to integrate over correct area
    const scalar offset = (tc <= $(seT_pre)) ? exp(-($(seT_pre) - tc) / $(tauC)) : exp(-(tc - $(seT_pre)) / $(tauD));
    // Update weight and clamp
    $(g) += ($(c) * $(D_pre) * $(scale)) * ((tagDecay * dopamineDecay) - offset);
    $(g) = fmax($(wMin), fmin($(wMax), $(g)));
    """
izhikevich_stdp_model = genn_model.create_custom_weight_update_class(
    "izhikevich_stdp",
    
    param_names=["tauPlus",  "tauMinus", "tauC", "tauD", "aPlus", "aMinus",
                 "wMin", "wMax"],
    derived_params=[
        ("scale", genn_model.create_dpf_class(lambda pars, dt: 1.0 / -((1.0 / pars[2]) + (1.0 / pars[3])))())],
    var_name_types=[("g", "scalar"), ("c", "scalar")],

    sim_code=
        """
        $(addToInSyn, $(g));
        // Calculate time of last tag update
        const scalar tc = fmax($(prev_sT_pre), fmax($(prev_sT_post), $(prev_seT_pre)));
        """
        + izhikevich_stdp_tag_update_code +
        """
        // Decay tag and apply STDP
        scalar newTag = $(c) * tagDecay;
        const scalar dt = $(t) - $(sT_post);
        if (dt > 0) {
            scalar timing = exp(-dt / $(tauMinus));
            newTag -= ($(aMinus) * timing);
        }
        // Write back updated tag and update time
        $(c) = newTag;
        """,
    event_code=
        """
        // Calculate time of last tag update
        const scalar tc = fmax($(sT_pre), fmax($(prev_sT_post), $(prev_seT_pre)));
        """
        + izhikevich_stdp_tag_update_code +
        """
        // Decay tag
        $(c) *= tagDecay;
        """,
    learn_post_code=
        """
        // Calculate time of last tag update
        const scalar tc = fmax($(sT_pre), fmax($(prev_sT_post), $(seT_pre)));
        """
        + izhikevich_stdp_tag_update_code + 
        """
        // Decay tag and apply STDP
        scalar newTag = $(c) * tagDecay;
        const scalar dt = $(t) - $(sT_pre);
        if (dt > 0) {
            scalar timing = exp(-dt / $(tauPlus));
            newTag += ($(aPlus) * timing);
        }
        // Write back updated tag and update time
        $(c) = newTag;
        """,
    event_threshold_condition_code="injectDopamine",

    is_pre_spike_time_required=True, 
    is_post_spike_time_required=True,
    is_pre_spike_event_time_required=True,
    
    is_prev_pre_spike_time_required=True, 
    is_prev_post_spike_time_required=True,
    is_prev_pre_spike_event_time_required=True)

# ----------------------------------------------------------------------------
# Stimuli generation
# ----------------------------------------------------------------------------
# Generate stimuli sets of neuron IDs
stim_gen_start_time =  perf_counter()
input_sets = [np.random.choice(NUM_CELLS, STIMULI_SET_SIZE, replace=False)
              for _ in range(NUM_STIMULI_SETS)]

# Lists of stimulus and reward times for use when plotting
start_stimulus_times = []
end_stimulus_times = []
start_reward_times = []
end_reward_times = []

# Create list for each neuron
neuron_stimuli_times = [[] for _ in range(NUM_CELLS)]
total_num_exc_stimuli = 0
total_num_inh_stimuli = 0

# Create zeroes numpy array to hold reward timestep bitmask
reward_timesteps = np.zeros((convert_ms_timestep(DURATION_MS) + 31) // 32, dtype=np.uint32)

# Loop while stimuli are within simulation duration
next_stimuli_timestep = np.random.randint(convert_ms_timestep(MIN_INTER_STIMULI_INTERVAL_MS),
                                          convert_ms_timestep(MAX_INTER_STIMULI_INTERVAL_MS))
while next_stimuli_timestep < convert_ms_timestep(DURATION_MS):
    # Pick a stimuli set to present at this timestep
    stimuli_set = np.random.randint(NUM_STIMULI_SETS)
    
    # Loop through neurons in stimuli set and add time to list
    for n in input_sets[stimuli_set]:
        neuron_stimuli_times[n].append(next_stimuli_timestep * TIMESTEP_MS)
    
    # Count the number of excitatory neurons in input set and add to total
    num_exc_in_input_set = np.sum(input_sets[stimuli_set] < NUM_EXCITATORY)
    total_num_exc_stimuli += num_exc_in_input_set
    total_num_inh_stimuli += (NUM_CELLS - num_exc_in_input_set)
    
    # If we should be recording at this point, add stimuli to list
    if next_stimuli_timestep < convert_ms_timestep(RECORD_TIME_MS):
        start_stimulus_times.append((next_stimuli_timestep * TIMESTEP_MS, stimuli_set))
    elif next_stimuli_timestep > convert_ms_timestep(DURATION_MS - RECORD_TIME_MS):
        end_stimulus_times.append((next_stimuli_timestep * TIMESTEP_MS, stimuli_set))
    
    # If this is the rewarded stimuli
    if stimuli_set == 0:
        # Determine time of next reward
        reward_timestep = next_stimuli_timestep + np.random.randint(convert_ms_timestep(REWARD_DELAY_MS))
        
        # If this is within simulation
        if reward_timestep < convert_ms_timestep(DURATION_MS):
            # Set bit in reward timesteps bitmask
            reward_timesteps[reward_timestep // 32] |= (1 << (reward_timestep % 32))
            
            # If we should be recording at this point, add reward to list
            if reward_timestep < convert_ms_timestep(RECORD_TIME_MS):
                start_reward_times.append(reward_timestep * TIMESTEP_MS)
            elif reward_timestep > convert_ms_timestep(DURATION_MS - RECORD_TIME_MS):
                end_reward_times.append(reward_timestep * TIMESTEP_MS)
        
    # Advance to next stimuli
    next_stimuli_timestep += np.random.randint(convert_ms_timestep(MIN_INTER_STIMULI_INTERVAL_MS),
                                               convert_ms_timestep(MAX_INTER_STIMULI_INTERVAL_MS))

# Count stimuli each neuron should emit
neuron_stimuli_counts = [len(n) for n in neuron_stimuli_times]

stim_gen_end_time =  perf_counter()
print("Stimulus generation time: %fms" % ((stim_gen_end_time - stim_gen_start_time) * 1000.0))

# ----------------------------------------------------------------------------
# Network creation
# ----------------------------------------------------------------------------
# Assert that duration is a multiple of record time
assert (convert_ms_timestep(DURATION_MS) % convert_ms_timestep(RECORD_TIME_MS)) == 0

model = genn_model.GeNNModel("float", "izhikevich_pavlovian_gpu_stim")
model.dT = TIMESTEP_MS
model._model.set_merge_postsynaptic_models(True)
model._model.set_default_narrow_sparse_ind_enabled(True)
model.timing_enabled = MEASURE_TIMING

# Excitatory model parameters
exc_params = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, 
              "tauD": TAU_D, "dStrength": DOPAMINE_STRENGTH}

# Excitatory initial state
exc_init = {"V": -65.0, "U": -13.0, "D": 0.0}

# Inhibitory model parameters
inh_params = {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0}

# Inhibitory initial state
inh_init = {"V": -65.0, "U": -13.0}

# Current source parameters
curr_source_params = {"n": 6.5, "stimMagnitude": STIMULI_CURRENT}

# Calculate start and end indices of stimuli to be injected by each current source
start_exc_stimuli, end_exc_stimuli = get_start_end_stim(neuron_stimuli_counts[:NUM_EXCITATORY])
start_inh_stimuli, end_inh_stimuli = get_start_end_stim(neuron_stimuli_counts[NUM_EXCITATORY:])

# Current source initial state
exc_curr_source_init = {"startStim": start_exc_stimuli, "endStim": end_exc_stimuli}
inh_curr_source_init = {"startStim": start_inh_stimuli, "endStim": end_inh_stimuli}

# Static inhibitory synapse initial state
inh_syn_init = {"g": INH_WEIGHT}

# STDP parameters
stdp_params = {"tauPlus": 20.0,  "tauMinus": 20.0, "tauC": 1000.0, 
               "tauD": TAU_D, "aPlus": 0.1, "aMinus": 0.15, 
               "wMin": 0.0, "wMax": MAX_EXC_WEIGHT}

# STDP initial state
stdp_init = {"g": INIT_EXC_WEIGHT, "c": 0.0}

# Create excitatory and inhibitory neuron populations
e_pop = model.add_neuron_population("E", NUM_EXCITATORY, izhikevich_dopamine_model, 
                                    exc_params, exc_init)
i_pop = model.add_neuron_population("I", NUM_INHIBITORY, "Izhikevich", 
                                    inh_params, inh_init)

# Set dopamine timestep bitmask
e_pop.set_extra_global_param("rewardTimesteps", reward_timesteps)

# Enable spike recording
e_pop.spike_recording_enabled = USE_GENN_RECORDING
i_pop.spike_recording_enabled = USE_GENN_RECORDING

# Add background current sources
e_curr_pop = model.add_current_source("ECurr", stim_noise_model, "E", 
                                      curr_source_params, exc_curr_source_init)
i_curr_pop = model.add_current_source("ICurr", stim_noise_model, "I", 
                                      curr_source_params, inh_curr_source_init)

# Set stimuli times
e_curr_pop.set_extra_global_param("stimTimes", np.hstack(neuron_stimuli_times[:NUM_EXCITATORY]))
i_curr_pop.set_extra_global_param("stimTimes", np.hstack(neuron_stimuli_times[NUM_EXCITATORY:]))

# Add synapse population
e_e_pop = model.add_synapse_population("EE", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                       "E", "E",
                                       izhikevich_stdp_model, stdp_params, stdp_init, {}, {},
                                       "DeltaCurr", {}, {},
                                       genn_model.init_connectivity("FixedProbability", {"prob": PROBABILITY_CONNECTION}))

e_i_pop = model.add_synapse_population("EI", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                             "E", "I",
                             izhikevich_stdp_model, stdp_params, stdp_init, {}, {},
                             "DeltaCurr", {}, {},
                             genn_model.init_connectivity("FixedProbability", {"prob": PROBABILITY_CONNECTION}))

model.add_synapse_population("II", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                             "I", "I",
                             "StaticPulse", {}, inh_syn_init, {}, {},
                             "DeltaCurr", {}, {},
                             genn_model.init_connectivity("FixedProbability", {"prob": PROBABILITY_CONNECTION}))

model.add_synapse_population("IE", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                             "I", "E",
                             "StaticPulse", {}, inh_syn_init, {}, {},
                             "DeltaCurr", {}, {},
                             genn_model.init_connectivity("FixedProbability", {"prob": PROBABILITY_CONNECTION}))

if BUILD_MODEL:
    print("Building model")
    model.build()

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
# Load model, allocating enough memory for recording
print("Loading model")
model.load(num_recording_timesteps=convert_ms_timestep(RECORD_TIME_MS))

print("Simulating")
# Loop through timesteps
sim_start_time =  perf_counter()
start_exc_spikes = None
start_inh_spikes = None
end_exc_spikes = None
end_inh_spikes = None
while model.t < DURATION_MS:
    # Simulation
    model.step_time()
    
    # If we've just filled the recording buffer with data we want
    if ((model.timestep == convert_ms_timestep(RECORD_TIME_MS)) or
        (model.timestep == convert_ms_timestep(DURATION_MS))):

        # Download recording data
        model.pull_recording_buffers_from_device()

        if model.timestep == convert_ms_timestep(RECORD_TIME_MS):
            start_exc_spikes = e_pop.spike_recording_data
            start_inh_spikes = i_pop.spike_recording_data
        else:
            end_exc_spikes = e_pop.spike_recording_data
            end_inh_spikes = i_pop.spike_recording_data

sim_end_time =  perf_counter()
print("Simulation time: %fms" % ((sim_end_time - sim_start_time) * 1000.0))

if MEASURE_TIMING:
    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tPresynaptic update:%f" % (1000.0 * model.presynaptic_update_time))
    print("\tPostsynaptic update:%f" % (1000.0 * model.postsynaptic_update_time))

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
# Find the earliest rewarded stimuli in start and end recording data
first_rewarded_stimuli_time_start = next(s[0] for s in start_stimulus_times if s[1] == 0)
first_rewarded_stimuli_time_end = next(s[0] for s in end_stimulus_times if s[1] == 0)

# Find the corresponding reward
corresponding_reward_time_start = next(r for r in start_reward_times if r > first_rewarded_stimuli_time_start)
corresponding_reward_time_end = next(r for r in end_reward_times if r > first_rewarded_stimuli_time_end)

padding_start = (DISPLAY_TIME_MS - (corresponding_reward_time_start - first_rewarded_stimuli_time_start)) / 2
padding_end = (DISPLAY_TIME_MS - (corresponding_reward_time_end - first_rewarded_stimuli_time_end)) / 2

# Create plot
figure, axes = plt.subplots(2)

# Plot spikes that occur in first second
axes[0].scatter(start_exc_spikes[0], start_exc_spikes[1], s=2, edgecolors="none", color="red")
axes[0].scatter(start_inh_spikes[0], start_inh_spikes[1] + NUM_EXCITATORY, s=2, edgecolors="none", color="blue")

# Plot reward times and rewarded stimuli that occur in first second
plot_reward(axes[0], start_reward_times);
plot_stimuli(axes[0], start_stimulus_times)

# Plot spikes that occur in final second
axes[1].scatter(end_exc_spikes[0], end_exc_spikes[1], s=2, edgecolors="none", color="red")
axes[1].scatter(end_inh_spikes[0], end_inh_spikes[1] + NUM_EXCITATORY, s=2, edgecolors="none", color="blue")

# Plot reward times and rewarded stimuli that occur in final second
plot_reward(axes[1], end_reward_times);
plot_stimuli(axes[1], end_stimulus_times)

# Configure axes
axes[0].set_title("Before")
axes[1].set_title("After")
axes[0].set_xlim((first_rewarded_stimuli_time_start - padding_start, corresponding_reward_time_start + padding_start))
axes[1].set_xlim((first_rewarded_stimuli_time_end - padding_end, corresponding_reward_time_end + padding_end))
axes[0].set_ylim((0, NUM_CELLS))
axes[1].set_ylim((0, NUM_CELLS))
axes[0].set_ylabel("Neuron number")
axes[1].set_ylabel("Neuron number")
axes[0].set_xlabel("Time [ms]")
axes[1].set_xlabel("Time [ms]")

# Show plot
plt.show()