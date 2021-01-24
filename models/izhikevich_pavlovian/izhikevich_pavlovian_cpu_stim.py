import numpy as np
import matplotlib.pyplot as plt 

from pygenn import genn_model, genn_wrapper
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from time import perf_counter

from common import (izhikevich_dopamine_model, izhikevich_stdp_model, 
                    build_model, get_params, plot, convert_spikes)

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
stim_noise_model = genn_model.create_custom_current_source_class(
    "stim_noise",

    param_names=["n"],
    var_name_types=[("iExt", "scalar", VarAccess_READ_ONLY)],
    injection_code=
        """
        $(injectCurrent, $(iExt) + ($(gennrand_uniform) * $(n) * 2.0) - $(n));
        """)

# ----------------------------------------------------------------------------
# Stimuli generation
# ----------------------------------------------------------------------------
# Get standard model parameters
params = get_params(build_model=True, measure_timing=False, use_genn_recording=True)

# Generate stimuli sets of neuron IDs
num_cells = params["num_excitatory"] + params["num_inhibitory"]
stim_gen_start_time =  perf_counter()
input_sets = [np.random.choice(num_cells, params["stimuli_set_size"], replace=False)
              for _ in range(params["num_stimuli_sets"])]
input_sets_exc = [i[i < params["num_excitatory"]] for i in input_sets]
input_sets_inh = [i[i >= params["num_excitatory"]] - params["num_excitatory"] 
                  for i in input_sets]

# Lists of stimulus and reward times for use when plotting
stimulus_timesteps = []
start_stimulus_times = []
end_stimulus_times = []
start_reward_times = []
end_reward_times = []

# Create zeroes numpy array to hold reward timestep bitmask
reward_timesteps = np.zeros((params["duration_timestep"] + 31) // 32, dtype=np.uint32)

# Loop while stimuli are within simulation duration
next_stimuli_timestep = np.random.randint(params["min_inter_stimuli_interval_timestep"],
                                          params["max_inter_stimuli_interval_timestep"])
while next_stimuli_timestep < params["duration_timestep"]:
    # Pick a stimuli set to present at this timestep
    stimuli_set = np.random.randint(params["num_stimuli_sets"])
    
    # Add stimuli to list
    stimulus_timesteps.append((next_stimuli_timestep, stimuli_set))
    
    # If we should be recording at this point, add stimuli to list
    if next_stimuli_timestep < params["record_time_timestep"]:
        start_stimulus_times.append((next_stimuli_timestep * params["timestep_ms"], stimuli_set))
    elif next_stimuli_timestep > (params["duration_timestep"] - params["record_time_timestep"]):
        end_stimulus_times.append((next_stimuli_timestep * params["timestep_ms"], stimuli_set))
    
    # If this is the rewarded stimuli
    if stimuli_set == 0:
        # Determine time of next reward
        reward_timestep = next_stimuli_timestep + np.random.randint(params["max_reward_delay_timestep"])
        
        # If this is within simulation
        if reward_timestep < params["duration_timestep"]:
            # Set bit in reward timesteps bitmask
            reward_timesteps[reward_timestep // 32] |= (1 << (reward_timestep % 32))
            
            # If we should be recording at this point, add reward to list
            if reward_timestep < params["record_time_timestep"]:
                start_reward_times.append(reward_timestep * params["timestep_ms"])
            elif reward_timestep > (params["duration_timestep"] - params["record_time_timestep"]):
                end_reward_times.append(reward_timestep * params["timestep_ms"])
        
    # Advance to next stimuli
    next_stimuli_timestep += np.random.randint(params["min_inter_stimuli_interval_timestep"],
                                               params["max_inter_stimuli_interval_timestep"])

stim_gen_end_time =  perf_counter()
print("Stimulus generation time: %fms" % ((stim_gen_end_time - stim_gen_start_time) * 1000.0))

# ----------------------------------------------------------------------------
# Network creation
# ----------------------------------------------------------------------------
# Assert that duration is a multiple of record time
assert (params["duration_timestep"] % params["record_time_timestep"]) == 0

# Build base model
model, e_pop, i_pop, e_e_pop, e_i_pop = build_model("izhikevich_pavlovian_gpu_stim", 
                                                    params, reward_timesteps)

# Current source parameters
curr_source_params = {"n": 6.5}

# Current source initial state
curr_source_init = {"iExt": 0.0}

# Add background current sources
e_curr_pop = model.add_current_source("ECurr", stim_noise_model, "E", 
                                      curr_source_params, curr_source_init)
i_curr_pop = model.add_current_source("ICurr", stim_noise_model, "I", 
                                      curr_source_params, curr_source_init)
if params["use_zero_copy"]:
        e_curr_pop.pop.set_var_location("iExt", genn_wrapper.VarLocation_HOST_DEVICE_ZERO_COPY)
        i_curr_pop.pop.set_var_location("iExt", genn_wrapper.VarLocation_HOST_DEVICE_ZERO_COPY)
    
if params["build_model"]:
    print("Building model")
    model.build()

# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
# Load model, allocating enough memory for recording
print("Loading model")
model.load(num_recording_timesteps=params["record_time_timestep"])

print("Simulating")

# Get memory views to access currents
e_curr_ext_view = e_curr_pop.vars["iExt"].view
i_curr_ext_view = i_curr_pop.vars["iExt"].view

# Loop through timesteps
sim_start_time =  perf_counter()
start_exc_spikes = None if params["use_genn_recording"] else []
start_inh_spikes = None if params["use_genn_recording"] else []
end_exc_spikes = None if params["use_genn_recording"] else []
end_inh_spikes = None if params["use_genn_recording"] else []
while model.t < params["duration_ms"]:
    # If we should inject stimuli during this timestep
    should_stimulate = (len(stimulus_timesteps) > 0 and 
                        model.timestep == stimulus_timesteps[0][0])
    if should_stimulate:
        # Get input set associated with stimuli
        stimuli_input_set_exc = input_sets_exc[stimulus_timesteps[0][1]]
        stimuli_input_set_inh = input_sets_inh[stimulus_timesteps[0][1]]
        
        # Set input current to neurons in set to stimuli
        e_curr_ext_view[stimuli_input_set_exc] = params["stimuli_current"]
        i_curr_ext_view[stimuli_input_set_inh] = params["stimuli_current"]

        # Upload
        e_curr_pop.push_var_to_device("iExt")
        i_curr_pop.push_var_to_device("iExt")

    # Simulation
    model.step_time()
    
    # If we injected stimuli during this timestep
    if should_stimulate:
        # Remove this stimulus from array
        stimulus_timesteps.pop(0)
        
        # Zero host currents
        e_curr_ext_view[:] = 0.0
        i_curr_ext_view[:] = 0.0
        
        # Upload
        e_curr_pop.push_var_to_device("iExt")
        i_curr_pop.push_var_to_device("iExt")
        
    if params["use_genn_recording"]:
        # If we've just finished simulating the initial recording interval
        if model.timestep == params["record_time_timestep"]:
            # Download recording data
            model.pull_recording_buffers_from_device()
            
            start_exc_spikes = e_pop.spike_recording_data
            start_inh_spikes = i_pop.spike_recording_data
        # Otherwise, if we've finished entire simulation
        elif model.timestep == params["duration_timestep"]:
            # Download recording data
            model.pull_recording_buffers_from_device()
            
            end_exc_spikes = e_pop.spike_recording_data
            end_inh_spikes = i_pop.spike_recording_data
    else:
        if model.timestep <= params["record_time_timestep"]:
            e_pop.pull_current_spikes_from_device()
            i_pop.pull_current_spikes_from_device()
            start_exc_spikes.append(np.copy(e_pop.current_spikes))
            start_inh_spikes.append(np.copy(i_pop.current_spikes))
        elif model.timestep > (params["duration_timestep"] - params["record_time_timestep"]):
            e_pop.pull_current_spikes_from_device()
            i_pop.pull_current_spikes_from_device()
            end_exc_spikes.append(np.copy(e_pop.current_spikes))
            end_inh_spikes.append(np.copy(i_pop.current_spikes))

sim_end_time =  perf_counter()
print("Simulation time: %fms" % ((sim_end_time - sim_start_time) * 1000.0))

if not params["use_genn_recording"]:
    start_timesteps = np.arange(0.0, params["record_time_ms"], params["timestep_ms"])
    end_timesteps = np.arange(params["duration_ms"] - params["record_time_ms"], params["duration_ms"], params["timestep_ms"])
    
    start_exc_spikes = convert_spikes(start_exc_spikes, start_timesteps)
    start_inh_spikes = convert_spikes(start_inh_spikes, start_timesteps)
    end_exc_spikes = convert_spikes(end_exc_spikes, end_timesteps)
    end_inh_spikes = convert_spikes(end_inh_spikes, end_timesteps)

if params["measure_timing"]:
    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tPresynaptic update:%f" % (1000.0 * model.presynaptic_update_time))
    print("\tPostsynaptic update:%f" % (1000.0 * model.postsynaptic_update_time))

# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
plot(start_exc_spikes,start_inh_spikes, end_exc_spikes, end_inh_spikes,
     start_stimulus_times, start_reward_times, 
     end_stimulus_times, end_reward_times,
     2000.0, params)

# Show plot
plt.show()