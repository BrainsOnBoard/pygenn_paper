import numpy as np
import matplotlib.pyplot as plt 

from pygenn import genn_model, genn_wrapper
from pygenn.genn_wrapper.Models import VarAccess_READ_ONLY
from six import iteritems

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
# Helper functions
# ----------------------------------------------------------------------------
def get_params(size_scale_factor=1, build_model=True, 
               measure_timing=False, use_genn_recording=True):
    weight_scale_factor = 1.0 / size_scale_factor

    # Build params dictionary
    params = {
        "timestep_ms": 1.0,

        # Should we rebuild model
        "build_model": build_model,

        # Generate code for kernel timing
        "measure_timing": measure_timing,

        # Use GeNN's built in spike recording system
        "use_genn_recording": use_genn_recording,

        # Use zero copy for recording
        "use_zero_copy": False,
        
        # Simulation duration
        "duration_ms": 60.0 * 60.0 * 1000.0,

        # How much of start and end of simulation to record
        # **NOTE** we want to see at least one rewarded stimuli in each recording window
        "record_time_ms": 50.0 * 1000.0,

        # How often should outgoing weights from each synapse be recorded
        "weight_record_interval_ms": 10.0 * 1000.0,

        # STDP params
        "tau_d": 200.0,

        # Scaled number of cells
        "num_excitatory": 800 * int(size_scale_factor),
        "num_inhibitory": 200 * int(size_scale_factor),

        # Weights
        "inh_weight": -1.0 * weight_scale_factor,
        "init_exc_weight": 1.0 * weight_scale_factor,
        "max_exc_weight": 4.0 * weight_scale_factor,
        "dopamine_strength": 0.5 * weight_scale_factor,

        # Connection probability
        "probability_connection": 0.1,

        # Input sets
        "num_stimuli_sets": 100,
        "stimuli_set_size": 50,
        "stimuli_current": 40.0,

        # Regime
        "min_inter_stimuli_interval_ms": 100.0,
        "max_inter_stimuli_interval_ms": 300.0,

        # Reward
        "max_reward_delay_ms": 1000.0}
    
    # Loop through parameters
    dt = params["timestep_ms"]
    timestep_params = {}
    for n, v in iteritems(params):
        # If parameter isn't timestep and it ends with millisecond suffix,
        # Add new version of parameter in timesteps to temporary dictionary
        if n != "timestep_ms" and n.endswith("_ms"):
            timestep_params[n[:-2] + "timestep"] = int(round(v / dt))
    
    # Update parameters dictionary with new parameters and return
    params.update(timestep_params)
    return params

def plot_reward(axis, times):
    for t in times:
        axis.annotate("reward",
                      xy=(t, 0), xycoords="data",
                      xytext=(0, -15.0), textcoords="offset points",
                      arrowprops=dict(facecolor="black", headlength=6.0),
                      annotation_clip=True, ha="center", va="top")

def plot_stimuli(axis, times, num_cells):
    for t, i in times:
        colour = "green" if i == 0 else "black"
        axis.annotate("S%u" % i,
                      xy=(t, num_cells), xycoords="data",
                      xytext=(0, 15.0), textcoords="offset points",
                      arrowprops=dict(facecolor=colour, edgecolor=colour, headlength=6.0),
                      annotation_clip=True, ha="center", va="bottom", color=colour)

def convert_spikes(spike_list, timesteps):
    # Determine how many spikes were emitted in each timestep
    spikes_per_timestep = [len(s) for s in spike_list]
    assert len(timesteps) == len(spikes_per_timestep)

    # Repeat timesteps correct number of times to match number of spikes
    spike_times = np.repeat(timesteps, spikes_per_timestep)
    spike_ids = np.hstack(spike_list)
    
    return spike_times, spike_ids
    
def build_model(name, params, reward_timesteps):
    model = genn_model.GeNNModel("float", name)
    model.dT = params["timestep_ms"]
    model._model.set_merge_postsynaptic_models(True)
    model._model.set_default_narrow_sparse_ind_enabled(True)

    if "seed" in params:
        model._model.set_seed(params["seed"])
    model.timing_enabled = params["measure_timing"]

    # Excitatory model parameters
    exc_params = {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0, 
                  "tauD": params["tau_d"], "dStrength": params["dopamine_strength"]}

    # Excitatory initial state
    exc_init = {"V": -65.0, "U": -13.0, "D": 0.0}

    # Inhibitory model parameters
    inh_params = {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0}

    # Inhibitory initial state
    inh_init = {"V": -65.0, "U": -13.0}

    # Static inhibitory synapse initial state
    inh_syn_init = {"g": params["inh_weight"]}

    # STDP parameters
    stdp_params = {"tauPlus": 20.0,  "tauMinus": 20.0, "tauC": 1000.0, 
                   "tauD": params["tau_d"], "aPlus": 0.1, "aMinus": 0.15, 
                   "wMin": 0.0, "wMax": params["max_exc_weight"]}

    # STDP initial state
    stdp_init = {"g": params["init_exc_weight"], "c": 0.0}
    
    # Fixed probability connector parameters
    fixed_prob_params = {"prob": params["probability_connection"]}
    
    # Create excitatory and inhibitory neuron populations
    e_pop = model.add_neuron_population("E", params["num_excitatory"], izhikevich_dopamine_model, 
                                        exc_params, exc_init)
    i_pop = model.add_neuron_population("I", params["num_inhibitory"], "Izhikevich", 
                                        inh_params, inh_init)
    
    # Turn on zero-copy for spikes if required
    if params["use_zero_copy"]:
        e_pop.pop.set_spike_location(genn_wrapper.VarLocation_HOST_DEVICE_ZERO_COPY)
        i_pop.pop.set_spike_location(genn_wrapper.VarLocation_HOST_DEVICE_ZERO_COPY)
    
    # Set dopamine timestep bitmask
    e_pop.set_extra_global_param("rewardTimesteps", reward_timesteps)

    # Enable spike recording
    e_pop.spike_recording_enabled = params["use_genn_recording"]
    i_pop.spike_recording_enabled = params["use_genn_recording"]

    # Add synapse population
    e_e_pop = model.add_synapse_population("EE", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                           "E", "E",
                                           izhikevich_stdp_model, stdp_params, stdp_init, {}, {},
                                           "DeltaCurr", {}, {},
                                           genn_model.init_connectivity("FixedProbability", fixed_prob_params))

    e_i_pop = model.add_synapse_population("EI", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
                                 "E", "I",
                                 izhikevich_stdp_model, stdp_params, stdp_init, {}, {},
                                 "DeltaCurr", {}, {},
                                 genn_model.init_connectivity("FixedProbability", fixed_prob_params))

    model.add_synapse_population("II", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                 "I", "I",
                                 "StaticPulse", {}, inh_syn_init, {}, {},
                                 "DeltaCurr", {}, {},
                                 genn_model.init_connectivity("FixedProbability", fixed_prob_params))

    model.add_synapse_population("IE", "SPARSE_GLOBALG", genn_wrapper.NO_DELAY,
                                 "I", "E",
                                 "StaticPulse", {}, inh_syn_init, {}, {},
                                 "DeltaCurr", {}, {},
                                 genn_model.init_connectivity("FixedProbability", fixed_prob_params))
    
    # Return model and populations
    return model, e_pop, i_pop, e_e_pop, e_i_pop

def plot(start_exc_spikes, start_inh_spikes, end_exc_spikes, end_inh_spikes,
         start_stimulus_times, start_reward_times, 
         end_stimulus_times, end_reward_times, 
         display_time_ms, params):
    # Find the earliest rewarded stimuli in start and end recording data
    first_rewarded_stimuli_time_start = next(s[0] for s in start_stimulus_times if s[1] == 0)
    first_rewarded_stimuli_time_end = next(s[0] for s in end_stimulus_times if s[1] == 0)

    # Find the corresponding reward
    corresponding_reward_time_start = next(r for r in start_reward_times if r > first_rewarded_stimuli_time_start)
    corresponding_reward_time_end = next(r for r in end_reward_times if r > first_rewarded_stimuli_time_end)

    padding_start = (display_time_ms - (corresponding_reward_time_start - first_rewarded_stimuli_time_start)) / 2
    padding_end = (display_time_ms - (corresponding_reward_time_end - first_rewarded_stimuli_time_end)) / 2

    # Create plot
    figure, axes = plt.subplots(2)

    # Plot spikes that occur in first second
    axes[0].scatter(start_exc_spikes[0], start_exc_spikes[1], s=2, edgecolors="none", color="red")
    axes[0].scatter(start_inh_spikes[0], start_inh_spikes[1] + params["num_excitatory"], s=2, edgecolors="none", color="blue")

    # Plot reward times and rewarded stimuli that occur in first second
    num_cells = params["num_excitatory"] + params["num_inhibitory"]
    plot_reward(axes[0], start_reward_times);
    plot_stimuli(axes[0], start_stimulus_times, num_cells)

    # Plot spikes that occur in final second
    axes[1].scatter(end_exc_spikes[0], end_exc_spikes[1], s=2, edgecolors="none", color="red")
    axes[1].scatter(end_inh_spikes[0], end_inh_spikes[1] + params["num_excitatory"], s=2, edgecolors="none", color="blue")

    # Plot reward times and rewarded stimuli that occur in final second
    plot_reward(axes[1], end_reward_times);
    plot_stimuli(axes[1], end_stimulus_times, num_cells)

    # Configure axes
    axes[0].set_title("Before")
    axes[1].set_title("After")
    axes[0].set_xlim((first_rewarded_stimuli_time_start - padding_start, corresponding_reward_time_start + padding_start))
    axes[1].set_xlim((first_rewarded_stimuli_time_end - padding_end, corresponding_reward_time_end + padding_end))
    axes[0].set_ylim((0, num_cells))
    axes[1].set_ylim((0, num_cells))
    axes[0].set_ylabel("Neuron number")
    axes[1].set_ylabel("Neuron number")
    axes[0].set_xlabel("Time [ms]")
    axes[1].set_xlabel("Time [ms]")