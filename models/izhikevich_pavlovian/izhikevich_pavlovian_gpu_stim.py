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

# Generate code for kernel timing
MEASURE_TIMING = False

# Use GeNN's built in spike recording system
USE_GENN_RECORDING = True

# Simulation duration
DURATION_MS = 60.0 * 60.0 * 1000.0

# How much of start and end of simulation to record
# **NOTE** we want to see at least one rewarded stimuli in each recording window
RECORD_TIME_MS = 50.0 * 1000.0

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
izhikevich_dopamine_model = genn_model.create_custom_neuron_class(
    "izhikevich_dopamine",

    param_names=["a", "b", "c", "d", "tauD", "dStrength"],
    var_name_types=[("V", "scalar"), ("U", "scalar"), ("D", "scalar")],
    extra_global_params=[("dTimes", "uint32_t*")],
    sim_code=
        """
        $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT; //at two times for numerical stability
        $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT;
        $(U)+=$(a)*($(b)*$(V)-$(U))*DT;
        const unsigned int timestep = (unsigned int)($(t) / DT);
        const bool injectDopamine = (($(dTimes)[timestep / 32] & (1 << (timestep % 32))) != 0);
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
# Network creation
# ----------------------------------------------------------------------------
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

# Current source initial state
curr_source_init = {"startStim": None, "endStim": None}

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

# Enable spike recording
e_pop.spike_recording_enabled = USE_GENN_RECORDING
i_pop.spike_recording_enabled = USE_GENN_RECORDING

# Add background current sources
e_curr_pop = model.add_current_source("ECurr", stim_noise_model, "E", 
                                      curr_source_params, curr_source_init)
i_curr_pop = model.add_current_source("ICurr", stim_noise_model, "I", 
                                      curr_source_params, curr_source_init)
    
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

model.build()
