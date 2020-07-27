import numpy as np 
import matplotlib.pyplot as plt 

from pygenn import genn_model, genn_wrapper
from scipy.special import lambertw
from six import iteritems, itervalues
from time import perf_counter

# Using scipy to mimic the gsl_sf_lambert_Wm1 function.
def lambert_wm1(x):
    return lambertw(x, k=-1 if x < 0 else 0).real

# Computes conversion factor for synapse weight from mV to pA
# This function is specific to the leaky integrate-and-fire neuron
# model with alpha-shaped postsynaptic currents.
def convert_synapse_weight(tau_m, tau_syn, C_m):
    # compute time to maximum of V_m after spike input
    # to neuron at rest
    a = tau_m / tau_syn
    b = 1.0 / tau_syn - 1.0 / tau_m
    t_rise = 1.0 / b * (-lambert_wm1(-np.exp(-1.0 / a) / a).real - 1.0 / a)

    v_max = np.exp(1.0) / (tau_syn * C_m * b) * (
        (np.exp(-t_rise / tau_m) - np.exp(-t_rise / tau_syn)) /
        b - t_rise * np.exp(-t_rise / tau_syn))
    return 1. / v_max
    
# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# Simulation timestep [ms]
DT_MS = 0.1

# Simulation duration [ms]
DURATION_MS = 350.0

# Should kernel timing be measured?
MEASURE_TIMING = True

# Should we rebuild the model rather than loading previous version
BUILD_MODEL = False

# Should we use GeNN's built-in recording system
USE_GENN_RECORDING = True

# Total network size = SCALE * 11250 neurons
SCALE = 2.0

# Number of excitatory neurons
NUM_EXCITATORY = int(9000 * SCALE)

# Number of inhibitory neurons
NUM_INHIBITORY = int(2250 * SCALE)

# Number of incoming excitatory connections
NUM_INCOMING_EXCITATORY = int(1.0 * NUM_EXCITATORY / SCALE)

# Number of incomining inhibitory connections
NUM_INCOMING_INHIBITORY = int(1.0 * NUM_INHIBITORY / SCALE)

# Synaptic delay
DELAY_MS = 1.5

# Convert delay to timesteps
DELAY_TIMESTEPS = int(DELAY_MS / DT_MS);

# Synaptic time constant [ms]
TAU_SYN = 0.32582722403722841

# Membrane time constant [ms]
TAU_M = 10.0

# Membrane capacitance [pF]
C_M_PF = 250.0

# Threshold voltage [mV]
V_THRESH = 20.0

# Calculate synaptic weight [pA]
SYNAPTIC_WEIGHT_PA = convert_synapse_weight(TAU_M, TAU_SYN, C_M_PF) * 0.14

NU_THRESH = V_THRESH / (NUM_INCOMING_EXCITATORY * TAU_M / C_M_PF * SYNAPTIC_WEIGHT_PA * np.exp(1.) * TAU_SYN)
NU_EXT = NU_THRESH * 1.685

# ----------------------------------------------------------------------------
# Custom models
# ----------------------------------------------------------------------------
alpha_curr_model = genn_model.create_custom_postsynaptic_class(
    "alpha_curr",

    param_names=["tau"],
    var_name_types=[("x", "scalar")],
    derived_params=[
        ("ExpDecay", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[0]))()),
        ("Init", genn_model.create_dpf_class(lambda pars, dt: np.exp(1) / pars[0])())],

    decay_code=
        """
        $(x) = (DT * $(ExpDecay) * $(inSyn) * $(Init)) + ($(ExpDecay) * $(x));
        $(inSyn)*=$(ExpDecay);
        """,

    apply_input_code=
        """
        $(Isyn) += $(x);
        """)

poisson_alpha_model = genn_model.create_custom_current_source_class(
    "poisson_alpha",

    param_names=["weight", "tauSyn", "rate"],
    var_name_types=[("current", "scalar"), ("current2", "scalar")],
    derived_params=[
        ("ExpDecay", genn_model.create_dpf_class(lambda pars, dt: np.exp(-dt / pars[1]))()),
        ("ExpMinusLambda", genn_model.create_dpf_class(lambda pars, dt: np.exp(-(pars[2] / 1000.0) * dt))()),
        ("Init", genn_model.create_dpf_class(lambda pars, dt: pars[0] * (np.exp(1) / pars[1]))())],

    injection_code=
        """
        scalar p = 1.0f;
        unsigned int numPoissonSpikes = 0;
        do {
            numPoissonSpikes++;
            p *= $(gennrand_uniform);
        } while (p > $(ExpMinusLambda));
        
        $(current) += $(Init) * (scalar)(numPoissonSpikes - 1);
        $(injectCurrent, $(current2));
        $(current2) = (DT * $(ExpDecay) * $(current)) + ($(ExpDecay) * $(current2));
        $(current) *= $(ExpDecay);
        """)

stdp_model = genn_model.create_custom_weight_update_class(
    "stdp",

    param_names=["tauPlus", "tauMinus", "lambda", "alpha", "mu", "denDelay"],
    derived_params=[
        ("denDelayStep", genn_model.create_dpf_class(lambda pars, dt: np.floor(pars[5] / dt) - 1.0)())],

    var_name_types=[("g", "scalar")],
    pre_var_name_types=[("preTrace", "scalar")],
    post_var_name_types=[("postTrace", "scalar")],

    sim_code=
        """
        const scalar dt = $(t) - $(sT_post);
        if (dt > 0) {
            const scalar timing = $(postTrace) * exp(-dt / $(tauMinus));
            const scalar deltaG = -$(lambda) * $(alpha) * $(g) * timing;
            $(g) += deltaG;
        }
        $(addToInSynDelay, $(g), (unsigned int)$(denDelayStep));
        """,

    learn_post_code=
        """
        const scalar dt = $(t) - $(sT_pre);
        if (dt > 0) {
            const scalar timing = $(preTrace) * exp(-dt / $(tauPlus));
            const scalar deltaG = $(lambda) * pow($(g), $(mu)) * timing;
            $(g) += deltaG;
        }
        """,

    pre_spike_code=
        """
        scalar dt = $(t) - $(sT_pre);
        $(preTrace) = ($(preTrace) * exp(-dt / $(tauPlus))) + 1.0;
        """,

    post_spike_code=
        """
        scalar dt = $(t) - $(sT_post);
        $(postTrace) = ($(postTrace) * exp(-dt / $(tauMinus))) + 1.0;
        """,

    is_pre_spike_time_required=True,
    is_post_spike_time_required=True)

# ----------------------------------------------------------------------------
# Network creation
# ----------------------------------------------------------------------------
model = genn_model.GeNNModel("float", "hpc_benchmark")
model.dT = DT_MS
model._model.set_merge_postsynaptic_models(True)
model._model.set_default_narrow_sparse_ind_enabled(True)
model.timing_enabled = MEASURE_TIMING
model.default_var_location = genn_wrapper.VarLocation_DEVICE
model.default_sparse_connectivity_location = genn_wrapper.VarLocation_DEVICE

# LIF neuron model parameters and initial state
lif_params = {"C": C_M_PF / 1000.0, 
              "TauM": TAU_M, 
              "Vrest": 0.0, 
              "Vreset": 0.0, 
              "Vthresh": V_THRESH,
              "Ioffset": 0.0, 
              "TauRefrac": 0.5}
lif_init = {"V": genn_model.init_var("Normal", {"mean": 5.7, "sd": 7.2}), 
            "RefracTime": 0.0}

# Poisson current source model parameters and initial state
poisson_params = {"weight": SYNAPTIC_WEIGHT_PA / 1000.0,
                  "tauSyn": TAU_SYN, 
                  "rate": NU_EXT * NUM_INCOMING_EXCITATORY * 1000.0}
poisson_init = {"current": 0.0, 
                "current2": 0.0}

# Create excitatory and inhibitory neuron populations
excitatory_pop = model.add_neuron_population("Exc", NUM_EXCITATORY, "LIF", lif_params, lif_init)
inhibitory_pop = model.add_neuron_population("Inh", NUM_INHIBITORY, "LIF", lif_params, lif_init)

# Enable spike recording
excitatory_pop.spike_recording_enabled = USE_GENN_RECORDING
inhibitory_pop.spike_recording_enabled = USE_GENN_RECORDING
        
# Add background current sources
model.add_current_source("ExcPoisson", poisson_alpha_model, "Exc", poisson_params, poisson_init)
model.add_current_source("InhPoisson", poisson_alpha_model, "Inh", poisson_params, poisson_init)

# Set spike location so they can be accessed on host
excitatory_pop.pop.set_spike_location(genn_wrapper.VarLocation_HOST_DEVICE)
inhibitory_pop.pop.set_spike_location(genn_wrapper.VarLocation_HOST_DEVICE)

print("Total num neurons: %u" % (NUM_EXCITATORY + NUM_INHIBITORY))

# Alpha curr postsynaptic mode parameters and initial state
alpha_curr_params = {"tau": TAU_SYN}
alpha_curr_init = {"x": 0.0}

# Weight update model initial state
excitatory_synapse_init = {"g": SYNAPTIC_WEIGHT_PA / 1000.0 }
inhibitory_synapse_init = {"g": -5.0 * SYNAPTIC_WEIGHT_PA / 1000.0 }

# STDP model parameters
stdp_synapse_params = {"tauPlus": 15.0, 
                       "tauMinus": 15.0, 
                       "lambda": 0.1, 
                       "alpha": 0.0513,
                       "mu": 0.4, 
                       "denDelay": DELAY_MS}
stdp_synapse_pre_init = {"preTrace": 0.0}
stdp_synapse_post_init = {"postTrace": 0.0}

# Add synapse population
exc_exc_pop = model.add_synapse_population("ExcExc", "SPARSE_INDIVIDUALG", genn_wrapper.NO_DELAY,
    "Exc", "Exc",
    stdp_model, stdp_synapse_params, excitatory_synapse_init, stdp_synapse_pre_init, stdp_synapse_post_init,
    alpha_curr_model, alpha_curr_params, alpha_curr_init,
    genn_model.init_connectivity("FixedNumberPreWithReplacement", {"colLength": NUM_INCOMING_EXCITATORY}))

# Configure dendritic delay and matching back propagation delay
exc_exc_pop.pop.set_max_dendritic_delay_timesteps(DELAY_TIMESTEPS)
exc_exc_pop.pop.set_back_prop_delay_steps(DELAY_TIMESTEPS - 4);

model.add_synapse_population("ExcInh", "SPARSE_GLOBALG_INDIVIDUAL_PSM", DELAY_TIMESTEPS,
                             "Exc", "Inh",
                             "StaticPulse", {}, excitatory_synapse_init, {}, {},
                             alpha_curr_model, alpha_curr_params, alpha_curr_init,
                             genn_model.init_connectivity("FixedNumberPreWithReplacement", {"colLength": NUM_INCOMING_EXCITATORY}))

model.add_synapse_population("InhInh", "SPARSE_GLOBALG_INDIVIDUAL_PSM", DELAY_TIMESTEPS,
                             "Inh", "Inh",
                             "StaticPulse", {}, inhibitory_synapse_init, {}, {},
                             alpha_curr_model, alpha_curr_params, alpha_curr_init,
                             genn_model.init_connectivity("FixedNumberPreWithReplacement", {"colLength": NUM_INCOMING_INHIBITORY}))

model.add_synapse_population("InhExc", "SPARSE_GLOBALG_INDIVIDUAL_PSM", DELAY_TIMESTEPS,
                             "Inh", "Exc",
                             "StaticPulse", {}, inhibitory_synapse_init, {}, {},
                             alpha_curr_model, alpha_curr_params, alpha_curr_init,
                             genn_model.init_connectivity("FixedNumberPreWithReplacement", {"colLength": NUM_INCOMING_INHIBITORY}))

if BUILD_MODEL:
    print("Building model")
    model.build()
    
print("Loading model")
duration_timesteps = int(round(DURATION_MS / DT_MS))
model.load(num_recording_timesteps=duration_timesteps)

print("Simulating")
sim_start_time = perf_counter()

while model.t < DURATION_MS:
    model.step_time()

sim_end_time =  perf_counter()

# Download recording data
if USE_GENN_RECORDING:
    model.pull_recording_buffers_from_device()

print("Timing:")
print("\tSimulation:%f" % ((sim_end_time - sim_start_time) * 1000.0))

if MEASURE_TIMING:
    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tPresynaptic update:%f" % (1000.0 * model.presynaptic_update_time))
    print("\tPostsynaptic update:%f" % (1000.0 * model.postsynaptic_update_time))
