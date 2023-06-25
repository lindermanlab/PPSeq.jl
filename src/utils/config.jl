
"""
Constructs SeqModel object from config dict.
"""
function construct_model(config::Dict,
                         max_time::Float64,
                         num_neurons::Int64)

    # Prior on sequence type proportions / relative frequencies.
    seq_type_proportions = SymmetricDirichlet(
        config[:seq_type_conc_param],
        config[:num_sequence_types]
    )

    # Prior on expected number of spikes induces by a sequence events.
    seq_event_amplitude = specify_gamma(
        config[:mean_event_amplitude],    # mean of gamma; α / β
        config[:var_event_amplitude]      # variance of gamma; α / β²
    )

    # Prior on relative response amplitudes per neuron to each sequence type.
    neuron_response_proportions = SymmetricDirichlet(
        config[:neuron_response_conc_param],
        num_neurons
    )

    # Prior on the response offsets and widths for each neuron.
    neuron_response_profile = NormalInvChisq(
        config[:neuron_offset_pseudo_obs],
        0.0, # prior mean
        config[:neuron_width_pseudo_obs],
        config[:neuron_width_prior],
    )

    # Prior on expected number of background spikes in a unit time interval.
    bkgd_amplitude = specify_gamma(   
        config[:mean_bkgd_spike_rate],    # mean of gamma; α / β
        config[:var_bkgd_spike_rate]      # variance of gamma; α / β²
    )

    # Prior on relative background firing rates across neurons.
    bkgd_proportions = SymmetricDirichlet(
        config[:bkgd_spikes_conc_param],
        num_neurons
    )

    model = SeqModel(
        # constants
        max_time,
        config[:max_sequence_length],

        # warp parameters
        config[:num_warp_values],
        config[:max_warp],
        config[:warp_variance],
        config[:warp_type],

        # priors
        config[:seq_event_rate],
        seq_type_proportions,
        seq_event_amplitude,
        neuron_response_proportions,
        neuron_response_profile,
        bkgd_amplitude,
        bkgd_proportions
    )

    if (:num_threads in keys(config)) && config[:num_threads] > 0
        return DistributedSeqModel(model, config[:num_threads])
    else
        return model
    end
end

"""
Takes a model and sets the neuron response parameters for the first R_sacred sequences to those from
another (previously trained) model.
"""
function sanctify_model(
    model::Union{SeqModel,DistributedSeqModel},
    sacred_neuron_responses::Matrix{Float64}, #  neuron responses from old model to be written into new one, shape = (3 x R_sacred) x N_neurones
    config::Dict
    )

    if (:num_threads in keys(config)) && config[:num_threads] > 0
        globals = model.primary_model.globals
    else
        globals = model.globals
    end

    number_sacred_sequences = size(sacred_neuron_responses)[2]÷3

    print("Size of offsets = "*string(size(globals.neuron_response_offsets)))
    print("Offset[1,1] before: "*string(globals.neuron_response_log_proportions[1,1])*"     Offset[1,-1] before: "*string(globals.neuron_response_log_proportions[1,Int(size(globals.neuron_response_log_proportions)[2])]))
    globals.neuron_response_log_proportions[:,1:number_sacred_sequences] = sacred_neuron_responses[:,1:number_sacred_sequences]
    globals.neuron_response_offsets[:,1:number_sacred_sequences] = sacred_neuron_responses[:,number_sacred_sequences+1:2*number_sacred_sequences]
    globals.neuron_response_widths[:,1:number_sacred_sequences] = sacred_neuron_responses[:,2*number_sacred_sequences+1:3*number_sacred_sequences]
    print("Offset[1,1] after: "*string(globals.neuron_response_log_proportions[1,1])*"     Offset[1,-1] after: "*string(globals.neuron_response_log_proportions[1,Int(size(globals.neuron_response_log_proportions)[2])]))

    return model
end