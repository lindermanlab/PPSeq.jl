
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

    SeqModel(
        # constants
        max_time,
        config[:max_sequence_length],

        # warp parameters
        config[:num_warp_values],
        config[:max_warp],
        config[:warp_variance],

        # priors
        config[:seq_event_rate],
        seq_type_proportions,
        seq_event_amplitude,
        neuron_response_proportions,
        neuron_response_profile,
        bkgd_amplitude,
        bkgd_proportions
    )
end
