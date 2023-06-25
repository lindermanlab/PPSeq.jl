"""
Annealed Gibbs sampling, to remove small amplitude events.
"""
function annealed_gibbs!(
        model::Union{SeqModel,DistributedSeqModel},
        spikes::Vector{Spike},
        initial_assignments::Vector{Int64},
        config::Dict;
        verbose::Bool=false
    )

    num_anneals = config[:num_anneals]
    max_temperature = config[:max_temperature]

    # Initialize storage.
    assignment_hist = zeros(Int64, length(spikes), 0)
    log_p_hist = Float64[]
    latent_event_hist = Vector{EventSummaryInfo}[]
    globals_hist = SeqGlobals[]

    # Return early if no samples.
    if num_anneals == 0
        return (
            initial_assignments,
            assignment_hist,
            log_p_hist,
            latent_event_hist,
            globals_hist
        )
    end

    # Final amplitude for anneal.
    target_mean = mean(priors(model).seq_event_amplitude)
    target_var = var(priors(model).seq_event_amplitude)

    # Begin annealing.
    temperatures = exp10.(range(log10(max_temperature), 0, length=num_anneals))
    assignments = initial_assignments

    for temp in temperatures
        
        # Print progress.
        verbose && println("TEMP:  ", temp)
        flush(stdout)

        # Anneal prior on sequence amplitude.
        prior = priors(model)
        set_priors!(
            model, 
            SeqPriors(
                prior.seq_event_rate,
                prior.seq_type_proportions,
                specify_gamma(target_mean, target_var * temp), # Anneal prior.
                prior.neuron_response_proportions,
                prior.neuron_response_profile,
                prior.bkgd_amplitude,
                prior.bkgd_proportions,
                prior.warp_values,
                prior.warp_log_proportions
            )
        )

        # Recompute probability of introducing a new cluster.
        #  ==> TODO: maybe set_priors! should do this automatically?
        max_time = get_max_time(model)
        prior = priors(model)
        α = prior.seq_event_amplitude.α
        β = prior.seq_event_amplitude.β
        λ = prior.seq_event_rate
        set_new_cluster_log_prob!(
            model,
            log(α) + log(λ) + log(max_time) + α * (log(β) - log(1 + β))
        )
        # TODO fix this, can't use primary model in not distributed code
        set_bkgd_log_prob!(
            model,
            (log(get_bkgd_amplitude(model))
            + log(max_time)
            + log(1 + β))
        )

        # Draw gibbs samples.
        (
            assignments,
            _assgn,
            _logp,
            _latents,
            _globals
        ) =
        gibbs_sample!(
            model,
            spikes,
            assignments,
            config[:samples_per_anneal],
            config[:split_merge_moves_during_anneal],
            config[:save_every_during_anneal],
            config;
            verbose=verbose
        )

        # Save samples.
        assignment_hist = cat(assignment_hist, _assgn, dims=2)
        append!(log_p_hist, _logp)
        append!(latent_event_hist, _latents)
        append!(globals_hist, _globals)

    end

    return (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    )

end
