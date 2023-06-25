
"""
Run gibbs sampler.
"""
function gibbs_sample!(
        model::SeqModel,
        spikes::Vector{Spike},
        initial_assignments::Vector{Int64},
        num_samples::Int64,
        extra_split_merge_moves::Int64,
        save_every::Int64,
        config::Dict;
        verbose::Bool=false
    )

    split_merge_window = config[:split_merge_window]

    # Initialize spike assignments.
    assignments = initial_assignments
    recompute!(model, spikes, assignments)

    # ======== THINGS TO SAVE ======== #

    # number of samples to save.
    n_saved_samples = Int(round(num_samples / save_every))
    
    # log likelihoods.
    log_p_hist = zeros(n_saved_samples)

    # parent assignments.
    assignment_hist = zeros(
        Int64, length(spikes), n_saved_samples
    )

    # latent event assignment ids, times, types, amplitudes.
    latent_event_hist = Vector{EventSummaryInfo}[]

    # global variables (offsets, response amplitudes, etc.)
    globals_hist = SeqGlobals[]

    # Order to iterate over spikes.
    spike_order = collect(1:length(spikes))

    # ======== MAIN LOOP ======== #

    for s = 1:num_samples

        # Update spike assignments in random order.
        Random.shuffle!(spike_order)
        for i = spike_order
            remove_datapoint!(model, spikes[i], assignments[i])
            assignments[i] = gibbs_add_datapoint!(model, spikes[i])
        end

        # Add extra split merge moves.
        split_merge_sample!(
            model,
            spikes,
            extra_split_merge_moves,
            assignments,
            split_merge_window,
        )

        # Update latent events.
        gibbs_update_latents!(model, config)

        # Update globals
        gibbs_update_globals!(model, spikes, assignments, config)

        # Store results
        if (s % save_every) == 0

            # Index into hist vectors.
            j = Int(s / save_every)

            # Note that the likelihood should be computed after
            # gibbs_update_latents! and before spike parent assignments
            # are recomputed. Otherwise, newly added latent events are
            # not initialized.
            log_p_hist[j] = log_like(model, spikes)

            # Save assignments.
            assignment_hist[:, j] .= assignments

            # Save latent event information.
            push!(
                latent_event_hist,
                event_list_summary(model)
            )

            # Save global variables.
            push!(globals_hist, deepcopy(model.globals))

            # Display progress.
            verbose && print(s, "-")
            flush(stdout)
        end
    end

    # Finish progress bar.
    verbose && (n_saved_samples > 0) && println("Done")
    flush(stdout)

    return (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    )
end


"""
Adds spikes `s` to an existing sequence event, to a new sequence event,
or to the background process.

For each sequence event k = 1 ... K, we compute

    prob[k] = p(x_i | z_i = k, x_{neq i}) * (N_k + alpha)

The probability of forming a new cluster is

    prob[K + 1] propto p(x_i | z_i = K + 1) * alpha * (V(K + 1) / V(K))

where p(x_i | z_i = K + 1) is the marginal probability of a singleton cluster.
See section 6 of Miller & Harrison (2018).

The probability of the background is

    prob[K + 2] propto p(x_i | bkgd) * lambda0 * (1 + beta)

where lambda0 = m.bkgd_rate and (alpha, beta) are the shape and 
rate parameters of the gamma prior on sequence event amplitudes.
"""
function gibbs_add_datapoint!(model::SeqModel, x::Spike)

    # Create log-probability vector to sample assignments.
    #
    #  - We need to sample K + 2 possibilities. There are K existing clusters
    #    we could assign to. We could also form a new cluster (index K + 1),
    #    or assign the spike to the background (index K + 2).

    # Shape and rate parameters of gamma prior on latent event amplitude.
    α = model.priors.seq_event_amplitude.α
    β = model.priors.seq_event_amplitude.β

    K = num_sequence_events(model)
    
    # Grab vector without allocating new memory.
    log_probs = resize!(model._K_buffer, K + 2)

    # Iterate over model events, indexed by k = {1, 2, ..., K}.
    for (k, event) in enumerate(model.sequence_events)

        # Check if event is too far away to be considered. If sampled_type < 0,
        # then the event timestamp hasn't been sampled yet, so we can't give
        # up yet. Be aware that this previously led to a subtle bug, which was
        # very painful to fix.
        too_far = abs(x.timestamp - event.sampled_timestamp) > model.max_sequence_length
        if too_far && (event.sampled_type > 0)
            log_probs[k] = -Inf

        # Compute probability of adding spike to cluster k.
        else
            log_probs[k] = (
                log(event.spike_count + α)
                + log_posterior_predictive(event, x, model)
            )
        end
    end

    # New cluster probability.
    log_probs[K + 1] = (
        model.new_cluster_log_prob
        + log_posterior_predictive(x, model)
    )

    # Background probability
    log_probs[K + 2] = (
        model.bkgd_log_prob
        + model.globals.bkgd_log_proportions[x.neuron]
        - log(model.max_time)
    )

    # Sample new assignment for spike x.
    z = sample_logprobs!(log_probs)

    # New sample corresponds to background, do nothing.
    if z == (K + 2)
        return -1

    # New sample corresponds to new sequence event / cluster.
    elseif z == (K + 1)
        return add_event!(model, x)  # returns new assignment

    # Otherwise, add datapoint to existing sequence event. Note
    # that z is an integer in [1:K], while assignment indices
    # can be larger and non-contiguous.
    else
        k = model.sequence_events.indices[z]  # look up assignment index.
        return add_datapoint!(model, x, k)
    end

end


function gibbs_update_globals!(
        model::SeqModel,
        spikes::Vector{Spike},
        assignments::AbstractVector{Int64},
        config::Dict
    )

    K = num_sequence_events(model)
    N = num_neurons(model)
    R = num_sequence_types(model)

    priors = model.priors
    globals = model.globals

    # === RESAMPLE BACKGROUND SPIKE PARAMETERS === #

    num_bkgd_spikes = 0
    bkgd_counts = zeros(N)

    for (i, x) in enumerate(spikes)
        if assignments[i] < 0
            num_bkgd_spikes += 1
            bkgd_counts[x.neuron] += 1
        end
    end

    # Sample proportions of background spikes across neurons
    # as a probability vector then map to log space for
    # future computations.
    rand!(
        posterior(
            bkgd_counts, priors.bkgd_proportions
        ),
        globals.bkgd_log_proportions
    ) # multinomial - symmetric dirichlet conjugate pair.
    map!(
        log,
        globals.bkgd_log_proportions,
        globals.bkgd_log_proportions
    )

    # Sample the rate of background spikes. Here, we need to adjust
    # for the length of the time interval, max_time. The scaling
    # property of the gamma distribution implies that:
    #
    #   bkgd_amp * max_time ~ RateGamma(α_bkgd, β_bkgd / max_time)
    #
    # If we observe num_bkgd, then the posterior (by Poisson-gamma
    # conjugacy) is:
    #
    #   bkgd_amp * T | num_bkgd_spikes ~ RateGamma(num_bkgd + α_bkgd, 1 + β_bkgd / T)
    #
    # Now apply the gamma scaling property again, dividing by T this
    # time, so we get:
    #
    #   bkgd_amp | num_bkgd_spikes ~ RateGamma(num_bkgd + α_bkgd, T + β_bkgd)
    #
    globals.bkgd_amplitude = rand(
        RateGamma(
            num_bkgd_spikes + priors.bkgd_amplitude.α,
            priors.bkgd_amplitude.β + model.max_time
        )
    )


    # === RESAMPLE SEQUENCE TYPE PROPORTIONS === #

    seq_type_counts = zeros(R)
    for event in model.sequence_events
        seq_type_counts[event.sampled_type] += 1
    end

    rand!(
        posterior(
            seq_type_counts, priors.seq_type_proportions
        ),
        model.globals.seq_type_log_proportions
    )  # multinomial - symmetric dirichlet conjugate pair.
    map!(
        log,
        globals.seq_type_log_proportions,
        globals.seq_type_log_proportions
    )

    # === RESAMPLE NEURON RESPONSE PROFILES === #

    # TODO -- preallocate arrays for this? Or is it not worth it?
    spk_count = zeros(Int64, N, R)
    spk_fm = zeros(N, R)
    spk_sm = zeros(N, R)

    for (i, x) = enumerate(spikes)

        # spike assignment to latent event.
        k = assignments[i]
        
        # skip if spike is assigned to background.
        (k < 0) && continue

        event = model.sequence_events[k]
        n = x.neuron
        r = event.sampled_type
        w = event.sampled_warp
        offset = (x.timestamp - event.sampled_timestamp) / model.priors.warp_values[w]

        # compute sufficient statistics
        spk_count[n, r] += 1
        spk_fm[n, r] += offset
        spk_sm[n, r] += offset * offset

    end

    # config dict has an integer parameter X = config[:sacred sequences].
    # The first X sequences should NOT have their parameters changed (i.e. neuron width, offset, and log proportions)

    for r = 1+config[:sacred_sequences]:R

        rand!(
            posterior(
                view(spk_count, :, r),
                priors.neuron_response_proportions
            ),
            view(globals.neuron_response_log_proportions, :, r)
        ) # multinomial - dirichlet conjugate pair.

        for n = 1:N

            (
                globals.neuron_response_offsets[n, r],
                globals.neuron_response_widths[n, r]
            ) =
            rand(
                posterior(
                    spk_count[n, r],
                    spk_fm[n, r],
                    spk_sm[n, r],
                    priors.neuron_response_profile
                )
            ) # normal - norm-inv-chi-squared conjugate pair.

        end

        # Now take logarithm of the log_proportions
        globals.neuron_response_log_proportions[:,r] = log.(globals.neuron_response_log_proportions[:,r])
    end

    # === RECOMPUTE NEW CLUSTER PROBABILITIES === #

    α = priors.seq_event_amplitude.α
    β = priors.seq_event_amplitude.β

    # TODO : make this a field of globals, not model.
    model.new_cluster_log_prob = (
        log(α)
        + log(model.priors.seq_event_rate)  # TODO: resample this as well.
        + log(model.max_time)
        + α * (log(β) - log(1 + β))
    )

    # TODO : make this a field of globals, not model.
    model.bkgd_log_prob = (
        log(globals.bkgd_amplitude)
        + log(model.max_time)
        + log(1 + β)
    )

    # === RECOMPUTE SUFFICIENT STATISTICS === #

    # Because the sequence types have been resampled, the sufficient statistics
    # computed for each sequence event must be re-computed (they depend on the
    # neuron offsets, relative amplitudes, etc.!)

    recompute!(model, spikes, assignments)

end


"""
Resamples sequence type, timestamp, and amplitude. For all
latent events.
"""
function gibbs_update_latents!(model::SeqModel,
                               config::Dict)

    # Grab length-R vector (already pre-allocated).
    log_probs = model._RW_buffer

    for event in model.sequence_events

        # We should only be resampling non-empty events.
        @assert (event.spike_count > 0)

        # Sample sequence type.
        log_probs .= event.seq_type_posterior
        ind = sample_logprobs!(vec(log_probs))
        r, w = Tuple(CartesianIndices(size(log_probs))[ind])
        event.sampled_type = r
        event.sampled_warp = w

        # Sample event time, t ~ N(μ, σ2), given sequence type.
        σ2 = 1 / event.summed_precisions[r, w]
        μ = event.summed_potentials[r, w] * σ2
        event.sampled_timestamp = μ + randn() * sqrt(σ2)

        # Sample event amplitude.
        event.sampled_amplitude = rand(
            posterior(
                event.spike_count,
                model.priors.seq_event_amplitude
            )
        ) # Poisson - gamma conjugate pair.

    end

end
