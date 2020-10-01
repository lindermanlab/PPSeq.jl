# ============================= #
# ===                       === #
# === Predictive Likelihood === #
# ===                       === #
# ============================= #

"""
Computes:

    log p ( x | x_1, ..., x_m, θ, z_k)

where `x` is a new spike, {x_1, ..., x_m} are a set of spikes that are
currently assigned to a latent event `z_k`, and `θ` are global
parameters (neuron offsets, etc.).
"""
function log_posterior_predictive(event::SeqEvent, x::Spike, model::SeqModel)

    # Only valid for non-empty events.
    @assert event.spike_count > 0

    # Shorter names for convenience.
    n = x.neuron
    log_p_neuron = model.globals.neuron_response_log_proportions
    offsets = model.globals.neuron_response_offsets
    widths = model.globals.neuron_response_widths
    warps = model.priors.warp_values

    # Grab size (R,W) matrix (already pre-allocated)
    log_prob = model._RW_buffer

    # Compute p({x1...xm} | r)
    for r = 1:num_sequence_types(model)
        for w in 1:num_warp_values(model)

            m = event.summed_potentials[r, w]
            v = event.summed_precisions[r, w]
            μ = m / v
            σ2 = 1 / v

            log_prob[r, w] = (
                event.seq_type_posterior[r, w]   # p( r, w | x1 ... xm)
                + log_p_neuron[n, r]             # p( n | r)
                + normlogpdf(                    # ∫ p( t | s) p(s | x1...xm, r) dτ
                    μ + offsets[n, r] * warps[w],
                    sqrt(σ2 + widths[n, r] * (warps[w] ^ 2)),
                    x.timestamp)
            )

            # See pg. 41 of Gelman et al., Bayesian Data Analysis 3rd Ed. for
            # computing integral ∫ p( t | s) p(s | μ, σ2, r) dτ
        end
    end

    # Return log p(x | x1...xm, Θ)
    return logsumexp(log_prob)
end


"""
Computes:

    log p( {x} | θ )

Where `{x}` is a singleton cluster containing one spike `x`
and `θ` are global parameters (neuron offsets, etc.),
"""
function log_posterior_predictive(x::Spike, model::SeqModel)

    R = num_sequence_types(model)
    W = num_warp_values(model)
    n = x.neuron

    log_p_neuron = model.globals.neuron_response_log_proportions
    log_p_seqtype = model.globals.seq_type_log_proportions
    log_p_warps = model.priors.warp_log_proportions

    # Grab length-R array
    log_prob = model._R_buffer

    # Compute probability of singleton: p(x) = sum_r ( p(n | r) p(r) )
    # Note: no need to sum over warps as they do not affect the likelihood
    # of a spike at time t on neuron n.
    for r = 1:R
        log_prob[r] = log_p_seqtype[r] + log_p_neuron[n, r]
    end

    return logsumexp(log_prob) - log(model.max_time)

end

# =============================================== #
# ===                                         === #
# === Prior distribution on global parameters === #
# ===                                         === #
# =============================================== #

"""
Computes:

    log p({θ_r} | ξ)

where {θ_r} is the set of global parameters for each sequence
type r.
"""
function log_prior(model::SeqModel)

    priors = model.priors
    globals = model.globals

    R = priors.seq_type_proportions.dim  # num sequence types
    N = priors.bkgd_proportions.dim      # num neurons  

    # Rate of background spikes.
    lp = logpdf(
        priors.bkgd_amplitude,           # gamma distribution.
        globals.bkgd_amplitude           # nonnegative rate parameter.
    )

    # Background spike proportions across neurons.
    lp += logpdf(
        priors.bkgd_proportions,            # dirichlet distribution.
        exp.(globals.bkgd_log_proportions)  # probability vector.
    )

    # Add contribution of each sequence type.
    for r = 1:R

        # Log prior on neuron amplitudes for each sequence type.
        lp += logpdf(
            priors.neuron_response_proportions,                  # dirichlet distribution.
            exp.(globals.neuron_response_log_proportions[:, r])  # probability vector.
        )
    
        # Neuron offsets and widths for each sequence type.
        for n = 1:N
            lp += logpdf(
                priors.neuron_response_profile,         # norm-inv-chisq distribution.
                globals.neuron_response_offsets[n, r],  # mean parameter.
                globals.neuron_response_widths[n, r]    # variance parameter.
            )
        end

    end

    return lp
end

# ==================================================== #
# ===                                              === #
# === Log probability density of the latent events === #
# ===                                              === #
# ==================================================== #


"""
Computes:

    log p({z_k} | {θ_r}, η)

where {z_k} is the current set of latent events, {θ_r}
is the set of global parameters for each sequence type r.
"""
function log_p_latents(model::SeqModel)
    
    priors = model.priors
    globals = model.globals
    sequence_events = model.sequence_events

    lp = 0.0
    α = priors.seq_event_amplitude.α
    β = priors.seq_event_amplitude.β

    # # === Log probability of the number and size of partitions === #
    #
    # # Add log V(t) term, in notation of Miller & Harrison (2018).
    # lp += log_p_num_events(model, num_sequence_events(model))
    #
    # # Add contribution of cluster sizes    
    # lp -= lgamma(α) * num_sequence_events(model)
    #
    # for event in sequence_events
    #     lp += lgamma(event.spike_count + α)
    # end
    #
    # # === Log probability of sequence event types and amplitudes === #

    # For each latent event add...
    for event in sequence_events

        # the log prior on event amplitude.
        lp += logpdf(
            priors.seq_event_amplitude,  # gamma distribution.
            event.sampled_amplitude      # nonnegative rate parameter.
        )

        # the log probability of event type.
        lp += globals.seq_type_log_proportions[event.sampled_type]

        # the log probability of event warp.
        lp += priors.warp_log_proportions[event.sampled_warp]

    end

    return lp
end

# ====================== #
# ===                === #
# === Log Likelihood === #
# ===                === #
# ====================== #

"""
Computes the log likelihood of observed events, conditioned
on the current set of latent events and global parameters.

    log p({x_s} | {z_k}, {θ_r})
"""    
function log_like(
        model::SeqModel,
        spikes::Vector{Spike}
    )

    max_time = model.max_time
    globals = model.globals
    sequence_events = model.sequence_events

    bkgd_amplitude = globals.bkgd_amplitude
    offsets = globals.neuron_response_offsets
    widths = globals.neuron_response_widths
    warps = model.priors.warp_values
    ll = 0.0

    # == FIRST TERM == #
    # -- Sum of Poisson Process intensity at all spikes -- #

    for x in spikes

        # Compute intensity.
        g = bkgd_amplitude * exp(globals.bkgd_log_proportions[x.neuron])
        for event in sequence_events
            w = warps[event.sampled_warp]
            μ = event.sampled_timestamp + offsets[x.neuron, event.sampled_type] * w
            σ2 = widths[x.neuron, event.sampled_type] * (w ^ 2)
            g += event.sampled_amplitude * normpdf(μ, sqrt(σ2), x.timestamp)
        end

        # Add term to log-likelihood.
        ll += log(g)
    end

    # == SECOND TERM == #
    # -- Penalty on integrated intensity function -- #

    ll -= bkgd_amplitude * max_time
    for event in sequence_events
        ll -= event.sampled_amplitude
    end

    return ll

end


# ================================ #
# ===                          === #
# === Per Event Log Likelihood === #
# ===                          === #
# ================================ #

function event_log_like(
        model::SeqModel,
        spikes::AbstractVector{Spike}
    )

    # Compute likelihood of initial spike.
    ll = log_posterior_predictive(spikes[1], model)

    # Return early if one spike.
    length(spikes) == 1 && return ll

    # Create singleton cluster (at index k).
    k = add_event!(model, spikes[1])
    event = model.sequence_events[k]

    # Account for remaining spikes.
    i = 2
    while true

        # Compute likelihood, conditioned on spikes <i.
        ll += log_posterior_predictive(event, spikes[i], model)

        # Return if we reached the last spike.
        if i == length(spikes)
            remove_event!(model.sequence_events, k)
            return ll
        end

        # Otherwise, add spike to event to update new posterior.
        add_datapoint!(model, spikes[i], k)

        # Increment to next spike.
        i += 1
    end

end
