# Each mask is a tuple, (n, (t0, t1)) indicating that the neuron
# at index n is masked over the time interval (t0, t1)
const Mask = Tuple{Int64,Tuple{Float64,Float64}}

"""
Computes log-likelihood in masked regions.
"""
function log_like(
        model::SeqModel,
        spikes::Vector{Spike},
        masks::Vector{Mask}
    )

    globals = model.globals
    offsets = globals.neuron_response_offsets
    widths = globals.neuron_response_widths
    warps = model.priors.warp_values
    ll = 0.0

    # == FIRST TERM == #
    # -- Sum of Poisson Process intensity at all spikes -- #

    for x in spikes

        # Compute intensity.
        g = globals.bkgd_amplitude * exp(globals.bkgd_log_proportions[x.neuron])
        for event in model.sequence_events
            w = warps[event.sampled_warp]
            r = event.sampled_type
            g += event.sampled_amplitude * normpdf(
                event.sampled_timestamp + w * offsets[x.neuron, r],
                w * sqrt(widths[x.neuron, r]),
                x.timestamp
            )
        end

        # Add term to log-likelihood.
        ll += log(g)
    end

    # == SECOND TERM == #
    # -- Penalty on integrated intensity function -- #

    for (n, (start, stop)) in masks
        
        # Add contribution of background.
        ll -= (
            globals.bkgd_amplitude
            * exp(globals.bkgd_log_proportions[n])
            * (stop - start)
        )

        # Add contribution of each latent event.
        for event in model.sequence_events
            w = warps[event.sampled_warp]
            r = event.sampled_type
            g = Normal(
                w * offsets[n, r] + event.sampled_timestamp,
                w * sqrt(widths[n, r]) 
            )
            ll -= (
                event.sampled_amplitude
                * exp(globals.neuron_response_log_proportions[n, r])
                * (cdf(g, stop) - cdf(g, start))
            )
        end

    end

    return ll

end


"""
Computes log-likelihood of homogeneous poisson process
of spikes within a masked region. The `spikes` vector
should only contain spikes within some masked region.
"""
function homogeneous_baseline_log_like(
        spikes::Vector{Spike},
        masks::Vector{Mask}
    )

    # Count number of spikes and total time observed for each neuron.
    time_per_neuron = Dict{Int64,Float64}()
    spikes_per_neuron = Dict{Int64,Int64}()

    for (n, (t0, t1)) in masks

        # Add neuron n to dict keys
        if !(n in keys(time_per_neuron))
            time_per_neuron[n] = 0.0
            spikes_per_neuron[n] = 0
        end

        # Add masked time interval to neuron n
        time_per_neuron[n] += (t1 - t0)
    end

    # Count number of spikes for each neuron
    for x in spikes
        spikes_per_neuron[x.neuron] += 1
    end

    # Compute MLE estimate for a homogeneous PP.
    neurons = keys(time_per_neuron)
    mle_rates = Dict(n => max(eps(), spikes_per_neuron[n]) / time_per_neuron[n] for n in neurons)

    # Use MLE to compute total log-likelihood.
    ll = 0.0
    for n in neurons
        ll += spikes_per_neuron[n] * log(mle_rates[n])
        ll -= time_per_neuron[n] * mle_rates[n]
    end
    return ll

end


log_like(
    model::DistributedSeqModel,
    spikes::Vector{Spike},
    masks::Vector{Mask}
) = log_like(model.primary_model, spikes, masks)
