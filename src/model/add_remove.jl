"""
Remove spike `s` from event at index `k`.
"""
function remove_datapoint!(
    model::SeqModel,
    s::Spike,
    k::Int64
)
    
    # Nothing to do if spike is already in background.
    (k == -1) && return

    # Remove the contributions of datapoint i to the
    # sufficient statistics of sequence event k.
    t = s.timestamp
    n = s.neuron
    event = model.sequence_events[k]

    log_p_neuron = model.globals.neuron_response_log_proportions
    offsets = model.globals.neuron_response_offsets
    widths = model.globals.neuron_response_widths
    warps = model.priors.warp_values

    # If this is the last spike in the event, we can return early.
    (event.spike_count == 1) && (return remove_event!(model.sequence_events, k))

    # Otherwise, subtract off the sufficient statistics.
    event.spike_count -= 1
    
    for r = 1:num_sequence_types(model)
        for w in 1:num_warp_values(model)
            # Compute 1D precision and potential.
            v = 1 / (widths[n, r] * warps[w]^2)
            m = (t - offsets[n, r] * warps[w]) * v
            sv = event.summed_precisions[r, w]

            # Subtract sufficient statistics.
            event.summed_potentials[r, w] -= m
            event.summed_precisions[r, w] = max(0, sv - v) # prevent negative precision.
            event.summed_logZ[r, w] -= (log_p_neuron[n, r] - gauss_info_logZ(m, v))            
            event.seq_type_posterior[r, w] = (
                model.globals.seq_type_log_proportions[r]
                + model.priors.warp_log_proportions[w]
                + event.summed_logZ[r, w]
                + gauss_info_logZ(
                    event.summed_potentials[r, w],
                    event.summed_precisions[r, w])
            )
        end
    end

    event.seq_type_posterior .-= logsumexp(event.seq_type_posterior)

end


"""
Add spike `s` to event at index `k`. Return assignment index `k`.
"""
function add_datapoint!(
        model::SeqModel,
        s::Spike,
        k::Int64;
        recompute_posterior::Bool=true
    )

    t = s.timestamp
    n = s.neuron
    event = model.sequence_events[k]

    log_p_neuron = model.globals.neuron_response_log_proportions
    offsets = model.globals.neuron_response_offsets
    widths = model.globals.neuron_response_widths
    warps = model.priors.warp_values

    event.spike_count += 1
    
    for r = 1:num_sequence_types(model)
        for w in 1:num_warp_values(model)
            # Compute 1D precision and potential.
            v = 1 / (widths[n, r] * warps[w]^2)
            m = (t - offsets[n, r] * warps[w]) * v
            
            # Add sufficient statistics.
            event.summed_potentials[r, w] += m
            event.summed_precisions[r, w] += v
            event.summed_logZ[r, w] += (log_p_neuron[n, r] - gauss_info_logZ(m, v))
        end
    end

    if recompute_posterior
        set_posterior!(model, k)
    end

    return k
end


function set_posterior!(
    model::SeqModel,
    k::Int64;
)
    event = model.sequence_events[k]

    for r = 1:num_sequence_types(model)
        for w in 1:num_warp_values(model)
            event.seq_type_posterior[r, w] = (
                model.globals.seq_type_log_proportions[r]
                + model.priors.warp_log_proportions[w]
                + event.summed_logZ[r, w]
                + gauss_info_logZ(
                    event.summed_potentials[r, w],
                    event.summed_precisions[r, w])
            )
        end
    end

    event.seq_type_posterior .-= logsumexp(event.seq_type_posterior)
end



"""
Create a singleton cluster / sequence event with spike `s`
and return new assignment index `k`.
"""
function add_event!(model::SeqModel, s::Spike)
    # Mark event k as non-empty.
    k = add_event!(model.sequence_events)
    # Add spike s to event k.
    return add_datapoint!(model, s, k)
end
