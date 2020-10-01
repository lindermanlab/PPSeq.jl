
# ==================================================== #
# ===                                              === #
# === Methods to create, resample sequence events. === #
# ===                                              === #
# ==================================================== #

"""Constructs empty sequence event."""
function SeqEvent(
        num_sequence_types::Int64,
        num_warp_values::Int64
    )

    SeqEvent(
        0,
        zeros(num_sequence_types, num_warp_values),  # summed potentials
        zeros(num_sequence_types, num_warp_values),  # summed precisions
        zeros(num_sequence_types, num_warp_values),  # summed log Z
        zeros(num_sequence_types, num_warp_values),  # posterior on sequence type.
        -1,   # sampled_type is ignored until spike_count > 0
        -1,   # sampled_warp is ignored until spike_count > 0
        0.0,  # sampled_timestamp is ignored until spike_count > 0
        0.0   # sampled_amplitude is ignored until spike_count > 0
    )
end


"""Resets event to be empty."""
function reset!(e::SeqEvent)
    e.spike_count = 0
    fill!(e.summed_potentials, 0)
    fill!(e.summed_precisions, 0)
    fill!(e.summed_logZ, 0)
    fill!(e.seq_type_posterior, 0)
end


# =============================================================== #
# ===                                                         === #
# === Methods to maintain an array of sequence event structs. === #
# ===                                                         === #
# =============================================================== #


# Initialize SeqEventList
function SeqEventList(
        num_sequence_types::Int64,
        num_warp_values::Int64
    )
    events = [SeqEvent(num_sequence_types, num_warp_values)]  # one empty event
    indices = Int64[]
    SeqEventList(events, indices)
end

# Makes SeqEventList a read-only array.
getindex(ev::SeqEventList, i::Int64) = ev.events[i]

# Length of the event list is the number of occupied clusters.
length(ev::SeqEventList) = length(ev.indices)

# Define interface for iteration. Enables "for event in sequences"
# to iterate over the occupied sequence events.
iterate(ev::SeqEventList) = (
    isempty(ev.indices) ? nothing : (ev.events[ev.indices[1]], 2)
)

iterate(ev::SeqEventList, i::Int64) = (
    (i > length(ev)) ? nothing : (ev.events[ev.indices[i]], i + 1)
)


"""
Finds either an empty SeqEvent struct, or creates a new
one. Returns the index of the new cluster.
"""
function add_event!(ev::SeqEventList)

    # Check if any indices are skipped. If so, use the smallest skipped
    # integer as the index for the new event.
    i = 1
    for j in ev.indices

        # We have j == ev.indices[i].

        # If (indices[i] != i) then events[i] is empty.
        if i != j
            insert!(ev.indices, i, i)  # mark events[i] as occupied
            return i                   # i is the index for the new event.
        end

        # Increment to check if (i + 1) is empty.
        i += 1
    end

    # If we reached here without returning, then indices is a vector 
    # [1, 2, ..., K] without any skipped integers. So we'll use K + 1
    # as the new integer index.
    push!(ev.indices, length(ev.indices) + 1)

    # Create a new SeqEvent object if necessary.
    if length(ev.events) < length(ev.indices)
        R, W = size(ev[1].summed_potentials)
        push!(ev.events, SeqEvent(R, W))
    end

    # Return index of the empty SeqEvent struct.
    return ev.indices[end]
end


"""
Marks a SeqEvent struct as empty and resets its sufficient
statistics. This does not delete the SeqEvent.
"""
function remove_event!(ev::SeqEventList, index::Int64)
    reset!(ev.events[index])
    deleteat!(ev.indices, searchsorted(ev.indices, index))
end


"""
Recompute sufficient statistics for all sequence events.
"""
function recompute!(
        model::SeqModel,
        spikes::Vector{Spike},
        assignments::AbstractVector{Int64}
    )
    
    # Grab sequence event list.
    ev = model.sequence_events

    # Reset all events to zero spikes.
    for k in ev.indices
        reset!(ev.events[k])
    end
    empty!(ev.indices)

    # Add spikes back to their previously assigned event.
    for (s, k) in zip(spikes, assignments)
        
        # Skip background spikes.
        (k < 0) && continue

        # Check that event k exists.
        while k > length(ev.events)
            push!(
                ev.events,
                SeqEvent(
                    num_sequence_types(model),
                    num_warp_values(model)
                )
            )
        end

        # Add datapoint to event k.
        add_datapoint!(model, s, k, recompute_posterior=false)

        # Make sure that event k is marked as non-empty.
        j = searchsortedfirst(ev.indices, k)
        if (j > length(ev.indices)) || (ev.indices[j] != k)
            insert!(ev.indices, j, k)
        end
    end

    # Set the posterior, since we didn't do so when adding datapoints
    for k in ev.indices
        set_posterior!(model, k)
    end
end


"""
Returns a vector of EventSummaryInfo structs
summarizing latent events and throwing away
sufficient statistics.
"""
function event_list_summary(model::SeqModel)
    infos = EventSummaryInfo[]
    ev = model.sequence_events
    warp_vals = model.priors.warp_values

    for ind in ev.indices
        push!(
            infos,
            EventSummaryInfo(
                ind,
                ev[ind].sampled_timestamp,
                ev[ind].sampled_type,
                warp_vals[ev[ind].sampled_warp],
                ev[ind].sampled_amplitude
            )
        )
    end

    return infos
end
