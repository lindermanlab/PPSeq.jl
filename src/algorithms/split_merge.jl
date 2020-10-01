
function split_merge_sample!(
        model::SeqModel,
        spikes::Vector{Spike},
        num_samples::Int64,
        assignments::Vector{Int64},
        split_merge_window::Float64
    )

    # Return early if no samples.
    num_samples == 0 && return assignments

    # Arrays holding indices for split clusters.
    A = Int64[]
    B = Int64[]
    C = Int64[]

    # Spike indices not assigned to the background.
    non_bkgd_idx = findall(c->(c != -1), assignments)

    # Return early if everything is assigned to the background.
    length(non_bkgd_idx) < 3 && return assignments

    # Concentration parameter.
    α = model.priors.seq_event_amplitude.α

    for s = 1:num_samples

        # Sample random spike.
        i = rand(non_bkgd_idx)
        ti = spikes[i].timestamp
        
        # Sample another random spike, within 'split_merge_window' time
        # units from spike i. Do this by placing all eligible spikes in
        # set 'A' and then sampling a spike index j from A.
        for k in non_bkgd_idx
            tk = spikes[k].timestamp
            if (i != k) && (abs(ti - tk) < split_merge_window)
                push!(A, k)
            end
        end

        # Failed to identify any valid proposal.
        isempty(A) && continue

        # Sample second spike and empty set A.
        j = rand(A)
        empty!(A)

        # Cluster assignment for spike i and spike j.
        ci = assignments[i]
        cj = assignments[j]

        # Propose split.
        if ci == cj

            # println("proposing split...")

            # Put spikes[i] in set A, spikes[j] in set B.
            push!(A, i)
            push!(B, j)

            # Randomly assign other spikes in cluster to A or B.
            for k in non_bkgd_idx
                if (k != i) && (k != j) && (assignments[k] == ci)
                    if rand() > 0.5
                        push!(A, k)
                    else
                        push!(B, k)
                    end
                end
            end

            # Create combined list of all spikes.
            append!(C, A)
            append!(C, B)

            # Compute log acceptance probability.
            accept_log_prob = (
                event_log_like(model, view(spikes, A))
                + event_log_like(model, view(spikes, B))
                - event_log_like(model, view(spikes, C))
                + lgamma(length(A) + α)
                + lgamma(length(B) + α)
                - lgamma(length(C) + α)
                - lgamma(α)
                + model.new_cluster_log_prob
                - (length(C) - 2) * log(0.5)
            )
            # println("  accept_log_prob : ", accept_log_prob)

            # Accept move.
            if rand() < exp(accept_log_prob)

                # println("!!!!!! SPLIT ACCEPTED !!!!!!!!!!")
                # println("num events before = ", length(model.sequence_events))

                # Create new event for spike j and move it there.
                remove_datapoint!(model, spikes[j], assignments[j])
                k = add_event!(model, spikes[j])
                assignments[j] = k

                # Move other spikes in set B to the new event at index k.
                for b in B
                    if b != j
                        remove_datapoint!(model, spikes[b], assignments[b])
                        assignments[b] = add_datapoint!(model, spikes[b], k)
                    end
                end

                # @assert model.sequence_events[ci].spike_count == length(A)
                # @assert model.sequence_events[k].spike_count == length(B)
                # @assert sum(assignments .== ci) == length(A)
                # @assert sum(assignments .== k) == length(B)

                # println("num events after = ", length(model.sequence_events))

                # Spike i and other spikes in set A stay where they are now.
            end

            # Empty arrays for future split moves.
            empty!(A)
            empty!(B)
            empty!(C)

        # Propose merge.
        else

            # println("proposing merge...")

            # Put spikes[i] in A, spikes[j] in B.
            push!(A, i)
            push!(B, j)

            # Put spikes where (assignment == ci) in A.
            # Put spikes where (assignment == cj) in B.
            for k in non_bkgd_idx
                if (k != i) && (k != j)
                    if (assignments[k] == ci)
                        push!(A, k)
                    elseif (assignments[k] == cj)
                        push!(B, k)
                    end
                end
            end

            # Create combined list of all spikes.
            append!(C, A)
            append!(C, B)

            # Compute log acceptance probability.
            accept_log_prob = (
                event_log_like(model, view(spikes, C))
                - event_log_like(model, view(spikes, A))
                - event_log_like(model, view(spikes, B))
                + lgamma(length(C) + α)
                + lgamma(α)
                - lgamma(length(A) + α)
                - lgamma(length(B) + α)
                - model.new_cluster_log_prob
                + (length(C) - 2) * log(0.5)
            )
            # println("  accept_log_prob : ", accept_log_prob)

            # Accept move.
            if rand() < exp(accept_log_prob)

                # println("!!!!!! MERGE ACCEPTED !!!!!!!!!!")
                # println("num events before = ", length(model.sequence_events))

                # Move spike j and others in set B to cluster ci.
                for b in B
                    remove_datapoint!(model, spikes[b], assignments[b])
                    assignments[b] = add_datapoint!(model, spikes[b], ci)
                end

                # @assert model.sequence_events[cj].spike_count == 0
                # @assert sum(assignments .== cj) == 0
                # @assert sum(assignments .== ci) == length(C)

                # println("num events after = ", length(model.sequence_events))

                # Spike i and other spikes in set A stay where they are now.
            end

            # Empty arrays for future merge moves.
            empty!(A)
            empty!(B)
            empty!(C)

        end

    end

    return assignments

end
