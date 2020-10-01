"""
Run gibbs sampler.
"""
function gibbs_sample!(
    model::DistributedSeqModel,
    spikes::Vector{Spike},
    initial_assignments::Vector{Int64},
    num_samples::Int64,
    extra_split_merge_moves::Int64,
    split_merge_window::Float64,
    save_every::Int64;
    verbose=false
)

    if extra_split_merge_moves > 0
        @warn "Split merge not implemented for distributed model."
    end

    num_partitions = model.num_partitions
    max_time = model.primary_model.max_time

    # Partition spikes.
    split_points = Tuple(p * max_time / num_partitions for p in 0:num_partitions)
    spk_partition = Tuple(Spike[] for m in model.submodels)
    assgn_partition = [Int64[] for m in model.submodels]  # TODO -- Tuple?
    partition_ids = [Int64[] for m in model.submodels]    # TODO -- Tuple?

    for s = 1:length(spikes)
        for p = 1:model.num_partitions
            if split_points[p] <= spikes[s].timestamp <= split_points[p + 1]
                push!(spk_partition[p], spikes[s])
                push!(assgn_partition[p], initial_assignments[s])
                push!(partition_ids[p], s)
                break
            end
        end
    end

    # Dense rank assignments within each partition.
    for p = 1:model.num_partitions
        idx = assgn_partition[p] .> 0 # ignore background spikes.
        assgn_partition[p][idx] .= denserank(assgn_partition[p][idx])
    end

    # Pass assignments to submodels
    for p in 1:model.num_partitions
        recompute!(
            model.submodels[p],
            spk_partition[p],
            assgn_partition[p],
        )
    end

    # Save spike assignments over samples.
    assignments = initial_assignments
    collect_assignments!(model, assignments, assgn_partition, partition_ids) # updates assignments.

    # Order to iterate over spikes.
    spike_order_partition = Tuple(
        collect(1:length(subspikes)) for subspikes in spk_partition
    )

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

     # ======== MAIN LOOP ======== #

    # Draw samples.
    for s = 1:num_samples

        # Update spike assignments in random order.
        Threads.@threads for p in 1:model.num_partitions
            Random.shuffle!(spike_order_partition[p])
            for i in spike_order_partition[p]
                remove_datapoint!(
                    model.submodels[p], 
                    spk_partition[p][i], 
                    assgn_partition[p][i]
                )
                assgn_partition[p][i] = gibbs_add_datapoint!(
                    model.submodels[p], 
                    spk_partition[p][i]
                )
            end
        end

        # Pass spike assignment updates and latent event updates to primary model.
        # Latent events.
        collect_assignments!(model, assignments, assgn_partition, partition_ids)

        # Update latent events.
        gibbs_update_latents!(model.primary_model)

        # Update globals
        gibbs_update_globals!(model.primary_model, spikes, assignments)

        # No need to pass updates to submodels, since the globals
        # and events point to the same objects.

        # Store results
        if (s % save_every) == 0
            j = Int(s / save_every)
            
            # Save log likelihood 
            log_p_hist[j] = log_like(model, spikes)

            # Save assignments.
            assignment_hist[:, j] .= assignments

            # Save latent event information.
            push!(
                latent_event_hist,
                event_list_summary(model.primary_model)
            )

            # Save global variables.
            push!(globals_hist, deepcopy(model.primary_model.globals))

            verbose && print(s, "-")
        end
    end
    verbose && println("Done")

    return (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    )
end


"""
Collects assignments from submodels into `assignments` vector
and populates `model.primary_model.sequence_events` with
submodel's events.
"""
function collect_assignments!(
        model::DistributedSeqModel,
        assignments::Vector{Int64},
        assgn_partition::Vector{Vector{Int64}},
        partition_ids::Vector{Vector{Int64}}
    )

    empty!(model.primary_model.sequence_events.events)
    empty!(model.primary_model.sequence_events.indices)
    event_id = 1
    event_map = Dict{Tuple{Int64,Int64},Int64}()

    # Create flat assignment index for each event, to be used
    # in `primary_model`.
    for p = 1:model.num_partitions

        # Iterate over event indices in submodel-p.
        for ind in model.submodels[p].sequence_events.indices
            event_map[(p, ind)] = event_id
            push!(
                model.primary_model.sequence_events.events,
                model.submodels[p].sequence_events[ind]
            )
            push!(
                model.primary_model.sequence_events.indices,
                event_id
            )
            event_id += 1
        end

        # Assign each spike in `primary_model` using the
        # event ids generated above.
        for (s, assgn) in enumerate(assgn_partition[p])
            if assgn == -1
                assignments[partition_ids[p][s]] = -1
            else
                assignments[partition_ids[p][s]] = event_map[(p, assgn)]
            end
        end

    end
end

