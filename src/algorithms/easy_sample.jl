
"""
Trains SeqModel model given a config dict.

Returns dict containing results.
"""
function easy_sample!(
        model::SeqModel,
        spikes::Vector{Spike},
        initial_assignments::Vector{Int64},
        config::Dict
    )

    # Save copy of initial assignments.
    _inits = copy(initial_assignments)

    # Draw annealed Gibbs samples.
    (
        assignments,
        anneal_assignment_hist,
        anneal_log_p_hist,
        anneal_latent_event_hist,
        anneal_globals_hist
    ) =
    annealed_gibbs!(
        model,
        spikes,
        initial_assignments,
        config[:num_anneals],
        config[:samples_per_anneal],
        config[:max_temperature],
        config[:split_merge_moves_during_anneal],
        config[:split_merge_window],
        config[:save_every_during_anneal];
        verbose=true
    )

    # Sanity check.
    for k in model.sequence_events.indices
        event = model.sequence_events[k]
        @assert event.spike_count == sum(assignments .== k)
    end

    # Draw regular Gibbs samples.
    (
        assignments,
        assignment_hist,
        log_p_hist,
        latent_event_hist,
        globals_hist
    ) =
    gibbs_sample!(
        model,
        spikes,
        assignments,
        config[:samples_after_anneal],
        config[:split_merge_moves_after_anneal],
        config[:split_merge_window],
        config[:save_every_after_anneal];
        verbose=true
    )

    # Sanity check.
    for k in model.sequence_events.indices
        event = model.sequence_events[k]
        @assert event.spike_count == sum(assignments .== k)
    end

    return Dict(
        
        # Initial assignment variables.
        :initial_assignments => _inits,

        # Results during annealing.
        :anneal_assignment_hist => anneal_assignment_hist,
        :anneal_log_p_hist => anneal_log_p_hist,
        :anneal_latent_event_hist => anneal_latent_event_hist,
        :anneal_globals_hist => anneal_globals_hist,

        # Results after annealing.
        :final_assignments => assignments,
        :assignment_hist => assignment_hist,
        :latent_event_hist => latent_event_hist,
        :globals_hist => globals_hist,
        :log_p_hist => log_p_hist
    )
end
