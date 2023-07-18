"""
Trains SeqModel given a config dict.
Returns dict containing results.
Differs from easy_sample.jl in that it uses masks and masked gibbs sampling.
Used for doing cross-validation on speckled hold-out data.
"""
function easy_sample_masked!(
        model::Union{SeqModel,DistributedSeqModel},
        spikes::Vector{Spike},
        masks::Vector{Mask},
        initial_assignments::Vector{Int64},
        config::Dict;
        callback=(args...) -> nothing,
    )

    # Save copy of initial assignments.
    _inits = copy(initial_assignments)

    # Draw annealed masked Gibbs samples.
    (
        unmasked_assignments,
        anneal_assignment_hist,
        anneal_train_log_p_hist,
        anneal_test_log_p_hist,
        anneal_latent_event_hist,
        anneal_globals_hist
    ) =
    annealed_masked_gibbs!(
        model,
        spikes,
        masks,
        initial_assignments,
        config[:num_anneals],
        config[:max_temperature],
        config[:num_spike_resamples_per_anneal],
        config[:samples_per_resample],
        config[:split_merge_moves_during_anneal],
        config[:split_merge_window],
        config[:save_every_during_anneal];
        verbose=true,
        callback=callback,
    )

    # Draw regular masked Gibbs samples.
    (
        unmasked_assignments,
        assignment_hist,
        train_log_p_hist,
        test_log_p_hist,
        latent_event_hist,
        globals_hist
    ) =
    masked_gibbs!(
        model,
        spikes,
        masks,
        unmasked_assignments,
        config[:num_spike_resamples],
        config[:samples_per_resample],
        config[:split_merge_moves_after_anneal],
        config[:split_merge_window],
        config[:save_every_after_anneal];
        verbose=true,
        callback=callback,
    )

    # return the results
    return Dict(
        
        # Initial assignment variables.
        :initial_assignments => _inits,

        # Results during annealing.
        :anneal_assignment_hist => anneal_assignment_hist,
        :anneal_train_log_p_hist => anneal_train_log_p_hist,
        :anneal_test_log_p_hist => anneal_test_log_p_hist,
        :anneal_latent_event_hist => anneal_latent_event_hist,
        :anneal_globals_hist => anneal_globals_hist,

        # Results after annealing.
        :final_assignments => unmasked_assignments,
        :assignment_hist => assignment_hist,
        :latent_event_hist => latent_event_hist,
        :globals_hist => globals_hist,
        :train_log_p_hist => train_log_p_hist,
        :test_log_p_hist => test_log_p_hist
        )
end
