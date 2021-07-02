
"""
Run Gibbs sampler on masked spike train. Alternate 
between imputing data in masked regions and updating
the model through classic gibbs_sample!(...) function.
"""
function masked_gibbs!(
        model::Union{SeqModel,DistributedSeqModel},
        masked_spikes::Vector{Spike},
        unmasked_spikes::Vector{Spike},
        masks::Vector{Mask},
        initial_assignments::Vector{Int64},
        num_spike_resamples::Int64,
        samples_per_resample::Int64,
        extra_split_merge_moves::Int64,
        split_merge_window::Float64,
        save_every::Int64;
        verbose::Bool=true
    )

    sampled_spikes = Spike[]
    sampled_assignments = Int64[]

    # Compute proportion of the data that is masked.
    masked_proportion = 0.0
    for (_, (t0, t1)) in masks
        masked_proportion += (t1 - t0)
    end
    masked_proportion /= (model.max_time * num_neurons(model))
    @show masked_proportion

    # Create inverted masks to compute train log likelihood.
    inv_masks = compute_complementary_masks(
        masks, num_neurons(model), model.max_time+0.000000001)

    # Sanity check.
    assert_spikes_in_mask(masked_spikes, masks)
    assert_spikes_in_mask(unmasked_spikes, inv_masks)
    assert_spikes_not_in_mask(masked_spikes, inv_masks)
    assert_spikes_not_in_mask(unmasked_spikes, masks)

    # Compute log-likelihood of a homogeneous Poisson process in
    # the train and test sets.
    train_baseline = homogeneous_baseline_log_like(unmasked_spikes, inv_masks)
    test_baseline = homogeneous_baseline_log_like(masked_spikes, masks)

    @show train_baseline test_baseline

    n_unmasked = length(unmasked_spikes)
    assignment_hist = zeros(Int64, n_unmasked, 0)
    train_log_p_hist = Float64[]
    test_log_p_hist = Float64[]
    latent_event_hist = Vector{EventSummaryInfo}[]
    globals_hist = SeqGlobals[]

    unmasked_assignments = initial_assignments

    for i = 1:num_spike_resamples

        # Sample new spikes in each masked region.
        sample_masked_spikes!(
            sampled_spikes,
            sampled_assignments,
            model,
            masks
        )

        # assert_spikes_in_mask(sampled_spikes, masks)
        # assert_spikes_not_in_mask(sampled_spikes, inv_masks)

        # Run gibbs sampler. Note that sufficient statistics are
        # recomputed at the beginning of gibb_sample!(...) so all
        # events will have the appropriate spike assignments / initialization
        (
            assignments,
            _assgn_hist,
            _lp_hist,
            _latents,
            _globals
        ) = 
        gibbs_sample!(
            model,
            vcat(unmasked_spikes, sampled_spikes),
            vcat(unmasked_assignments, sampled_assignments),
            samples_per_resample,
            extra_split_merge_moves,
            split_merge_window,
            save_every;
            verbose=false
        )

        # Update initial assignments for next Gibbs sampler run.
        unmasked_assignments .= view(assignments, 1:n_unmasked)

        # Save history
        assignment_hist = cat(
            assignment_hist,
            view(_assgn_hist, 1:n_unmasked, :),
            dims=2
        )

        # Evaluate model likelihood on observed spikes.        
        push!(
            train_log_p_hist,
            log_like(model, unmasked_spikes, inv_masks) - train_baseline
        )

        # Evaluate model likelihood on heldout spikes.
        push!(
            test_log_p_hist,
            log_like(model, masked_spikes, masks) - test_baseline
        )

        append!(latent_event_hist, _latents)
        append!(globals_hist, _globals)

        verbose && print(i * samples_per_resample, "-")

    end

    verbose && println("Done")

    # Before returning, remove assignments assigned to imputed spikes.
    recompute!(model, unmasked_spikes, unmasked_assignments)

    # Rescale train and test log likelihoods.
    train_log_p_hist ./= ((1 - masked_proportion) * model.max_time * num_neurons(model))
    test_log_p_hist ./= (masked_proportion * model.max_time * num_neurons(model))

    return (
        unmasked_assignments,
        assignment_hist,
        train_log_p_hist,
        test_log_p_hist,
        latent_event_hist,
        globals_hist
    )

end


"""
Annealed Gibbs sampling with masked data.
"""
function annealed_masked_gibbs!(
        model::Union{SeqModel,DistributedSeqModel},
        spikes::Vector{Spike},
        masks::Vector{Mask},
        initial_assignments::Vector{Int64},
        num_anneals::Int64,
        max_temperature::Float64,
        num_spike_resamples_per_anneal::Int64,
        samples_per_resample::Int64,
        extra_split_merge_moves::Int64,
        split_merge_window::Float64,
        save_every::Int64;
        verbose::Bool=true
    )

    masked_spikes, unmasked_spikes = split_spikes_by_mask(spikes, masks)

    target_mean = mean(priors(model).seq_event_amplitude)
    target_var = var(priors(model).seq_event_amplitude)

    temperatures = exp10.(range(log10(max_temperature), 0, length=num_anneals))

    unmasked_assignments = fill(-1, length(unmasked_spikes))
    assignment_hist = zeros(Int64, length(unmasked_spikes), 0)
    train_log_p_hist = Float64[]
    test_log_p_hist = Float64[]
    latent_event_hist = Vector{EventSummaryInfo}[]
    globals_hist = SeqGlobals[]

    for temp in temperatures
        
        # Print progress.
        verbose && println("TEMP:  ", temp)

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
        prior = priors(model)
        α = prior.seq_event_amplitude.α
        β = prior.seq_event_amplitude.β
        λ = prior.seq_event_rate
        set_new_cluster_log_prob!(
            model,
            log(α) + log(λ) + log(model.max_time) + α * (log(β) - log(1 + β))
        )
        model.bkgd_log_prob = (
            log(model.globals.bkgd_amplitude)
            + log(model.max_time)
            + log(1 + β)
        )

        # Draw gibbs samples.
        (
            unmasked_assignments,
            _assgn,
            _train_hist,
            _test_hist,
            _latents,
            _globals
        ) =
        masked_gibbs!(
            model,
            masked_spikes,
            unmasked_spikes,
            masks,
            unmasked_assignments,
            num_spike_resamples_per_anneal,
            samples_per_resample,
            extra_split_merge_moves,
            split_merge_window,
            save_every;
            verbose=verbose
        )

        # Save samples.
        assignment_hist = cat(assignment_hist, _assgn, dims=2)
        append!(train_log_p_hist, _train_hist)
        append!(test_log_p_hist, _test_hist)
        append!(latent_event_hist, _latents)
        append!(globals_hist, _globals)

    end

    return (
        unmasked_assignments,
        assignment_hist,
        train_log_p_hist,
        test_log_p_hist,
        latent_event_hist,
        globals_hist
    )

end


function masked_gibbs!(
        model::Union{SeqModel,DistributedSeqModel},
        spikes::Vector{Spike},
        masks::Vector{Mask},
        initial_assignments::Vector{Int64},
        num_spike_resamples::Int64,
        samples_per_resample::Int64,
        extra_split_merge_moves::Int64,
        split_merge_window::Float64,
        save_every::Int64;
        verbose::Bool=true
    )

    masked_spikes, unmasked_spikes = split_spikes_by_mask(spikes, masks)

    return masked_gibbs!(
        model,
        masked_spikes,
        unmasked_spikes,
        masks,
        initial_assignments,
        num_spike_resamples,
        samples_per_resample,
        extra_split_merge_moves,
        split_merge_window,
        save_every;
        verbose=verbose
    )
end


"""
Impute missing data
"""
function sample_masked_spikes!(
        spikes::Vector{Spike},
        assignments::Vector{Int64},
        model::SeqModel,
        masks::Vector{Mask}
    )

    empty!(spikes)
    empty!(assignments)
    globals = model.globals

    # Sample background spikes.
    S_bkgd = rand(Poisson(
        globals.bkgd_amplitude * model.max_time))
    bkgd_dist = Categorical(exp.(globals.bkgd_log_proportions))

    n_bkgd = rand(bkgd_dist, S_bkgd)
    t_bkgd = rand(S_bkgd) * model.max_time

    for (sampled_n, sampled_t) in zip(n_bkgd, t_bkgd)
        for (n, (start, stop)) in masks
            if (start < sampled_t < stop) && (sampled_n == n)
                push!(spikes, Spike(n, sampled_t))
                push!(assignments, -1)
                break
            end
        end
    end

    # Compute neuron probabilities.
    neuron_rel_amps =
        exp.(globals.neuron_response_log_proportions)
    neuron_dists = 
        [Categorical(neuron_rel_amps[:, i]) for i = 1:size(neuron_rel_amps, 2)]

    # Sample sequence-evoked spikes.
    for (k, event) in enumerate(model.sequence_events)
        
        # Assignment id for latent event.
        z = model.sequence_events.indices[k]

        # Num spikes evoked by latent event.
        S = rand(Poisson(event.sampled_amplitude))
        
        # Sample neuron, then spike time, for each spike.
        for n in rand(neuron_dists[event.sampled_type], S)
            
            # Sample spike time.
            μ = globals.neuron_response_offsets[n, event.sampled_type]
            σ = sqrt(globals.neuron_response_widths[n, event.sampled_type])
            w = model.priors.warp_values[event.sampled_warp]
            t = w * (σ * randn() + μ) + event.sampled_timestamp
                     
            # Check if spike is inside a masked region.
            for (n_mask, (mask_start, mask_stop)) in masks
                if (mask_start < t < mask_stop) && (n == n_mask)
                    push!(spikes, Spike(n, t))
                    push!(assignments, z)
                    break
                end
            end

        end
    end

    return spikes, assignments

end

sample_masked_spikes!(
    spikes::Vector{Spike},
    assignments::Vector{Int64},
    model::DistributedSeqModel,
    masks::Vector{Mask}
) = sample_masked_spikes!(spikes, assignments, model.primary_model, masks)


# ===========
#
# Helper functions to create masks, split spikes.
#
# ===========

function compute_complementary_masks(
        masks::Vector{Mask},
        num_neurons::Int64,
        max_time::Float64
    )

    inverted_masks = [[(n, (0.0, max_time))] for n in 1:num_neurons]

    for (n, (t0, t1)) in masks
        for i = 1:length(inverted_masks[n])
            n_, (v0, v1) = inverted_masks[n][i]
            @assert n_ == n
            if (t0 >= v0) && (t1 <= v1)
                deleteat!(inverted_masks[n], i)
                push!(
                    inverted_masks[n],
                    (n, (v0, t0))
                )
                push!(
                    inverted_masks[n],
                    (n, (t1, v1))
                )
                break
            end
            @assert (i + 1) != length(inverted_masks)
        end
    end

    return vcat(inverted_masks...)

end

function create_random_mask(
        num_neurons::Integer,
        max_time::Real,
        mask_lengths::Real,
        percent_masked::Real
    )

    @assert num_neurons > 0
    @assert max_time > mask_lengths
    @assert mask_lengths > 0
    @assert 0 <= percent_masked < 100

    T_masked = percent_masked * max_time * num_neurons / 100.0
    n_masks = Int(round(T_masked / mask_lengths))

    intervals = Tuple{Float64, Float64}[]
    for start in range(0, max_time - mask_lengths, step=mask_lengths)
        push!(intervals, (start, start + mask_lengths))
    end

    sample(
        collect(Iterators.product(1:num_neurons, intervals)),
        n_masks, replace=false
    )
    
end

function create_blocked_mask(
        num_neurons::Integer,
        max_time::Real
    )

    masked_neurons = sample(1:num_neurons, num_neurons ÷ 2, replace=false)

    masked_intervals = vcat(
        repeat([(0.0, max_time / 2)], div(num_neurons ÷ 2, 2)),
        repeat([(max_time / 2, max_time)], cld(num_neurons ÷ 2, 2))
    )

    return collect(zip(masked_neurons, masked_intervals))

end



function split_spikes_by_mask(
        spikes::Vector{Spike},
        masks::Vector{Mask}
    )

    # Save list of spikes that are masked out.
    masked_spikes = Spike[]
    unmasked_spikes = Spike[]

    for x in spikes

        # See if x falls within any masked region.
        is_masked = false
        for (n, (start, stop)) in masks
            if (start < x.timestamp < stop) && (x.neuron == n)
                is_masked = true
                push!(masked_spikes, x)
                break
            end
        end

        # Mark x as unmasked if no match was found.
        !(is_masked) && push!(unmasked_spikes, x)
    end

    return masked_spikes, unmasked_spikes
end


function assert_spikes_in_mask(
        spikes::Vector{Spike},
        masks::Vector{Mask}
    )
    # Check that all spikes are in masked region.
    for x in spikes
        is_in_mask = false
        for (n, (t0, t1)) in masks
            if (t0 < x.timestamp < t1) && (x.neuron == n)
                is_in_mask = true
            end
        end
        if !(is_in_mask)
            @show x
            @assert false "spike is falsely excluded from mask..."
        end
    end
end

function assert_spikes_not_in_mask(
        spikes::Vector{Spike},
        masks::Vector{Mask}
    )
    # Check that all spikes are in masked region.
    for x in spikes
        for (n, (t0, t1)) in masks
            if (t0 < x.timestamp < t1) && (x.neuron == n)
                @show x
                @show (n, (t0, t1))
                @assert false "spike is falsely included in mask..."
            end
        end
    end
end

function clean_masks(
    masks::Vector{Mask},
    num_neurons::Integer
)
    new_masks = Mask[]
    for n = 1:num_neurons
        # Find this neuron's masks
        this_neur_masks = Mask[]
        for mask in masks
            if mask[1] == n
                push!(this_neur_masks, mask)
            end
        end

        # Find the start times of this neuron's masks
        start_times = []
        for (n,(t0,t1)) in this_neur_masks
            push!(start_times, t0)
        end

        # Sort these masks by start time.
        sorted_mask = this_neur_masks[sortperm(start_times)]

        # Find the offending masks, those whose start time is less than some small threshold after another mask's end time
        indices_to_delete = []
        for index = 1:(length(sorted_mask)-1)
            if (sorted_mask[index+1][2][1] - sorted_mask[index][2][2] < 0.0001)
                push!(indices_to_delete, index)
            end
        end

        # Remove them sequentially, in each case merging them into the mask occurring first.         
        counter = 0
        for index = 1:length(sorted_mask)
            if index in indices_to_delete
                counter = counter + 1
            else
                push!(new_masks, (sorted_mask[index][1],(sorted_mask[index-counter][2][1],sorted_mask[index][2][2])))
                counter = 0
            end
        end
    end

    return new_masks
end
