function DistributedSeqModel(
    # constants
    num_partitions::Int64,
    max_time::Float64,
    max_sequence_length::Float64,

    # warp parameters
    num_warp_values::Int64,
    max_warp::Float64,
    warp_variance::Float64,
    warp_type::Int64,

    # priors
    seq_event_rate::Float64,
    seq_type_proportions::SymmetricDirichlet,
    seq_event_amplitude::RateGamma,
    neuron_response_proportions::SymmetricDirichlet,
    neuron_response_profile::NormalInvChisq,
    bkgd_amplitude::RateGamma,
    bkgd_proportions::SymmetricDirichlet
)

    primary_model = SeqModel(
        max_time,
        max_sequence_length,
        num_warp_values,
        max_warp,
        warp_variance,
        warp_type,
        seq_event_rate,
        seq_type_proportions,
        seq_event_amplitude,
        neuron_response_proportions,
        neuron_response_profile,
        bkgd_amplitude,
        bkgd_proportions
    )

    return DistributedSeqModel(
        primary_model,
        num_partitions
    )
end

function DistributedSeqModel(primary_model::SeqModel, num_partitions::Int64)

    # Number of neurons (N), sequence types (R), and warps (W).
    N = num_neurons(primary_model)
    R = num_sequence_types(primary_model)
    W = num_warp_values(primary_model)


    # Even though the submodels are each allocated a time interval of
    # (max_time / num_partitions), we set "max_time", "bkgd_log_prob",
    # and "new_cluster_log_prob" to be the same as the primary model.
    # Thus, all the probability calculations within the submodels match
    # the primary model.

    submodels = SeqModel[]
    for part = 1:num_partitions
        push!(
            submodels,
            SeqModel(
                primary_model.max_time,
                primary_model.max_sequence_length,

                primary_model.priors,
                primary_model.globals,
                SeqEventList(R, W),
                primary_model.new_cluster_log_prob,
                primary_model.bkgd_log_prob,
                zeros(R),    # _R_buffer
                zeros(R, W), # _RW_buffer
                Float64[],   # _K_buffer
            )
        )
    end

    return DistributedSeqModel(
        primary_model,
        num_partitions,
        submodels
    )
end

num_neurons(model::DistributedSeqModel) = 
    model.primary_model.priors.bkgd_proportions.dim
num_sequence_types(model::DistributedSeqModel) = 
    model.primary_model.priors.seq_type_proportions.dim
num_sequence_events(model::DistributedSeqModel) = 
    length(model.primary_model.sequence_events)

log_prior(model::DistributedSeqModel) = 
    log_prior(model.primary_model)

log_p_latents(model::DistributedSeqModel) = 
    log_p_latents(model.primary_model)

log_like(model::DistributedSeqModel, spikes::Vector{Spike}) = 
    log_like(model.primary_model, spikes)

log_joint(model::DistributedSeqModel, spikes::Vector{Spike}) = 
    log_joint(model.primary_model, spikes)

sample(model::DistributedSeqModel; kwargs...) = sample(model.primary_model; kwargs...)

priors(model::DistributedSeqModel) =
    return model.primary_model.priors

function set_priors!(model::DistributedSeqModel, priors::SeqPriors)
    model.primary_model.priors = priors
    for submodel in model.submodels
        submodel.priors = priors
    end
end

function set_new_cluster_log_prob!(model::DistributedSeqModel, prob::Float64)
    # Since we constructed each submodel.max_time == primary_model.max_time, the
    # probability of forming new clusters is the same.
    model.primary_model.new_cluster_log_prob = prob
    for submodel in model.submodels
        submodel.new_cluster_log_prob = prob
    end
end

get_max_time(model::DistributedSeqModel) = model.primary_model.max_time

get_bkgd_amplitude(model::DistributedSeqModel) = model.primary_model.globals.bkgd_amplitude

function set_bkgd_log_prob!(model::DistributedSeqModel, prob::Float64)
    # Since we constructed each submodel.max_time == primary_model.max_time, the
    # probability of attributing spikes to the background is the same.
    model.primary_model.bkgd_log_prob = prob
    for submodel in model.submodels
        submodel.bkgd_log_prob = prob
    end
end
