# === MODEL FUNCTIONS === #

num_neurons(model::SeqModel) = model.priors.bkgd_proportions.dim
num_sequence_types(model::SeqModel) = model.priors.seq_type_proportions.dim
num_warp_values(model::SeqModel) = length(model.priors.warp_values)
num_sequence_events(model::SeqModel) = length(model.sequence_events)


"""Helper function to initialize warp probabilities."""
function default_warps(num_warp_values::Int64,
                       τ_max::Float64,
                       warp_variance::Float64)
    @assert num_warp_values >= 1
    @assert τ_max >= 1 "max warp value must be >= 1 by convention."
    @assert warp_variance >= 0

    # Initialize the warp parameters.
    if num_warp_values == 1
        warp_values = ones(1)
        warp_log_proportions = zeros(1)
    else
        # Log-spaced values between 1/τ_max and τ_max
        warp_values = τ_max .^ range(-1, 1, length=num_warp_values)
        # Set a mean-zero Gaussian prior on the log warp values
        warp_log_proportions = -0.5 / warp_variance * range(-1, 1, length=num_warp_values) .^ 2
        warp_log_proportions .-= logsumexp(warp_log_proportions)
    end
    warp_values, warp_log_proportions
end


"""User-Friendly Constructor of SeqModel model."""
function SeqModel(
        # constants
        max_time::Float64,
        max_sequence_length::Float64,

        # warp parameters
        num_warp_values::Int64,
        max_warp::Float64,
        warp_variance::Float64,

        # priors
        seq_event_rate::Float64,
        seq_type_proportions::SymmetricDirichlet,
        seq_event_amplitude::RateGamma,
        neuron_response_proportions::SymmetricDirichlet,
        neuron_response_profile::NormalInvChisq,
        bkgd_amplitude::RateGamma,
        bkgd_proportions::SymmetricDirichlet
    )

    # Number of neurons (N), sequence types (R), and warp values (W).
    N = neuron_response_proportions.dim
    R = seq_type_proportions.dim
    W = num_warp_values

    # Specify discrete distribution over warp values.
    warp_values, warp_log_proportions =
        default_warps(num_warp_values, max_warp, warp_variance)
    @show warp_values, warp_log_proportions

    # Create prior distributions.
    priors = SeqPriors(
        seq_event_rate,
        seq_type_proportions,
        seq_event_amplitude,
        neuron_response_proportions,
        neuron_response_profile,
        bkgd_amplitude,
        bkgd_proportions,
        warp_values,
        warp_log_proportions
    )

    # Initialize global variables.
    globals = sample(priors)

    # Initialize sequence events.
    sequence_events = SeqEventList(R, W)

    # Compute probability of introducing a new cluster.
    α = seq_event_amplitude.α
    β = seq_event_amplitude.β
    λ = seq_event_rate

    # Compute probability of introducing a background spike.
    bkgd_log_prob = (
        log(globals.bkgd_amplitude)
        + log(max_time)
        + log(1 + β)
    )
    new_cluster_log_prob = (
        log(α)
        + log(λ)
        + log(max_time)
        + α * (log(β) - log(1 + β))
    )

    # Package it all together into the model object.
    return SeqModel(
        max_time,
        max_sequence_length,
        priors,
        globals,
        sequence_events,
        new_cluster_log_prob,
        bkgd_log_prob,
        zeros(R),      # _R_buffer
        zeros(R, W),   # _RW_buffer
        Float64[],     # _K_buffer
    )
end

"""
By default, assume no warping.  (Or equivalently, warp_values=[1].)
"""
function SeqModel(
        # constants
        max_time::Float64,
        max_sequence_length::Float64,

        # priors
        seq_event_rate::Float64,
        seq_type_proportions::SymmetricDirichlet,
        seq_event_amplitude::RateGamma,
        neuron_response_proportions::SymmetricDirichlet,
        neuron_response_profile::NormalInvChisq,
        bkgd_amplitude::RateGamma,
        bkgd_proportions::SymmetricDirichlet
    )

    SeqModel(max_time,
          max_sequence_length,
          1, # number of warps
          1.0, # max warp value
          1.0, # warp variance
          seq_event_rate,
          seq_type_proportions,
          seq_event_amplitude,
          neuron_response_proportions,
          neuron_response_profile,
          bkgd_amplitude,
          bkgd_proportions)
end
    
    
# ======================================================================== #
# ===                                                                  === #
# === Methods to draw samples from a model (or the prior distribution) === #
# ===                                                                  === #
# ======================================================================== #

function sample(priors::SeqPriors)

    N = priors.neuron_response_proportions.dim
    R = priors.seq_type_proportions.dim

    # Initialize length-R probability vector over sequence types.
    seq_type_log_proportions = log.(rand(priors.seq_type_proportions))

    # Draw N x R matrix holding neuron response amplitudes.
    neuron_response_log_proportions = 
        log.(rand(priors.neuron_response_proportions, R))

    # Draw N x R matrix holding neuron response offsets and widths.
    neuron_response_offsets = zeros(N, R)
    neuron_response_widths = zeros(N, R)
    for n = 1:N
        for r = 1:R
            μ, σ2 = rand(priors.neuron_response_profile)
            neuron_response_offsets[n, r] = μ
            neuron_response_widths[n, r] = σ2
        end
    end

    # Draw background rate parameter.
    bkgd_amplitude = rand(priors.bkgd_amplitude)

    # Draw length-N probability vector holding normalized background firing rates.
    bkgd_log_proportions = log.(rand(priors.bkgd_proportions))

    return SeqGlobals(
        seq_type_log_proportions,
        neuron_response_log_proportions,
        neuron_response_offsets,
        neuron_response_widths,
        bkgd_amplitude,
        bkgd_log_proportions
    )
end


function sample(
        model::SeqModel;
        resample_latents::Bool=false,
        resample_globals::Bool=false
    )

    # =========== (i) SAMPLE GLOBAL PARAMS =========== #

    globals = 
        resample_globals ? sample(model.priors) : deepcopy(model.globals)

    # =========== (ii) SAMPLE LATENT EVENTS =========== #

    if resample_latents
        
        K = rand(Poisson(model.priors.seq_event_rate * model.max_time))

        seq_type_dist = Categorical(exp.(globals.seq_type_log_proportions))
        warp_dist = Categorical(exp.(model.priors.warp_log_proportions))
        events = EventSummaryInfo[]

        for k = 1:K
            τ = rand() * model.max_time
            r = rand(seq_type_dist)
            ω = model.priors.warp_values[rand(warp_dist)]
            A = rand(model.priors.seq_event_amplitude)
            push!(events, EventSummaryInfo(k, τ, r, ω, A))
        end

    else
        events = event_list_summary(model)
        # TODO -- need to sample empty latent events too...
    end
    
    # =========== (iii) SAMPLE SPIKES =========== #

    spikes = Spike[]
    assignments = Int64[]
  
    # Sample background spikes.
    S_bkgd = rand(Poisson(globals.bkgd_amplitude * model.max_time))
    bkgd_dist = Categorical(exp.(model.globals.bkgd_log_proportions))
    
    n_bkgd = rand(bkgd_dist, S_bkgd)
    t_bkgd = rand(S_bkgd) * model.max_time

    for (n, t) in zip(n_bkgd, t_bkgd)
        push!(spikes, Spike(n, t))
        push!(assignments, -1)
    end

    # Compute neuron probabilities.
    neuron_rel_amps =
        exp.(globals.neuron_response_log_proportions)

    # Sample sequence-evoked spikes.
    for e in events
        
        # Num spikes evoked by latent event.
        S = rand(Poisson(e.amplitude))

        # Neuron response proportions.
        nrn_dist = Categorical(neuron_rel_amps[:, e.seq_type])

        # Sample neuron, then spike time, for each spike.
        for n in rand(nrn_dist, S)
            
            # Sample spike time.
            μ = globals.neuron_response_offsets[n, e.seq_type]
            σ = sqrt(globals.neuron_response_widths[n, e.seq_type])
            t = e.seq_warp * (σ * randn() + μ) + e.timestamp

            # Exclude spikes outside of time window.
            if (0 < t < model.max_time)
                push!(spikes, Spike(n, t))
                push!(assignments, e.assignment_id)
            end

        end
    end

    return spikes, assignments, events

end

"""
Return firing (num_neurons x num_timebins) firing rate matrix.
"""
function firing_rates(
        globals::SeqGlobals,
        events::Vector{EventSummaryInfo},
        timebins::Vector{Float64};
        resample_latents::Bool=false,
        resample_globals::Bool=false
    )

    # Discretized time bins
    num_neurons = length(globals.bkgd_log_proportions)

    # Compute background intensities for each neuron.
    bkgd_firing_rates = globals.bkgd_amplitude .* exp.(globals.bkgd_log_proportions)

    # Initialize matrix of firing rates with background rates. 
    F = repeat(bkgd_firing_rates, outer=(1, length(timebins)))

    # Compute neuron probabilities.
    neuron_rel_amps =
        exp.(globals.neuron_response_log_proportions)

    # Sample sequence-evoked spikes.
    for e in events
        for n = 1:num_neurons
            μ = e.timestamp + globals.neuron_response_offsets[n, e.seq_type]
            σ = sqrt(globals.neuron_response_widths[n, e.seq_type])
            F[n, :] += e.amplitude * neuron_rel_amps[n] * normpdf.(μ, σ, timebins)
        end
    end

    return F

end


priors(model::SeqModel) = model.priors

function set_priors!(model::SeqModel, priors::SeqPriors)
    model.priors = priors
end

function set_globals!(model::SeqModel, globals::SeqGlobals)
    model.globals = globals
end

function set_new_cluster_log_prob!(model::SeqModel, prob::Float64)
    model.new_cluster_log_prob = prob
end

get_max_time(model::SeqModel) = model.max_time

get_bkgd_amplitude(model::SeqModel) = model.globals.bkgd_amplitude

function set_bkgd_log_prob!(model::SeqModel, prob::Float64)
    model.bkgd_log_prob = prob
end
