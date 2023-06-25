"""
Plots un-labeled spike raster. If permute_neurons == true, then randomly
permute the neuron labels before plotting.
"""
function plot_raster(spikes::Vector{Spike}; kwargs...)

    # Create figure and allocate space.
    fig = plt.figure()
    _x, _y = zeros(length(spikes)), zeros(length(spikes))

    # Plot raster and return.
    for (i, s) in enumerate(spikes)
        _x[i] = s.timestamp
        _y[i] = s.neuron
    end
    (length(fig.axes) < 1) && fig.add_subplot(1, 1, 1)
    fig.axes[1].scatter(_x, _y; s=4, kwargs...)
    fig.axes[1].set_ylabel("neurons")
    fig.axes[1].set_xlabel("time (s)")
    return fig
end


"""
Plots model-labled spike raster.
"""
function plot_raster(
        spikes::Vector{Spike},
        events::Vector{EventSummaryInfo},
        spike_assignments::Vector{Int64},
        neuron_order::Vector{Int64};
        color_cycle=["#E41A1C",
                     "#377EB8",
                     "#4DAF4A",
                     "#984EA3",
                     "#FF7F00",
                     "#FFFF33",
                     "#A65628",
                     "#F781BF"],
        kwargs...
    )

    fig = plt.figure()
    _x, _y = zeros(length(spikes)), zeros(length(spikes))
    _c = String[]

    typemap = Dict((e.assignment_id => e.seq_type) for e in events)
    yidx = sortperm(neuron_order)

    for (i, s) in enumerate(spikes)
        _x[i] = s.timestamp
        _y[i] = yidx[s.neuron]

        if spike_assignments[i] == -1
            push!(_c, "k")
        else
            k = typemap[spike_assignments[i]]
            col_ind = 1 + ((k - 1) % length(color_cycle))
            push!(_c, color_cycle[col_ind])
        end
    end

    (length(fig.axes) < 1) && fig.add_subplot(1, 1, 1)
    fig.axes[1].scatter(_x, _y; c=_c, s=4, kwargs...)
    fig.axes[1].set_ylabel("neurons")
    fig.axes[1].set_xlabel("time (s)")
    return fig
end


function plot_log_likes(config::Dict, results::Dict)
    
    # x-axis coordinates for annealing and post-annealing epochs.
    x1, x2 = _get_mcmc_x_coords(config, results)

    # Log-likelihoods for annealing and post-annealing epochs.
    if config[:are_we_masking] == 1
        y1 = results[:anneal_test_log_p_hist]
        y2 = [results[:anneal_test_log_p_hist][end]; results[:test_log_p_hist]]
        
        y3 = results[:anneal_train_log_p_hist]
        y4 = [results[:anneal_train_log_p_hist][end]; results[:train_log_p_hist]]
    else
        y1 = results[:anneal_log_p_hist]
        y2 = [results[:anneal_log_p_hist][end]; results[:log_p_hist]]
    end

    # Create figure
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)

    # Plot log-likelihood over MCMC samples.
    if config[:are_we_masking] == 1
        fig.axes[1].plot(x1,y1; label="test anneal")
        fig.axes[1].plot(x2,y2; label="test after anneal")
        
        fig.axes[1].plot(x1,y3; label="train anneal")
        fig.axes[1].plot(x2,y4; label="trian after anneal")
    else
        fig.axes[1].plot(x1, y1; label="anneal")
        fig.axes[1].plot(x2, y2; label="after anneal")
    end
    
    # Label axes, add legend.
    fig.axes[1].set_ylabel("log-likelihood")
    fig.axes[1].set_xlabel("MCMC samples")
    fig.axes[1].legend()

    return fig
end


function plot_num_seq_events(config::Dict, results::Dict)

    # x-axis coordinates for annealing and post-annealing epochs.
    x1, x2 = _get_mcmc_x_coords(config, results)

    # Number of sequence events (K) during annealing and post-annealing epochs.
    y1 = [length(ev) for ev in results[:anneal_latent_event_hist]]
    y2 = [y1[end]; [length(ev) for ev in results[:latent_event_hist]]]

    # Create figure
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)

    # Plot number of sequence occurences over MCMC samples.
    fig.axes[1].plot(x1, y1; label="anneal")
    fig.axes[1].plot(x2, y2; label="after anneal")

    # Label axes, add legend.
    fig.axes[1].set_ylabel("Number of Sequence Events")
    fig.axes[1].set_xlabel("MCMC samples")
    fig.axes[1].legend()

    return fig

end


function _get_mcmc_x_coords(config::Dict, results::Dict)

    if config[:are_we_masking] == 1
        # x-axis coordinates for annealing epoch
        s1 = config[:save_every_during_anneal]
        e1 = (length(results[:anneal_train_log_p_hist])*s1)
        x1 = collect(s1:s1:e1)

        # x-axis coordinates for post-anneal epoch.
        s2 = config[:save_every_after_anneal]
        e2 = e1 + (length(results[:train_log_p_hist]) * s2)
        x2 = collect(e1:s2:e2)
    else
        # x-axis coordinates for annealing epoch
        s1 = config[:save_every_during_anneal]
        e1 = (length(results[:anneal_log_p_hist]) * s1)
        x1 = collect(s1:s1:e1)

        # x-axis coordinates for post-anneal epoch.
        s2 = config[:save_every_after_anneal]
        e2 = e1 + (length(results[:log_p_hist]) * s2)
        x2 = collect(e1:s2:e2)
    end

    return x1, x2
end