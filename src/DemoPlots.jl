module DemoPlots

using PyPlot
using StatsBase, HistogramBinnings
using DemoInfer

export plot_demography,
    plot_hist,
    plot_residuals_sim, plot_residuals_th,
    plot_chain,
    xy, plot_input

"""
    xy(h::HistogramBinnings.Histogram{T, 1, E}; mode = :density) where {T, E}

Return the midpoints and the weights of the histogram `h`.

# Arguments
- `h`: the histogram
- `mode`: the normalization mode, `:density` normalizes counts by bins widths
"""
function xy(h::HistogramBinnings.Histogram{T, 1, E}; mode = :density) where {T, E}
    hn = StatsBase.normalize(h; mode)
    return midpoints(h.edges[1]), hn.weights
end

"""
    plot_demography(para::Vector{T}, stderrors::Vector{T} ax; kwargs...)
    plot_demography(fit::DemoInfer.FitResult, ax; kwargs...)

Plot the demographic profile encoded in the parameters inferred by the fit.

`ax` is the pyplot ax where to plot the demographic profile.

# Arguments
- `max_t = 1e6`: the furthest time to plot
- `color = "tab:red"`, `alpha = 1`, `linewidth = 1`, `kwargs...`: the keywords that PyPlot `plot` accepts
"""
function plot_demography(para::Vector{T}, stderrors::Vector{T}, ax;
    max_t = 1e6, color="tab:red", alpha = 1, linewidth = 1, 
    kwargs...
) where {T <: Number}
    
    nepochs = length(para)÷2
    para = para[end:-1:2]
    stderrors = stderrors[end:-1:2]
    old_t = max(sum(para[2:2:end-1])+1e4, max_t)
    
    Polygon = matplotlib.patches.Polygon
    mean_size = []
    upp_size = []
    low_size = []
    for (n,sn) in zip(para[1:2:end], stderrors[1:2:end])
        append!(mean_size, [n,n])
        append!(upp_size, [n+sn,n+sn])
        append!(low_size, [n-sn,n-sn])
    end

    mean_epochs = [0.]
    upp_epochs = [0.]
    low_epochs = [0.]
    for i in 1:nepochs-1
        t = sum(para[2:2:end-1][1:i])
        st = sqrt(sum(stderrors[2:2:end-1][1:i] .^2))
        append!(mean_epochs, [t,t])
        if (para[1:2:end][i] + stderrors[1:2:end][i]) > (para[1:2:end][i+1] + stderrors[1:2:end][i+1])
            append!(upp_epochs, [t+st,t+st])
        else
            append!(upp_epochs, [t-st,t-st])
        end
        if (para[1:2:end][i] - stderrors[1:2:end][i]) < (para[1:2:end][i+1] - stderrors[1:2:end][i+1])
            append!(low_epochs, [t+st,t+st])
        else
            append!(low_epochs, [t-st,t-st])
        end
    end
    push!(mean_epochs, old_t)
    push!(upp_epochs, old_t)
    push!(low_epochs, old_t)

    err = Polygon(collect(zip([upp_epochs;low_epochs[end:-1:1]],[upp_size;low_size[end:-1:1]])),facecolor=color, edgecolor="none",alpha=0.5*alpha)

    ax.plot(mean_epochs, mean_size; color = color, alpha=alpha, linewidth = linewidth, kwargs...)
    ax.add_patch(err)
    return nothing
end

function plot_demography(fit::DemoInfer.FitResult, ax;
    max_t = 1e6, color="tab:red", alpha = 1, linewidth = 1, kwargs...
)
    plot_demography(get_para(fit), vec(sds(fit)), ax; max_t, color, alpha, linewidth, kwargs...)
    return nothing
end

"""
    plot_input(TN, ax; max_t = 1e6, kwargs...)

Plot the demographic profile encoded in the parameters `TN` as input.

# Arguments
- `max_t`: the furthest time to plot
- `kwargs...`: the keywords that PyPlot `plot` accepts
"""
function plot_input(TN, ax; max_t = 1e6, kwargs...)
    if length(TN) > 2
        Ns = reverse(TN[2:2:end])
        Ts = cumsum(reverse(TN[3:2:end]))
        Ts = [0, Ts...]
        x_ = []
        y_ = []
        for i in eachindex(Ns[1:end-1])
            append!(x_, [Ts[i], Ts[i+1], Ts[i+1]])
            append!(y_, [Ns[i], Ns[i], Ns[i+1]])
        end
        append!(x_, [Ts[end], max_t])
        append!(y_, [Ns[end], Ns[end]])
        ax.plot(x_, y_; kwargs...)
    else
        ax.plot([0, max_t], [TN[2], TN[2]]; kwargs...)
    end
end

"""
    plot_hist(h::Histogram; kwargs...)

Plot the histogram `h` using PyPlot.

Optional arguments are passed to `scatter` from pyplot.
"""
function plot_hist(h::HistogramBinnings.Histogram{T, 1, E}; kwargs...) where {T, E}
    x, y = xy(h)
    scatter(x, y; kwargs...)
end

"""
    plot_residuals_sim(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64, ρ::Float64; kwargs...)
    plot_residuals_sim(h_obs::Histogram, para::Vector{T}, μ::Float64, ρ::Float64; kwargs...)

Plot the residuals of the simulation, with given `fit` result` or `para` as input, 
with respect to the observed histogram `h_obs`.

Optional arguments are passed to `scatter` from pyplot.
"""
function plot_residuals_sim(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64, ρ::Float64; kwargs...)
    if any(get_para(fit) .<= 0)
        plot_residuals_th(h_obs, fit, μ; kwargs...)
    else
        plot_residuals_sim(h_obs, get_para(fit), μ, ρ; kwargs...)
    end
end

function plot_residuals_sim(h_obs::Histogram, para::Vector{T}, μ::Float64, ρ::Float64; kwargs...) where {T <: Number}
    h_sim = HistogramBinnings.Histogram(h_obs.edges)
    DemoInfer.get_sim!(para, h_sim, μ, ρ, factor=1)
    residuals = (h_obs.weights .- h_sim.weights) ./ sqrt.(h_obs.weights .+ h_sim.weights)
    x = midpoints(h_obs.edges[1])
    mask = (h_obs.weights .> 0) .& (h_sim.weights .> 0)
    x_ = x[mask .& (x.>1e0)]
    y_ = residuals[mask .& (x.>1e0)]
    scatter(x_, y_; kwargs...)
end

"""
    plot_residuals_th(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64; kwargs...)
    plot_residuals_th(h_obs::Histogram, para::Vector{T}, μ::Float64; kwargs...)

Plot of residuals between observed histogram `h_obs` and the theory.

Optional arguments are passed to `scatter` from pyplot.
"""
function plot_residuals_th(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64; kwargs...)
    plot_residuals_th(h_obs, get_para(fit), μ; kwargs...)
end

function plot_residuals_th(h_obs::Histogram, para::Vector{T}, μ::Float64; kwargs...) where {T <: Number}
    weights_th = DemoInfer.integral_ws(h_obs.edges[1].edges, μ, para)
    residuals = (h_obs.weights .- weights_th) ./ sqrt.(h_obs.weights)
    x, y = xy(h_obs) 
    x_ = x[(y .!= 0).&(x.>1e0)]
    y_ = residuals[(y .!= 0).&(x.>1e0)]
    scatter(x_, y_; kwargs...)
end

"""
    plot_chain(fit::DemoInfer.FitResult, n::Int, ax; kwargs...)

Plot the chain stored in `fit` for parameter `n`-th.

`ax` is the pyplot ax where to plot the chain.
Optional arguments are passed to `plot` from pyplot.
"""
function plot_chain(fit::DemoInfer.FitResult, n::Int, ax; kwargs...)
    p, sd = get_chain(fit)
    values = p[n, :]
    stds = sd[n, :]
    ax.errorbar(eachindex(values), values, yerr=stds; kwargs...)
end

end
