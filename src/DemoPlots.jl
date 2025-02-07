module DemoPlots

using PyPlot
using StatsBase, HistogramBinnings
using DemoInfer

export plot_demography, plot_remnbps,
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

function coalescent(t::Int, TN::Vector)
    ts = [0;cumsum(reverse(TN[3:2:end-1]))]
    ns = reverse(TN[2:2:end])
    pnt = 1
    c = 0.
    while (pnt < length(ts)) && (ts[pnt] < t)
        gens = ts[pnt+1] >= t ? (t - ts[pnt]) : (ts[pnt+1] - ts[pnt])
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    if ts[pnt] < t
        gens = t - ts[pnt]
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    return exp(-c) / 2ns[pnt-1]
end

function extbps(t::Float64, TN::Vector)
    L = Float64(TN[1])
    ts = [0;cumsum(reverse(TN[3:2:end-1]))]
    ns = reverse(TN[2:2:end])
    pnt = 1
    c = 0.
    while (pnt < length(ts)) && (ts[pnt] < t)
        gens = ts[pnt+1] >= t ? (t - ts[pnt]) : (ts[pnt+1] - ts[pnt])
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    if ts[pnt] < t
        gens = t - ts[pnt]
        N = ns[pnt]
        c += gens / 2N
        pnt += 1
    end
    return round(L*exp(-c))
end


"""
    plot_remnbps(para::Vector, ax; max_t = 5e6, g = 25, kwargs...)

Plot the remaining number of base pairs as a function of time, given the parameters `para`.

`ax` is a pyplot axis.

# Arguments
- `max_t = 5e6`: the furthest time, in generations, at which the coalescent is evaluated
- `g = 25`: arbitrary scaling factor for a generation
Further optional arguments are passed to `plot` and `scatter` from pyplot.
"""
function plot_remnbps(para::Vector, ax; max_t = 5e6, g = 25, kwargs...)
    x_ = 1:max_t
    y_ = map(x->DemoPlots.extbps(x, para), x_)
    stop = findfirst(y_ .== 0)
    if isnothing(stop)
        stop = length(x_)
        @warn "at time $max_t the remaining number of base pairs is still not zero"
    end
    ax.plot(g*x_[1:stop], y_[1:stop]; kwargs...)
    ax.scatter(g*x_[stop], y_[stop]; kwargs...)
    return nothing
end

"""
    plot_demography(para::Vector{T}, stderrors::Vector{T} ax; kwargs...)
    plot_demography(fit::DemoInfer.FitResult, ax; kwargs...)

Plot the demographic profile encoded in the parameters inferred by the fit.

`ax` is the pyplot ax where to plot the demographic profile.

# Arguments
- `max_t = 5e6`: the furthest time to plot
- `g = 25`: arbitrary scaling factor for a generation
Further optional arguments are passed to `plot` from pyplot.
"""
function plot_demography(para::Vector{T}, stderrors::Vector{T}, ax;
    max_t = 5e6, g = 25, color="tab:red", alpha = 1, linewidth = 1, 
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

    err = Polygon(
        collect(zip(g*[upp_epochs;low_epochs[end:-1:1]],[upp_size;low_size[end:-1:1]])),
        facecolor=color, edgecolor="none",alpha=0.5*alpha
    )

    ax.plot(g*mean_epochs, mean_size; color = color, alpha=alpha, linewidth = linewidth, kwargs...)
    ax.add_patch(err)
    return nothing
end

function plot_demography(fit::DemoInfer.FitResult, ax;
    max_t = 5e6, g = 25, color="tab:red", alpha = 1, linewidth = 1, kwargs...
)
    plot_demography(get_para(fit), vec(sds(fit)), ax; max_t, g, color, alpha, linewidth, kwargs...)
    return nothing
end

"""
    plot_input(TN, ax; max_t = 5e6, g = 25, kwargs...)

Plot the demographic profile encoded in the parameters `TN` as input.

# Arguments
- `max_t`: the furthest time to plot
- `g = 25`: arbitrary scaling factor for a generation
- `kwargs...`: the keywords that PyPlot `plot` accepts
"""
function plot_input(TN, ax; max_t = 5e6, g = 25, kwargs...)
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
        ax.plot(g*x_, y_; kwargs...)
    else
        ax.plot(g*[0, max_t], [TN[2], TN[2]]; kwargs...)
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

# Arguments
- `factor = 1`: the factor which determines how many times the simulation is repeated
Further optional arguments are passed to `scatter` from pyplot, `ax` is the pyplot axis.
"""
function plot_residuals_sim(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64, ρ::Float64, ax;
    factor = 1, kwargs...
)
    return plot_residuals_sim(h_obs, get_para(fit), μ, ρ, ax; factor, kwargs...)
end

function plot_residuals_sim(h_obs::Histogram, para::Vector{T}, μ::Float64, ρ::Float64, ax;
    factor = 1, kwargs...
) where {T <: Number}
    residuals = zeros(length(h_obs.weights))
    if all(para .> 0)
        h_sim = HistogramBinnings.Histogram(h_obs.edges)
        DemoInfer.get_sim!(para, h_sim, μ, ρ; factor)
        residuals = compute_residuals(h_obs, h_sim; fc2 = factor)
        x = midpoints(h_obs.edges[1])
        ax.scatter(x, residuals; kwargs...)
    end
    return residuals
end

"""
    plot_residuals_th(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64; kwargs...)
    plot_residuals_th(h_obs::Histogram, para::Vector{T}, μ::Float64; kwargs...)

Plot of residuals between observed histogram `h_obs` and the theory.

Optional arguments are passed to `scatter` from pyplot, `ax` is the pyplot axis.
"""
function plot_residuals_th(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64, ax; kwargs...)
    plot_residuals_th(h_obs, get_para(fit), μ, ax; kwargs...)
end

function plot_residuals_th(h_obs::Histogram, para::Vector{T}, μ::Float64, ax; kwargs...) where {T <: Number}
    residuals = compute_residuals(h_obs, μ, para)
    x = midpoints(h_obs.edges[1])
    ax.scatter(x, residuals; kwargs...)
    return nothing
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
