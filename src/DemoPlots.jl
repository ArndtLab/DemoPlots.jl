module DemoPlots

using PyPlot
using StatsBase, HistogramBinnings
using DemoInfer

export plot_demography, plot_hist, plot_residuals, plot_naive_residuals, xy, plot_input

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
    plot_demography(para::Vector{T}, stderrors::Vector{T} ax; max_t = 1e6, color="black", alpha = 1, linewidth = 1, kwargs...)
    plot_demography(fit::DemoInfer.FitResult, ax; max_t = 1e6, color="black", alpha = 1, linewidth = 1, kwargs...)

Plot the demographic profile encoded in the parameters inferred by the fit.

# Arguments
- `para` or `fit`: the fitted parameters (or the fit result) in the form of a TN vector as described in `DemoInfer.jl`
- `stderrors`: the standard errors of the parameters
- `ax`: the pyplot axis where to plot the demographic profile
- `max_t`: the furthest time to plot
- `color`, `alpha`, `linewidth`, `kwargs...`: the keywords that PyPlot `plot` accepts
"""
function plot_demography(para::Vector{T}, stderrors::Vector{T}, ax; max_t = 1e6, color="black", alpha = 1, linewidth = 1, kwargs...) where {T <: Number}
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
end

function plot_demography(fit::DemoInfer.FitResult, ax; max_t = 1e6, color="black", alpha = 1, linewidth = 1, kwargs...)
    plot_demography(fit.para, vec(fit.opt.stderrors), ax; max_t, color, alpha, linewidth, kwargs...)
end

"""
    plot_input(TN; max_t = 1e6, kwargs...)

Plot the demographic profile encoded in the parameters `TN` as input.

# Arguments
- `TN`: the demographic profile, in the form of a TN vector as described in `DemoInfer.jl`
- `max_t`: the furthest time to plot
- `kwargs...`: the keywords that PyPlot `plot` accepts
"""
function plot_input(TN; max_t = 1e6, kwargs...)
    if length(TN) > 2
        Ns = reverse(TN[2:2:end])
        Ts = cumsum(reverse(TN[3:2:end]))
        Ts = [0, Ts...]
        for i in eachindex(Ns[1:end-1])
            plot([Ts[i], Ts[i+1]], [Ns[i], Ns[i]]; kwargs...)
            plot([Ts[i+1], Ts[i+1]], [Ns[i], Ns[i+1]]; kwargs...)
        end
        plot([Ts[end], max_t], [Ns[end], Ns[end]]; kwargs...)
    else
        plot([0, max_t], [TN[2], TN[2]]; kwargs...)
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
    plot_residuals(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64, ρ::Float64; kwargs...)
    plot_residuals(h_obs::Histogram, para::Vector{T}, μ::Float64, ρ::Float64; kwargs...)

Plot the residuals of the simulation, with given `fit` result` or `para` as input, 
with respect to the observed histogram `h_obs`.

Optional arguments are passed to `scatter` from pyplot.
"""
function plot_residuals(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64, ρ::Float64; kwargs...)
    plot_residuals(h_obs, fit.para, μ, ρ; kwargs...)
end

function plot_residuals(h_obs::Histogram, para::Vector{T}, μ::Float64, ρ::Float64; kwargs...) where {T <: Number}
    h_sim = HistogramBinnings.Histogram(h_obs.edges)
    DemoInfer.get_sim!(para, h_sim, μ, ρ, factor=1)
    residuals = (h_obs.weights .- h_sim.weights) ./ sqrt.(h_obs.weights)
    x, y = xy(h_obs) 
    x_ = x[(y .!= 0).&(x.>1e0)]
    y_ = residuals[(y .!= 0).&(x.>1e0)]
    scatter(x_, y_; kwargs...)
end

"""
    plot_naive_residuals(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64; kwargs...)
    plot_naive_residuals(h_obs::Histogram, para::Vector{T}, μ::Float64; kwargs...)

Plot of residuals between observed histogram `h_obs` and the naive theory.

See `plot_residuals` for more details.
"""
function plot_naive_residuals(h_obs::Histogram, fit::DemoInfer.FitResult, μ::Float64; kwargs...)
    plot_naive_residuals(h_obs, fit.para, μ; kwargs...)
end

function plot_naive_residuals(h_obs::Histogram, para::Vector{T}, μ::Float64; kwargs...) where {T <: Number}
    weights_th = DemoInfer.integral_ws(h_obs.edges[1].edges, μ, para)
    residuals = (h_obs.weights .- weights_th) ./ sqrt.(h_obs.weights)
    x, y = xy(h_obs) 
    x_ = x[(y .!= 0).&(x.>1e0)]
    y_ = residuals[(y .!= 0).&(x.>1e0)]
    scatter(x_, y_; kwargs...)
end

end
