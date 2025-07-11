module DemoPlots

using PyPlot
using StatsBase, HistogramBinnings
using DemoInfer, MLDs
using PyCall

export plot_demography, plot_remnbps,
    plot_lineages, plot_cumulative_lineages,
    plot_hist,
    plot_residuals_sim, plot_residuals_th,
    plot_chain,
    plot_results,
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
    plot_remnbps(para::Vector, ax; max_t = 1e7, g = 29, kwargs...)

Plot the remaining number of base pairs as a function of time, given the parameters `para`.

`ax` is a pyplot axis.

# Arguments
- `max_t = 1e7`: the furthest time, in generations, at which the coalescent is evaluated
- `g = 29`: arbitrary scaling factor for a generation
Further optional arguments are passed to `plot` and `scatter` from pyplot.
"""
function plot_remnbps(para::Vector, ax; max_t = 1e7, g = 29, kwargs...)
    x_ = 1:max_t
    y_ = map(x->MLDs.CoalescentBase.extbps(x, para), x_)
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
    plot_lineages(para::Vector, ax, rho; max_t = 1e7, g = 29, k = 0, kwargs...)

Plot the number of coalescing lineages as a function of time, given the parameters `para`.

`ax` is a pyplot axis, `rho` is the recombination rate per bp per generation.

# Arguments
- `max_t = 1e7`: the furthest time, in generations, at which the coalescent is evaluated
- `g = 29`: arbitrary scaling factor for a generation
- `k = 0`: the minimum length of associated IBD segment in bp
Further optional arguments are passed to `plot` and `scatter` from pyplot.
"""
function plot_lineages(para::Vector, ax, rho; max_t = 1e7, g = 29, k = 0, kwargs...)
    x_ = 1:max_t
    y_ = map(x->MLDs.CoalescentBase.lineages(x, rho, para; k = k), x_)
    stop = findlast(y_ .> 0.5)
    if isnothing(stop)
        stop = length(x_)
        @warn "at time $max_t the number of lineages is still not zero"
    end
    ax.plot(g*x_[1:stop], y_[1:stop]; kwargs...)
    ax.scatter(g*x_[stop], y_[stop]; kwargs...)
    return nothing
end

"""
    plot_cumulative_lineages(para::Vector, ax, rho; max_t = 1e7, g = 29, k = 0, kwargs...)

Plot the cumulative number of lineages coalescing within each epoch as a function of time, 
given the parameters `para`.

`ax` is a pyplot axis, `rho` is the recombination rate per bp per generation.

# Arguments
- `max_t = 1e7`: the furthest time, in generations, at which the coalescent is evaluated
- `g = 29`: arbitrary scaling factor for a generation
- `k = 0`: the minimum length of associated IBD segment in bp
Further optional arguments are passed to `plot` and `scatter` from pyplot.
"""
function plot_cumulative_lineages(para::Vector, ax, rho; max_t = 1e7, g = 29, k = 0, kwargs...)
    x_ = cumsum(para[end-1:-2:3])
    x_ = [0; x_; max_t]
    y_ = map(x->MLDs.CoalescentBase.cumulative_lineages(x, para, rho; k = k), x_)
    y_ = y_[2:end] .- y_[1:end-1]
    if MLDs.CoalescentBase.lineages(max_t, rho, para; k = k) > 0.5
        @warn "at time $max_t the number of lineages is still not zero"
    end
    ax.step(g*x_, [y_;y_[end]]; where = "post", kwargs...)
    return nothing
end

"""
    plot_demography(para::Vector{T}, stderrors::Vector{T} ax; kwargs...)
    plot_demography(fit::DemoInfer.FitResult, ax; kwargs...)

Plot the demographic profile encoded in the parameters inferred by the fit.

`ax` is the pyplot ax where to plot the demographic profile.

# Arguments
- `max_t = 1e7`: the furthest time to plot
- `g = 29`: arbitrary scaling factor for a generation
Further optional arguments are passed to `plot` from pyplot.
"""
function plot_demography(para::Vector{T}, stderrors::Vector{T}, ax;
    max_t = 1e7, g = 29, shift::Float64 = 0., eshift::Float64 = 0., 
    color="tab:red", alpha = 1, linewidth = 1,
    kwargs...
) where {T <: Number}
    
    nepochs = length(para)÷2
    para = para[end:-1:2]
    stderrors = stderrors[end:-1:2]
    stderrors = map((x,y)->min(x,y), para, stderrors)
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

    mean_epochs = [shift]
    upp_epochs = [shift-eshift]
    low_epochs = [shift-eshift]
    for i in 1:nepochs-1
        t = sum(para[2:2:end-1][1:i]) + shift
        st = sqrt(sum(stderrors[2:2:end-1][1:i] .^2) + eshift^2)
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
    max_t = 1e7, g = 29, shift::Float64 = 0., eshift::Float64 = 0.,
    color="tab:red", alpha = 1, linewidth = 1, kwargs...
)
    plot_demography(get_para(fit), vec(sds(fit)), ax; 
        max_t, g, shift, eshift, color, alpha, linewidth, kwargs...
    )
    return nothing
end

"""
    plot_input(TN, ax; max_t = 1e7, g = 29, kwargs...)

Plot the demographic profile encoded in the parameters `TN` as input.

# Arguments
- `max_t`: the furthest time to plot
- `g = 29`: arbitrary scaling factor for a generation
- `kwargs...`: the keywords that PyPlot `plot` accepts
"""
function plot_input(TN, ax; max_t = 1e7, g = 29, kwargs...)
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

function plot_results(segments, results::Vector{DemoInfer.FitResult}; 
    mu = 2.36e-8, rho = 1e-8, g = 29, 
    clin = "black", cl = "tab:red", evdcl = "navy", othercl = "green",
    Nlow = 1e2, Nhigh = 1e5, xres_l = 30,
    mld_fc = 10, res_fc = 10,
    kwargs...
)
    nfits = length(results)
    height = 3*(nfits+2)
    fig = figure(figsize=(15, height))
    # ax = fig.subplots(nfits+2, 4; width_ratios=[4, 4, 4, 1])
    gs0 = matplotlib.gridspec.GridSpec(3, 1; figure = fig, height_ratios=[0.8, nfits*0.75, 1], hspace = 0.1)
    gs11 = gs0[1,1].subgridspec(1, 3)
    gs2 = gs0[2,1].subgridspec(1, 3; wspace = 0.23, width_ratios=[1, 1, 1])
    gs21 = gs2[1,1].subgridspec(nfits, 1; hspace = 0.33)
    gs22 = gs2[1,2].subgridspec(nfits, 1; hspace = 0.33)
    gs23 = gs2[1,3].subgridspec(nfits, 5; hspace = 0.33, wspace = 0)
    gs31 = gs0[3,1].subgridspec(3, 11; wspace = 0.4)

    # plot input data
    h = HistogramBinnings.Histogram(LogEdgeVector(lo = 1, hi = 1_000_000, nbins = 200))
    append!(h, segments)
    x, ŷ = xy(h)
    ax = fig.add_subplot(gs11[1,2])
    ax.scatter(x,ŷ; s=3, color=clin, label="observed") #[1,2]
    ax.set_xlim(1, 1e6)
    ax.set_ylim(1e-5, 1e5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("segment length")
    ax.set_ylabel("density")
    ax.legend()
    # ax[1,1].axis("off")
    # ax[1,3].axis("off")
    # ax[1,4].axis("off")

    h_ = HistogramBinnings.Histogram(LogEdgeVector(lo = xres_l, hi = 1_000_000, nbins = 200))
    append!(h_, segments)

    for i in 1:nfits
        # plot demography
        pars = get_para(results[i])
        stds = sds(results[i])
        ax = fig.add_subplot(gs21[i])
        plot_demography(pars, stds, ax; g, color=cl, label="fit $(length(pars)÷2) epochs")
        ax.set_xlim(g, 120e6)
        ax.set_ylim(Nlow, Nhigh)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("years in the past")
        ax.set_ylabel(L"N(t)")
        ax.legend(loc = "lower right")
        # plot mld
        h_sim = HistogramBinnings.Histogram(h.edges)
        get_sim!(pars, h_sim, mu, rho; factor = mld_fc)
        x, y_th = xy(h_sim)
        ax = fig.add_subplot(gs22[i])
        ax.scatter(x, ŷ; s=3, color=clin, label="observed")
        ax.plot(x, y_th/mld_fc; color=cl, label="fit $(length(pars)÷2) epochs", linewidth = 1)
        ax.set_xlim(1, 1e6)
        ax.set_ylim(1e-5, 1e5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("segment length")
        ax.set_ylabel("density")
        ax.legend()
        # plot residuals
        ax = fig.add_subplot(py"$(gs23)[$(i-1),0:4]")
        # resf = plot_residuals_sim(h_, pars, mu, rho, ax; s = 5, color=cl, factor = res_fc)
        wo = results[i].opt.h_obs.weights
        wf = max.(0,results[i].opt.corrected_weights)
        resf = (wo .- wf) ./ sqrt.(wo .+ wf)
        scatter(midpoints(h_.edges[1]), resf; s=5, color=cl)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xscale("log")
        ax.set_xlim(xres_l, 1e6)
        ax.set_xlabel("segment length")
        ax.set_ylabel("residuals")
        ylimb = ax.get_ylim()
        ax = fig.add_subplot(gs23[i,5])
        ax.hist(resf, bins=20, color=cl, alpha=0.5, orientation = "horizontal")
        ax.set_ylim(ylimb)
        ax.set_yticks([])
        ax.axis("off")
    end

    evidences = evd.(results)
    maxev = maximum(evidences)
    lls = map(x->x.lp, results)
    maxll = maximum(lls)

    pars = get_para(results[argmax(evidences)])
    stds = sds(results[argmax(evidences)])

    ax = fig.add_subplot(py"$(gs31)[0,1:5]")
    plot_remnbps(pars, ax; g, color = othercl)
    ax.set_xlim(g, 120e6)
    ax.set_xscale("log")
    ax.set_xticks([])
    ax.set_ylabel("uncoalesced\nbasepairs")
    ax.tick_params(axis="y", labelcolor = othercl)
    ax_lin = ax.twinx()
    plot_lineages(pars, ax_lin, rho; g, color = evdcl)
    ax_lin.set_ylabel("coalescing\nlineages")
    ax_lin.tick_params(axis="y", labelcolor = evdcl)

    ax = fig.add_subplot(py"$(gs31)[1:,1:5]")
    plot_demography(pars, stds, ax; g, color = evdcl, label = "fit $(length(pars)÷2) epochs")
    ax.set_xlim(g, 120e6)
    ax.set_ylim(Nlow, Nhigh)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("years in the past")
    ax.set_ylabel(L"N(t)")
    ax.legend(loc = "lower right")

    ax = fig.add_subplot(py"$(gs31)[:,6:10]")
    ax.plot(1:nfits, evidences .- maxev .- 1; color = evdcl, label = "\$f(n) =\$ log-evidence")
    # ax.tick_params(axis="y", labelcolor = evdcl)
    ax.set_xlabel("number of epochs")
    ax.set_ylabel(L"f(n) - \max \,f")
    ax.set_yscale("symlog")
    # ax.legend(loc = "upper left")
    # axll = ax.twinx()
    ax.plot(1:nfits, lls .- maxll .- 1; color = othercl, label = "\$f(n) =\$ log-likelihood")
    # axll.tick_params(axis="y", labelcolor = othercl)
    # axll.set_ylabel("log-likelihood", color = othercl)
    # axll.set_yscale("symlog")
    ax.legend(loc = "lower right")

    close()

    return fig
end

end
