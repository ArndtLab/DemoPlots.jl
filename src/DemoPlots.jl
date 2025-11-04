module DemoPlots

using PyPlot
using StatsBase, HistogramBinnings
using HetDister
using HetDister: FitResult, get_para, sds, compute_residuals, mldsmcp
using HetDister.Spectra.CoalescentBase: 
    extbps, lineages, cumulative_lineages, getts, getns
using PyCall

export plot_demography, plot_remnbps,
    plot_lineages, plot_cumulative_lineages,
    plot_hist,
    plot_residuals_sim, plot_residuals_th, plot_residuals,
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
    y_ = map(x->extbps(x, para), x_)
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
    y_ = map(x->lineages(x, para, rho; k = k), x_)
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
    y_ = map(x->cumulative_lineages(x, para, rho; k = k), x_)
    y_ = y_[2:end] .- y_[1:end-1]
    if lineages(max_t, para, rho; k = k) > 0.5
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
- `rho = 1e-8`: recombination rate per bp per generation
- `shift::Float64 = 0.`: shift in generations to apply to the epoch
- `eshift::Float64 = 0.`: additional error to add in quadrature to the 
  standard errors of the epochs
- `color="tab:red"`: color of the line
- `alpha = 1`: transparency of the line
- `alphapatch = 0.5*alpha`: transparency of the confidence interval patch
- `linewidth = 1`: line width
- `endcoalwidth = 1`: line width of the vertical lines indicating
  when expected coalescing lineages per gen drop below 0.5
  and the number of uncoalesced base pairs drop below 1
- `plotendcoal = false`: if true the above vertical lines are plotted
Further optional arguments are passed to `plot` from pyplot.
"""
function plot_demography(para::Vector, stderrors::Vector, ax;
    max_t = 1e7, g = 29, rho = 1e-8, shift::Float64 = 0., eshift::Float64 = 0., 
    color="tab:red", alpha = 1, alphapatch = 0.5*alpha, linewidth = 1,
    endcoalwidth = 1, plotendcoal = false,
    kwargs...
)   
    nepochs = length(para)÷2
    vars = stderrors .^ 2

    old_t = max(2*getts(para, nepochs), max_t)

    endline = 0
    for i in 1:10:max_t
        if lineages(i, para, rho; k = 0) > 0.5
            endline = i
        end
    end

    endbp = max_t
    for i in 1:10:max_t
        if extbps(i, para) < 1
            endbp = i
            break
        end
    end

    mean_size = []
    upp_size = []
    low_size = []
    mean_epochs = []
    upp_epochs = []
    low_epochs = []
    t(i) = getts(para, i) + shift
    n(i) = getns(para, i)
    st(i) = sqrt(getts(vars, i) + eshift^2)
    sn(i) = sqrt(getns(vars, i))
    push!(mean_epochs, t(1))
    push!(upp_epochs, t(1) - st(1))
    push!(low_epochs, t(1) - st(1))
    push!(mean_size, n(1))
    push!(upp_size, n(1) + sn(1))
    push!(low_size, n(1) - sn(1))
    for i in 2:nepochs
        append!(mean_epochs, t(i), t(i))
        append!(mean_size, n(i-1), n(i))
        # step down or up in the upper confidence boundary
        if (n(i-1) + sn(i-1)) > (n(i) + sn(i))
            # cannot be larger than next the epoch itself
            if i == nepochs
                t_ = min(t(i) + st(i), old_t)
            else
                t_ = min(t(i) + st(i), t(i+1))
            end
            append!(upp_epochs, t_, t_)
        else
            # cannot be larger than the previous epoch itself
            t_ = max(t(i) - st(i), t(i-1))
            append!(upp_epochs, t_ , t_)
        end
        # step up or down in the lower confidence boundary
        if (n(i-1) - sn(i-1)) < (n(i) - sn(i))
            if i == nepochs
                t_ = min(t(i) + st(i), old_t)
            else
                t_ = min(t(i) + st(i), t(i+1))
            end
            append!(low_epochs, t_, t_)
        else
            t_ = max(t(i) - st(i), t(i-1))
            append!(low_epochs, t_, t_)
        end
        append!(upp_size, n(i-1) + sn(i-1), n(i) + sn(i))
        append!(low_size, n(i-1) - sn(i-1), n(i) - sn(i))
    end
    push!(mean_epochs, old_t)
    push!(mean_size, n(nepochs))
    push!(upp_epochs, old_t)
    push!(low_epochs, old_t)
    push!(upp_size, n(nepochs) + sn(nepochs))
    push!(low_size, n(nepochs) - sn(nepochs))
    
    Polygon = matplotlib.patches.Polygon
    err = Polygon(
        collect(zip(g*[upp_epochs;low_epochs[end:-1:1]],[upp_size;low_size[end:-1:1]])),
        facecolor=color, edgecolor="none",alpha=alphapatch
    )

    ax.plot(g*mean_epochs, mean_size; color = color, alpha=alpha, linewidth = linewidth, kwargs...)
    ax.add_patch(err)
    if plotendcoal
        ax.vlines(g*endline, 1, 1e9; color="white", linewidth=endcoalwidth)
        ax.vlines(g*endbp, 1, 1e9; color="white", linewidth=endcoalwidth)
    end
    return nothing
end

function plot_demography(fit::FitResult, ax;
    max_t = 1e7, g = 29, rho = 1e-8, shift::Float64 = 0., eshift::Float64 = 0.,
    color="tab:red", alpha = 1, alphapatch = 0.5*alpha, linewidth = 1,
    endcoalwidth = 1, plotendcoal = false, kwargs...
)
    plot_demography(get_para(fit), vec(sds(fit)), ax; 
        max_t, g, rho, shift, eshift, color, alpha, alphapatch, linewidth, endcoalwidth, plotendcoal, kwargs...
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

function plot_results(segments, results::Vector{FitResult}; 
    mu = 2.36e-8, rho = 1e-8, g = 29, 
    clin = "black", cl = "tab:red", evdcl = "navy", othercl = "green",
    Nlow = 1e2, Nhigh = 1e5, xres_l = 30,
    mld_fc = 10, res_fc = 10,
    order = 10, ndt = 800,
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
        # h_sim = HistogramBinnings.Histogram(h.edges)
        # get_sim!(pars, h_sim, mu, rho; factor = mld_fc)
        # x, y_th = xy(h_sim)
        y_th = mldsmcp(x, h.edges[1], mu, rho, pars; order, ndt)
        ax = fig.add_subplot(gs22[i])
        ax.scatter(x, ŷ; s=3, color=clin, label="observed")
        ax.plot(x, y_th; color=cl, label="fit $(length(pars)÷2) epochs", linewidth = 1)
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
        ho = results[i].opt.h_obs
        wo = ho.weights
        x = midpoints(ho.edges[1])
        wf = mldsmcp(x, ho.edges[1], mu, rho, pars; order, ndt) .* diff(ho.edges[1])
        resf = (wo .- wf) ./ sqrt.(wf)
        scatter(x, resf; s=5, color=cl)
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
