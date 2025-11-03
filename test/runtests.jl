using DemoPlots
using Test
using DemoPlots.HetDister: FitResult, get_para
using DemoPlots.PyPlot
using DemoPlots.HistogramBinnings
using DemoPlots.StatsBase

@testset "DemoPlots.jl" begin
    h = Histogram(LogEdgeVector(lo = 30, hi = 1_000_000, nbins = 200));
    TN = [1_000_000_000, 10_000, 2_000, 2_000, 5_000, 10_000]
    mu = 2.36e-8
    rho = 1e-8

    f = FitResult(
        3,
        length(h.weights),
        mu,
        rho,
        TN,
        TN ./ 10,
        "test",
        false,
        1,
        1,
        (; h_obs = h)
    )

    _, ax = subplots(figsize=(7, 5))

    xy(h)
    plot_remnbps(get_para(f), ax)
    plot_lineages(get_para(f), ax, rho)
    plot_cumulative_lineages(get_para(f), ax, rho)
    plot_demography(f, ax)
    plot_input(TN, ax)
    plot_results(rand(100), [f])
    @test true
end
