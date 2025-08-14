using DemoPlots
using Test
using DemoInfer, PyPlot, HistogramBinnings, StatsBase

@testset "DemoPlots.jl" begin
    h = Histogram(LogEdgeVector(lo = 30, hi = 1_000_000, nbins = 200));
    TN = [1_000_000_000, 10_000, 2_000, 2_000, 5_000, 10_000]
    mu = 2.36e-8
    rho = 1e-8
    get_sim!(TN, h, mu, rho)
    res = demoinfer(h, 3, mu, rho, 1_000_000_000, 1.0TN; iters = 1)

    _, ax = subplots(figsize=(7, 5))

    @test isa(plot_hist(h, s=4, c="blue"), PyPlot.PyObject)
    @test !isnothing(plot_residuals_sim(h, res, mu, rho, ax))
    @test isnothing(plot_residuals_th(h, res, mu, ax))
    @test isnothing(plot_demography(res, ax))
    plot_lineages(get_para(res), ax, rho)
    plot_cumulative_lineages(get_para(res), ax, rho)
    plot_results(rand(100), [res])
end
