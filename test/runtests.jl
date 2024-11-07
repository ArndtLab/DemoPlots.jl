using DemoPlots
using Test
using DemoInfer, PyPlot, HistogramBinnings, StatsBase

@testset "DemoPlots.jl" begin
    h = Histogram(LogEdgeVector(lo = 30, hi = 1_000_000, nbins = 200));
    get_sim!([1_000_000_000, 10_000, 2_000, 2_000, 5_000, 10_000], h, 2.36e-8, 1e-8)
    res = sequential_fit(h, 2.36e-8, 4)

    @test isa(plot_hist(h, s=4, c="blue"), PyPlot.PyObject)

    @test isa(plot_residuals(h, res[end], 2.36e-8, 1e-8), PyPlot.PyObject)

    @test isa(plot_naive_residuals(h, res[end], 2.36e-8), PyPlot.PyObject)

    _, ax = subplots(figsize=(7, 5))
    @test isnothing(plot_demography(res, ax))

    _, ax = subplots(figsize=(7, 5))
    @test isnothing(plot_demography(res[end], ax))
end
