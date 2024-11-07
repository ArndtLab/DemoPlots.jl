using DemoPlots
using Documenter

DocMeta.setdocmeta!(DemoPlots, :DocTestSetup, :(using DemoPlots); recursive=true)

makedocs(;
    modules=[DemoPlots],
    authors="Tommaso Stentella <stentell@molgen.mpg.de> and contributors",
    sitename="DemoPlots.jl",
    format=Documenter.HTML(;
        canonical="https://ArndtLab.github.io/DemoPlots.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/ArndtLab/DemoPlots.jl",
    devbranch="main",
)
