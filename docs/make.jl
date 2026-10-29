using Documenter
using Literate
using GridapROMs

ENV["GKSwstype"] = "100" # headless plotting

# tutorials: Literate sources in docs/tutorials, generated into docs/src/tutorials

tutorials_src = joinpath(@__DIR__,"tutorials")
tutorials_out = joinpath(@__DIR__,"src","tutorials")

tutorials = [
  "t001_parameterised_pdes.jl",
  "t002_reduced_order_models.jl",
  "t003_local_roms.jl",
  "t004_distributed.jl",
  ]

for t in tutorials
  Literate.markdown(joinpath(tutorials_src,t),tutorials_out;documenter=true,credit=false)
end

tutorial_pages = map(t -> joinpath("tutorials",replace(t,".jl" => ".md")),tutorials)

fem_interface = [
  "utils.md",
  "dof_maps.md",
  "tproduct.md",
  "param_data_structures.md",
  "param_algebra.md",
  "param_geometry.md",
  "param_fe_spaces.md",
  "param_steady.md",
  "param_odes.md",
  "extensions.md"
  ]

rom_interface = [
  "rbsteady.md",
  "rbtransient.md",
]

distributed_interface = [
  "distributed.md",
]

makedocs(;
  modules=[GridapROMs],
  format=Documenter.HTML(size_threshold=nothing,example_size_threshold=nothing),
  pages=[
    "Home" => "index.md",
    "Tutorials" => tutorial_pages,
    "Reference" => [
      "FEM" => fem_interface,
      "ROM" => rom_interface,
      "Distributed" => distributed_interface,
    ],
    "Contributing" => "contributing.md",
  ],
  sitename="GridapROMs.jl",
  warnonly=[:cross_references,:missing_docs],
)

deploydocs(
  repo = "github.com:gridap/GridapROMs.jl.git",
)
