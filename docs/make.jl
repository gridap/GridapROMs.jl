using Documenter
using GridapROMs

fem_interface = [
  "utils.md",
  "dof_maps.md",
  "tproduct.md",
  "param_data_structures.md",
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
    format=Documenter.HTML(size_threshold=nothing),
    pages=[
        "Home" => "index.md",
        "Usage" => ["steady.md","transient.md"],
        "FEM Interface" => fem_interface,
        "ROM Interface" => rom_interface,
        "Distributed Interface" => distributed_interface,
        "Contributing" => "contributing.md",
    ],
    sitename="GridapROMs.jl",
    warnonly=[:cross_references,:missing_docs],
)

deploydocs(
  repo = "github.com:gridap/GridapROMs.jl.git",
)
