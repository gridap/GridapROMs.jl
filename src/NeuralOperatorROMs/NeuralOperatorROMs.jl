module NeuralOperatorROMs

using Gridap
using Gridap.FESpaces
using Gridap.Geometry
using Gridap.CellData
using Gridap.ReferenceFEs

using Lux
using Optimisers
using Zygote
using Random
using LinearAlgebra
using Statistics

include("Snapshots.jl")

include("DeepONet.jl")

include("Training.jl")

include("Reconstruction.jl")

export SnapshotData
export collect_snapshots
export extract_coordinates
export sample_parameters

export DeepONetLayer
export build_deeponet
export precompute_trunk_matrix

export NormalizationStats
export TrainingConfig
export TrainingResult
export train_operator

export reconstruct_fe_function
export evaluate_rom

end # module
