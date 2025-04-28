module GridapROMs

include("FEM/Utils/Utils.jl")

include("FEM/DofMaps/DofMaps.jl")

include("FEM/TProduct/TProduct.jl")

include("FEM/ParamDataStructures/ParamDataStructures.jl")

include("FEM/ParamAlgebra/ParamAlgebra.jl")

include("FEM/ParamGeometry/ParamGeometry.jl")

include("FEM/ParamFESpaces/ParamFESpaces.jl")

include("FEM/ParamSteady/ParamSteady.jl")

include("FEM/ParamODEs/ParamODEs.jl")

include("FEM/Extensions/Extensions.jl")

include("RB/RBSteady/RBSteady.jl")

include("RB/RBTransient/RBTransient.jl")

include("Exports.jl")

end
