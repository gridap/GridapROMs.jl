module ParamGeometry

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.Fields
using Gridap.Geometry
using Gridap.Helpers
using Gridap.ReferenceFEs

using GridapEmbedded
using GridapEmbedded.Interfaces
using GridapEmbedded.LevelSetCutters

using GridapROMs.Utils
using GridapROMs.ParamDataStructures

import FillArrays: Fill
import Gridap.CellData: similar_cell_field
import Gridap.Visualization: _prepare_node_to_coords

export ParamGrid
export ParamMappedGrid
export ParamMappedDiscreteModel
export ParamUnstructuredGrid
export mapped_grid
include("ParamGrids.jl")

export ParamSubCellData
include("SubCellTriangulations.jl")

export ParamSubFacetData
include("SubFacetTriangulations.jl")

export ConfigurationStyle
export ReferenceConfiguration
export ConfigurationAtIndex
export ConfigurationTriangulation
export get_configuration_triangulation
export ReferenceMeasure
export reference_integrate
export ∇₀, n₀
include("ReferenceInterface.jl")

end
