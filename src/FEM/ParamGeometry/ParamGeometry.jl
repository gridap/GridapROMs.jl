"""
    module ParamGeometry

Parametric geometry — mapped grids and embedded triangulations for
parameter-dependent domains.

Supports PDEs on domains that deform with the parameters, as required by, e.g.,
shape-optimisation or fluid–structure-interaction problems.  Two main areas:

- **Mapped grids** — `PhysicalMap` and `DisplacementMap` represent the mapping
  from a reference domain to the physical domain at a given parameter sample.
  `ParamGrid`, `ParamMappedGrid`, `ParamUnstructuredGrid`, and
  `ParamMappedDiscreteModel` wrap the corresponding Gridap geometry objects so
  that the physical coordinates are `ParamArray`s (one set per sample).
  `mapped_grid` is the primary constructor.

- **Embedded / level-set triangulations** — `ParamSubCellData` and
  `ParamSubFacetData` carry cut-cell geometry (produced by `GridapEmbedded`'s
  `LevelSetCutters`) as `ParamArray`s so that the cut geometry can vary with the
  parameters.  These are consumed by the `Extensions` module for unfitted FEM.

The module depends on `Utils` (domain helpers) and `ParamDataStructures`
(`ParamArray`, `Realisation`), and is used by `ParamFESpaces` and `Extensions`.
"""
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

export PhysicalMap
export DisplacementMap
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

end
