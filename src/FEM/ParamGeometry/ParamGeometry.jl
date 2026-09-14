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

using GridapROMs.Utils
using GridapROMs.ParamDataStructures

import FillArrays: Fill
import Gridap.CellData: similar_cell_field

export PhysicalMap
export DisplacementMap
export ParamGrid
export ParamMappedGrid
export ParamMappedDiscreteModel
export ParamUnstructuredGrid
export mapped_grid
include("ParamGrids.jl")

# `ParamSubCellData`/`ParamSubFacetData` (cut-cell geometry from GridapEmbedded's
# `LevelSetCutters`) live in the `GridapROMsEmbeddedExt` package extension, not
# here - GridapEmbedded is a weak dependency (see Project.toml) since merely
# loading it alongside GridapDistributed triggers a severe Julia compiler stall
# on any distributed `TestFESpace` construction, unrelated to GridapROMs' own
# code (confirmed with a plain `Gridap+GridapDistributed+GridapEmbedded` repro).
# They become available as `Base.get_extension(GridapROMs,:GridapROMsEmbeddedExt)`
# members once the user's own script also does `using GridapEmbedded`.

end
