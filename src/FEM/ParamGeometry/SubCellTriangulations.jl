function mapped_grid(
  style::GridMapStyle,
  trian::Interfaces.SubCellTriangulation,
  phys_map::AbstractVector
  )

  model = get_background_model(trian)
  subgrid = mapped_grid(trian.subgrid,phys_map)
  subcells = _change_coords(trian.subcells,subgrid)
  Interfaces.SubCellTriangulation(subcells,model)
end

struct ParamSubCellData{Dr,Dp,Tp,Tr} <: GridapType
  cell_to_points::Table{Int32,Vector{Int32},Vector{Int32}}
  cell_to_bgcell::Vector{Int32}
  point_to_coords::ParamBlock{Vector{Point{Dp,Tp}}}
  point_to_rcoords::Vector{Point{Dr,Tr}}
end

function Interfaces.SubCellData(
  cell_to_points::Table,
  cell_to_bgcell::AbstractVector,
  point_to_coords::ParamBlock,
  point_to_rcoords::AbstractVector
  )

  ParamSubCellData(
    cell_to_points,
    cell_to_bgcell,
    point_to_coords,
    point_to_rcoords)
end

function Geometry.UnstructuredGrid(st::ParamSubCellData{D}) where D
  reffe = LagrangianRefFE(Float64,Interfaces.Simplex(Val{D}()),1)
  cell_types = fill(Int8(1),length(st.cell_to_points))
  UnstructuredGrid(
    st.point_to_coords,
    st.cell_to_points,
    [reffe,],
    cell_types)
end

# Implementation of the Gridap.Triangulation interface

"""
    struct ParamSubCellTriangulation{Dc,Dp,T,A} <: Triangulation{Dc,Dp}

A triangulation for subcells.
"""
struct ParamSubCellTriangulation{Dc,Dp,T,A} <: Triangulation{Dc,Dp}
  subcells::ParamSubCellData{Dc,Dp,T}
  bgmodel::A
  subgrid::ParamUnstructuredGrid{Dc,Dp,T,NonOriented,Nothing}
  function ParamSubCellTriangulation(
    subcells::ParamSubCellData{Dc,Dp,T},bgmodel::DiscreteModel) where {Dc,Dp,T}
    subgrid = UnstructuredGrid(subcells)
    A = typeof(bgmodel)
    new{Dc,Dp,T,A}(subcells,bgmodel,subgrid)
  end
end

function Interfaces.SubCellTriangulation(subcells::ParamSubCellData,model::DiscreteModel)
  ParamSubCellTriangulation(subcells,model)
end

function Geometry.get_background_model(a::ParamSubCellTriangulation)
  a.bgmodel
end

function Geometry.get_grid(a::ParamSubCellTriangulation)
  a.subgrid
end

function Geometry.get_active_model(a::ParamSubCellTriangulation)
  msg = """
  This is not implemented, but also not needed in practice.
  Embedded Grids implemented for integration, not interpolation.
  """
  @notimplemented  msg
end

function Geometry.get_glue(a::ParamSubCellTriangulation{Dc},::Val{D}) where {Dc,D}
  if D != Dc
    return nothing
  end
  tface_to_mface = a.subcells.cell_to_bgcell
  tface_to_mface_map = Interfaces._setup_cell_ref_map(a.subcells,a.subgrid)
  FaceToFaceGlue(tface_to_mface,tface_to_mface_map,nothing)
end

function Geometry.move_contributions(scell_to_val::AbstractArray,strian::ParamSubCellTriangulation)
  model = get_background_model(strian)
  ncells = num_cells(model)
  cell_to_touched = fill(false,ncells)
  scell_to_cell = strian.subcells.cell_to_bgcell
  cell_to_touched[scell_to_cell] .= true
  Ωa = Triangulation(model,cell_to_touched)
  acell_to_val = move_contributions(scell_to_val,strian,Ωa)
  acell_to_val, Ωa
end

# utils

function _change_coords(sc::Interfaces.SubCellData,grid::Grid)
  Interfaces.SubCellData(
    sc.cell_to_points,
    sc.cell_to_bgcell,
    get_node_coordinates(grid),
    sc.point_to_rcoords
    )
end

function Base.isapprox(t::T,s::T) where T<:ParamSubCellData
  (
    t.cell_to_points == s.cell_to_points &&
    t.cell_to_bgcell == s.cell_to_bgcell &&
    length(testitem(t.point_to_coords)) == length(testitem(s.point_to_coords)) &&
    t.point_to_rcoords == s.point_to_rcoords
  )
end

function Base.isapprox(t::ParamSubCellData,s::Interfaces.SubCellData)
  (
    t.cell_to_points == s.cell_to_points &&
    t.cell_to_bgcell == s.cell_to_bgcell &&
    length(testitem(t.point_to_coords)) == length(s.point_to_coords) &&
    t.point_to_rcoords == s.point_to_rcoords
  )
end

function Base.isapprox(t::Interfaces.SubCellData,s::ParamSubCellData)
  (
    t.cell_to_points == s.cell_to_points &&
    t.cell_to_bgcell == s.cell_to_bgcell &&
    length(t.point_to_coords) == length(testitem(s.point_to_coords)) &&
    t.point_to_rcoords == s.point_to_rcoords
  )
end

function Utils.to_child(parent::ParamSubCellTriangulation,child::Geometry.TriangulationView)
  @check isa(child.parent,ParamSubCellTriangulation)
  Geometry.TriangulationView(parent,child.cell_to_parent_cell)
end
