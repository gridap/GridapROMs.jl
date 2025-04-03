"""
    abstract type OrderedFESpace{S} <: SingleFieldFESpace end

Interface for FE spaces that feature a DOF reordering
"""
abstract type OrderedFESpace{S} <: SingleFieldFESpace end

# FESpace interface

FESpaces.get_fe_space(f::OrderedFESpace) = @abstractmethod

"""
    get_cell_odof_ids(f::OrderedFESpace) -> AbstractArray

Fetches the ordered connectivity structure
"""
get_cell_odof_ids(f::OrderedFESpace) = @abstractmethod

FESpaces.ConstraintStyle(::Type{<:OrderedFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::OrderedFESpace) = get_free_dof_ids(get_fe_space(f))

FESpaces.get_triangulation(f::OrderedFESpace) = get_triangulation(get_fe_space(f))

FESpaces.get_dof_value_type(f::OrderedFESpace) = get_dof_value_type(get_fe_space(f))

FESpaces.get_cell_dof_ids(f::OrderedFESpace) = get_cell_odof_ids(f)

FESpaces.get_fe_basis(f::OrderedFESpace) = get_fe_basis(get_fe_space(f))

FESpaces.get_trial_fe_basis(f::OrderedFESpace) = get_trial_fe_basis(get_fe_space(f))

FESpaces.get_fe_dof_basis(f::OrderedFESpace) = get_fe_dof_basis(get_fe_space(f))

FESpaces.get_cell_isconstrained(f::OrderedFESpace) = get_cell_isconstrained(get_fe_space(f))

FESpaces.get_cell_constraints(f::OrderedFESpace) = get_cell_constraints(get_fe_space(f))

FESpaces.get_dirichlet_dof_ids(f::OrderedFESpace) = get_dirichlet_dof_ids(get_fe_space(f))

FESpaces.get_cell_is_dirichlet(f::OrderedFESpace) = get_cell_is_dirichlet(get_fe_space(f))

FESpaces.num_dirichlet_dofs(f::OrderedFESpace) = num_dirichlet_dofs(get_fe_space(f))

FESpaces.num_dirichlet_tags(f::OrderedFESpace) = num_dirichlet_tags(get_fe_space(f))

FESpaces.get_dirichlet_dof_tag(f::OrderedFESpace) = get_dirichlet_dof_tag(get_fe_space(f))

FESpaces.get_vector_type(f::OrderedFESpace) = get_vector_type(get_fe_space(f))

# Scatters correctly ordered free and dirichlet values
function FESpaces.scatter_free_and_dirichlet_values(f::OrderedFESpace,fv,dv)
  cell_dof_ids = get_cell_dof_ids(f)
  cell_values = lazy_map(Broadcasting(PosNegReindex(fv,dv)),cell_dof_ids)
  cell_ovalue_to_value(f,cell_values)
end

# Gathers correctly ordered free and dirichlet values
function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::OrderedFESpace,cv)
  cell_ovals = cell_value_to_ovalue(f,cv)
  cell_dofs = get_cell_dof_ids(f)
  cache_vals = array_cache(cell_ovals)
  cache_dofs = array_cache(cell_dofs)
  cells = 1:length(cell_ovals)

  FESpaces._free_and_dirichlet_values_fill!(
    fv,
    dv,
    cache_vals,
    cache_dofs,
    cell_ovals,
    cell_dofs,
    cells)

  (fv,dv)
end

function get_dof_map(f::OrderedFESpace,args...)
  bg_dof_to_act_dof = get_bg_dof_to_act_dof(f,args...)
  bg_ndofs = length(bg_dof_to_act_dof)
  return VectorDofMap(bg_ndofs,bg_dof_to_act_dof)
end

# Constructors

function OrderedFESpace(trian::Triangulation,args...;kwargs...)
  act_f = OrderedFESpace(get_active_model(trian),args...;trian=act_trian,kwargs...)
  bg_f = OrderedFESpace(get_background_model(trian),args...;kwargs...)
  CartesianFESpace(act_f,bg_f)
end

function OrderedFESpace(model::CartesianDiscreteModel,args...;kwargs...)
  CartesianFESpace(FESpace(model,args...;kwargs...))
end

function OrderedFESpace(::DiscreteModel,args...;kwargs...)
  @notimplemented "Background model must be cartesian for the selected dof reordering"
end

struct CartesianFESpace{S<:SingleFieldFESpace} <: OrderedFESpace{S}
  space::S
  cell_odofs_ids::AbstractArray
  odofs_to_bg_odofs::AbstractVector
  bg_odofs_to_odofs::AbstractVector
end

function CartesianFESpace(f::SingleFieldFESpace)
  cell_odofs_ids = get_cell_odof_ids(f)
  odofs_to_bg_odofs = IdentityVector(num_free_dofs(f))
  bg_odofs_to_odofs = IdentityVector(num_free_dofs(f))
  CartesianFESpace(f,cell_odofs_ids,odofs_to_bg_odofs,bg_odofs_to_odofs)
end

function CartesianFESpace(model::DiscreteModel,args...;kwargs...)
  CartesianFESpace(FESpace(model,args...;kwargs...))
end

function CartesianFESpace(f::SingleFieldFESpace,g::SingleFieldFESpace)
  @notimplemented "Implement!"
end

FESpaces.get_fe_space(f::CartesianFESpace) = f.space

get_cell_odof_ids(f::CartesianFESpace) = f.cell_odofs_ids

get_bg_dof_to_act_dof(f::CartesianFESpace) = f.bg_odofs_to_odofs

get_act_dof_to_bg_dof(f::CartesianFESpace) = f.odofs_to_bg_odofs

function get_sparsity(f::CartesianFESpace,g::CartesianFESpace,args...)
  sparsity = SparsityPattern(f,g,args...)
  bg_rows_to_act_rows = get_bg_dof_to_act_dof(g)
  bg_cols_to_act_cols = get_bg_dof_to_act_dof(f)
  CartesianSparsity(sparsity,bg_rows_to_act_rows,bg_cols_to_act_cols)
end

# dof reordering

function get_cell_odof_ids(space::SingleFieldFESpace)
  cell_dofs_ids = get_cell_dof_ids(space)
  cell_to_parent_cell = get_cell_to_bg_cell(space)
  fe_dof_basis = get_data(get_fe_dof_basis(space))
  orders = get_polynomial_orders(space)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  _get_cell_odof_info(model,fe_dof_basis,cell_dofs_ids,cell_to_parent_cell,orders)
end

# utils

function _get_cell_odof_info(model::DiscreteModel,args...)
  @notimplemented "Background model must be cartesian the selected dof reordering"
end

function _get_cell_odof_info(
  model::CartesianDiscreteModel,
  fe_dof_basis::AbstractArray,
  cell_dofs_ids::AbstractArray,
  cell_to_parent_cell::AbstractVector,
  orders::Tuple)

  desc = get_cartesian_descriptor(model)
  periodic = desc.isperiodic
  ncells = desc.partition
  cells = CartesianIndices(ncells)
  pcells = view(cells,cell_to_parent_cell)
  onodes = LinearIndices(orders .* ncells .+ 1 .- periodic)

  dofs_to_odofs = get_dof_to_odof(fe_dof_basis,cell_dofs_ids,pcells,onodes,orders)
  cell_odof_ids = lazy_map(DofsToODofs(fe_dof_basis,dofs_to_odofs,orders),pcells)
  return cell_odof_ids
end

function get_dof_to_odof(fe_basis::AbstractVector{<:Dof},args...)
  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function get_dof_to_odof(fe_basis::AbstractVector{<:LagrangianDofBasis},args...)
  function compare(b1::LagrangianDofBasis,b2::LagrangianDofBasis)
    (b1.dof_to_node == b2.dof_to_node && b1.dof_to_comp == b2.dof_to_comp
    && b1.node_and_comp_to_dof == b2.node_and_comp_to_dof)
  end
  b1 = testitem(fe_basis)
  cmp = lazy_map(b2->compare(b1,b2),fe_basis)
  if sum(cmp) == length(fe_basis)
    get_dof_to_odof(b1,args...)
  else
    @notimplemented "This function is only implemented for Lagrangian dof bases"
  end
end

function get_dof_to_odof(fe_dof_basis::Fill{<:LagrangianDofBasis},args...)
  get_dof_to_odof(testitem(fe_dof_basis),args...)
end

function get_dof_to_odof(
  fe_dof_basis::LagrangianDofBasis{P,V},
  cell_dofs_ids::AbstractArray,
  cells::AbstractVector{CartesianIndex{D}},
  onodes::LinearIndices{D},
  orders::NTuple{D,Int}
  ) where {D,P,V}

  cache = array_cache(cell_dofs_ids)
  p = cubic_polytope(Val(D))
  node_to_i_onode = _local_node_to_pnode(p,orders)
  nnodes = length(onodes)
  ncomps = num_components(V)
  ndofs = nnodes*ncomps

  odofs = zeros(eltype(V),ndofs)
  for (icell,cell) in enumerate(cells)
    first_new_node = orders .* (Tuple(cell) .- 1) .+ 1
    onodes_range = map(enumerate(first_new_node)) do (i,ni)
      ni:ni+orders[i]
    end
    onodes_cell = view(onodes,onodes_range...)
    cell_dofs = getindex!(cache,cell_dofs_ids,icell)
    for node in 1:length(onodes_cell)
      comp_to_idof = fe_dof_basis.node_and_comp_to_dof[node]
      i_onode = node_to_i_onode[node]
      onode = onodes_cell[i_onode]
      for comp in 1:ncomps
        idof = comp_to_idof[comp]
        dof = cell_dofs[idof]
        odof = onode + (comp-1)*nnodes
        # change only location of free dofs
        odofs[odof] = dof > 0 ? one(eltype(V)) : dof
      end
    end
  end

  nfree = 0
  for (i,odof) in enumerate(odofs)
    if odof > 0
      nfree += 1
      odofs[i] = nfree
    end
  end

  node_and_comps_to_odof = _get_node_and_comps_to_odof(fe_dof_basis,odofs,onodes)
  return node_and_comps_to_odof
end

function _get_node_and_comps_to_odof(
  ::LagrangianDofBasis{P,V},
  vec_odofs,
  onodes
  ) where {P,V}

  reshape(vec_odofs,size(onodes))
end

function _get_node_and_comps_to_odof(
  ::LagrangianDofBasis{P,V},
  vec_odofs,
  onodes
  ) where {P,V<:MultiValue}

  nnodes = length(onodes)
  ncomps = num_components(V)
  odofs = zeros(V,size(onodes))
  m = zero(Mutable(V))
  for onode in 1:nnodes
    for comp in 1:ncomps
      odof = onode + (comp-1)*nnodes
      m[comp] = vec_odofs[odof]
    end
    odofs[onode] = m
  end
  return odofs
end

function cell_ovalue_to_value(f::OrderedFESpace,cv)
  cell_dof_ids = get_cell_dof_ids(f)
  odof_to_dof = cell_dof_ids.maps[1].odof_to_dof
  lazy_map(OReindex(odof_to_dof),cv)
end

function cell_value_to_ovalue(f::OrderedFESpace,cv)
  cell_dof_ids = get_cell_dof_ids(f)
  odof_to_dof = cell_dof_ids.maps[1].odof_to_dof
  dof_to_odof = invperm(odof_to_dof)
  lazy_map(OReindex(dof_to_odof),cv)
end

function _local_node_to_pnode(p::Polytope,orders)
  _nodes, = Gridap.ReferenceFEs._compute_nodes(p,orders)
  pnodes = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
  return pnodes
end

cubic_polytope(::Val{d}) where d = @abstractmethod
cubic_polytope(::Val{1}) = SEGMENT
cubic_polytope(::Val{2}) = QUAD
cubic_polytope(::Val{3}) = HEX

_get_parent_model(model::DiscreteModel) = @abstractmethod
_get_parent_model(model::MappedDiscreteModel) = model.model
_get_parent_model(model::DiscreteModelPortion) = model.parent_model
_get_parent_model(model::CartesianDiscreteModel) = model
