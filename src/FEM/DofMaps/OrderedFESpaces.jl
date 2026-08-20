"""
    struct OrderedFESpace{S<:SingleFieldFESpace} <: SingleFieldFESpace
      space::S
      cell_odofs_ids::AbstractArray
      fe_odof_basis::CellDof
      dirichlet_odof_tag::Vector{Int8}
    end

Interface for FE spaces whose DOFs are ordered lexycographically. Notes:
- There are two separate lexycographies for free and Dirichlet DOFs, as in standard
  Gridap FE spaces: positive ids correspond to free DOFs, negative ids correspond
  to Dirichlet DOFs
- The following lexycographical hierarchy is considered: x-axis, y-axis, z-axis,
  component axis (only for multi-valued problems)
- An exception of the ordering is given by the function scatter_free_and_dirichlet_values,
  which returns the scattered (cell) values in the standard Gridap ordering, in
  order to prevent errors when comparing two FEFunctions defined on a standard
  FE space or an ordered FE space. This feature can be removed with some work in
  future versions
"""
struct OrderedFESpace{S<:SingleFieldFESpace} <: SingleFieldFESpace
  space::S
  cell_odofs_ids::AbstractArray
  fe_odof_basis::CellDof
  dirichlet_odof_tag::Vector{Int8}
end

# Constructors

function OrderedFESpace(f::UnconstrainedFESpace)
  cell_odofs_ids = get_cell_odof_ids(f)
  odof_to_dof = get_local_ordering(cell_odofs_ids)
  fe_odof_basis = get_fe_odof_basis(f,odof_to_dof)
  dirichlet_odof_tag = get_dirichlet_odof_tag(f,cell_odofs_ids)
  OrderedFESpace(f,cell_odofs_ids,fe_odof_basis,dirichlet_odof_tag)
end

function OrderedFESpace(f::SingleFieldFESpace)
  @notimplemented "For now, only implemented for an UnconstrainedFESpace"
end

function OrderedFESpace(model::CartesianDiscreteModel,args...;kwargs...)
  OrderedFESpace(FESpace(model,args...;kwargs...))
end

function OrderedFESpace(::DiscreteModel,args...;kwargs...)
  @notimplemented "Background model must be cartesian for the selected dof reordering"
end

# FESpace interface

FESpaces.get_fe_space(f::OrderedFESpace) = f.space

FESpaces.ConstraintStyle(::Type{<:OrderedFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::OrderedFESpace) = get_free_dof_ids(get_fe_space(f))

FESpaces.get_triangulation(f::OrderedFESpace) = get_triangulation(get_fe_space(f))

FESpaces.get_dof_value_type(f::OrderedFESpace) = get_dof_value_type(get_fe_space(f))

FESpaces.get_cell_dof_ids(f::OrderedFESpace) = f.cell_odofs_ids

FESpaces.get_fe_basis(f::OrderedFESpace) = get_fe_basis(get_fe_space(f))

FESpaces.get_trial_fe_basis(f::OrderedFESpace) = get_trial_fe_basis(get_fe_space(f))

FESpaces.get_fe_dof_basis(f::OrderedFESpace) = f.fe_odof_basis

FESpaces.get_cell_isconstrained(f::OrderedFESpace) = get_cell_isconstrained(get_fe_space(f))

FESpaces.get_cell_constraints(f::OrderedFESpace) = get_cell_constraints(get_fe_space(f))

FESpaces.get_dirichlet_dof_ids(f::OrderedFESpace) = get_dirichlet_dof_ids(get_fe_space(f))

FESpaces.get_cell_is_dirichlet(f::OrderedFESpace) = get_cell_is_dirichlet(get_fe_space(f))

FESpaces.num_dirichlet_dofs(f::OrderedFESpace) = num_dirichlet_dofs(get_fe_space(f))

FESpaces.num_dirichlet_tags(f::OrderedFESpace) = num_dirichlet_tags(get_fe_space(f))

FESpaces.get_dirichlet_dof_tag(f::OrderedFESpace) = f.dirichlet_odof_tag

FESpaces.get_vector_type(f::OrderedFESpace) = get_vector_type(get_fe_space(f))

# Scatters free and dirichlet values ordered according to Gridap
function FESpaces.scatter_free_and_dirichlet_values(f::OrderedFESpace,fv,dv)
  cell_dof_ids = get_cell_dof_ids(f)
  cell_values = lazy_map(Broadcasting(PosNegReindex(fv,dv)),cell_dof_ids)
  cell_ovalue_to_value(f,cell_values)
end

# Gathers free and dirichlet values ordered according to Gridap
function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::OrderedFESpace,cv)
  cell_ovals = cv
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

"""
    get_dof_perm(f::SingleFieldFESpace) -> Vector{Int}

Returns a permutation `p` of length `num_free_dofs(f)`, `p[dof]` being the
lexicographic rank of the free dof `dof` (in `f`'s own, native numbering), i.e.
the free-dof id `dof` would be assigned if `f` were wrapped in an
[`OrderedFESpace`](@ref). Computed directly from `f`, without constructing an
`OrderedFESpace` (and therefore without touching `f`'s assembly, dirichlet
values, etc). For an already lexicographically-ordered space (e.g. the `space`
field of an `OrderedFESpace`), this is the identity permutation.

This is used to make [`get_sparse_dof_map`](@ref) work for plain (unordered)
`TProductFESpace`s, by mapping their native dof ids to the lexicographic ranks
the tensor-product sparsity decomposition assumes, without requiring `f` itself
to be reordered.
"""
function get_dof_perm(f::SingleFieldFESpace)
  cell_odofs = get_cell_odof_ids(f)
  cell_dofs = get_cell_dof_ids(f)
  ncells = length(cell_dofs)
  odof_to_dof = zeros(Int,num_free_dofs(f))
  oddof_to_ddof = zeros(Int,num_dirichlet_dofs(f))
  get_bg_dof_to_dof!(odof_to_dof,oddof_to_ddof,cell_odofs,cell_dofs,1:ncells)
  invperm(odof_to_dof)
end

# `get_cell_odof_ids` assumes its input `cell_dof_ids` is in Gridap's native local
# order (it's re-permuted alongside `f`'s `fe_dof_basis` in lockstep). An
# `OrderedFESpace`'s `cell_dof_ids` is already lex-local-ordered, so re-running the
# generic path on it does not reduce to the identity: it double-applies the local
# permutation. Since an `OrderedFESpace`'s free dofs are lex-ordered by
# construction, short-circuit to the identity instead.
get_dof_perm(f::OrderedFESpace) = Base.OneTo(num_free_dofs(f))

# To get an FESpace whose free dofs are *actually* relabeled to lexicographic
# rank (e.g. so that two independently-assembled matrices share the same dof
# numbering, making their `SparseMatrixDofMap`s literally comparable) without
# paying for `OrderedFESpace`'s local reordering/assembler overrides, combine
# `get_dof_perm` with Gridap's own `Gridap.FESpaces.reindex_free_dof_ids`:
#
#   space′ = Gridap.FESpaces.reindex_free_dof_ids(space,get_dof_perm(space))
#
# Gridap's `reindex_free_and_dirichlet_dof_ids` (for `UnconstrainedFESpace`)
# does exactly the value-relabeling-only trick this comment describes: it
# builds `lazy_map(Broadcasting(PosNegReindex(free_dof_ids,dir_dof_ids)),
# get_cell_dof_ids(space))`, i.e. relabels each cell's dof *values* while
# preserving their *positions* — safe for assembly, since Gridap's assembler
# only requires `get_cell_dof_ids(f)[cell][li]` and the cell-local element
# matrix `vs[li,lj]` to refer to the same local basis function at position
# `li`, independent of which value labels it. It returns a genuine
# `UnconstrainedFESpace`, so `get_dof_perm` (generic path) applies to it
# unchanged and correctly reduces to the identity.

# utils

function _get_cell_odof_info(model::DiscreteModel,args...)
  @notimplemented "Background model must be cartesian the selected dof reordering"
end

function _get_cell_odof_info(
  model::CartesianDiscreteModel,
  fe_dof_basis::AbstractArray,
  cell_dofs_ids::AbstractArray,
  cell_to_parent_cell::AbstractVector,
  orders::Tuple
  )

  desc = get_cartesian_descriptor(model)
  periodic = desc.isperiodic
  ncells = desc.partition
  cells = CartesianIndices(ncells)
  pcells = view(cells,cell_to_parent_cell)
  onodes = LinearIndices(orders .* ncells .+ 1)

  dofs_to_odofs = get_dof_to_odof(fe_dof_basis,cell_dofs_ids,pcells,onodes,orders,periodic)
  cell_odof_ids = lazy_map(DofsToODofs(fe_dof_basis,dofs_to_odofs,orders),pcells)
  return OTable(cell_odof_ids)
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
  cells::AbstractArray{CartesianIndex{D}},
  onodes::LinearIndices{D},
  orders::NTuple{D,Int},
  periodic::NTuple{D,Bool}
  ) where {D,P,V}

  cache = array_cache(cell_dofs_ids)
  p = cubic_polytope(Val(D))
  node_to_i_onode = _local_node_to_pnode(p,orders)
  nnodes = length(onodes)
  ncomps = num_components(V)
  ndofs = nnodes*ncomps

  o = one(eltype(V))
  odofs = zeros(eltype(V),ndofs)
  periodic_cell = falses(orders.+1)
  for (icell,cell) in enumerate(cells)
    onodes_cell = _fetch_from_cell!(periodic_cell,onodes,orders,cell,periodic)
    cell_dofs = getindex!(cache,cell_dofs_ids,icell)
    for node in 1:length(onodes_cell)
      comp_to_idof = fe_dof_basis.node_and_comp_to_dof[node]
      i_onode = node_to_i_onode[node]
      onode = onodes_cell[i_onode]
      periodic_cell[i_onode] && continue
      for comp in 1:ncomps
        idof = comp_to_idof[comp]
        dof = cell_dofs[idof]
        odof = onode + (comp-1)*nnodes
        odofs[odof] = dof > 0 ? o : -o
      end
    end
  end

  _posneg_cumsum!(odofs)
  node_and_comps_to_odof = _get_node_and_comps_to_odof(fe_dof_basis,odofs,onodes,periodic)
  return node_and_comps_to_odof
end

function _get_node_and_comps_to_odof(
  ::LagrangianDofBasis{P,V},
  vec_odofs,
  onodes,
  periodic
  ) where {P,V}

  node_and_comps_to_odof = reshape(vec_odofs,size(onodes))
  _add_periodicity!(node_and_comps_to_odof,periodic)
  return node_and_comps_to_odof
end

function _get_node_and_comps_to_odof(
  ::LagrangianDofBasis{P,V},
  vec_odofs,
  onodes,
  periodic
  ) where {P,V<:MultiValue}

  nnodes = length(onodes)
  ncomps = num_components(V)
  node_and_comps_to_odof = zeros(V,size(onodes))
  m = zero(Mutable(V))
  for onode in 1:nnodes
    for comp in 1:ncomps
      odof = onode + (comp-1)*nnodes
      m[comp] = vec_odofs[odof]
    end
    node_and_comps_to_odof[onode] = m
  end
  _add_periodicity!(node_and_comps_to_odof,periodic)
  return node_and_comps_to_odof
end

function cell_ovalue_to_value(f::OrderedFESpace,cv)
  odof_to_dof = get_local_ordering(f)
  lazy_map(OReindex(),odof_to_dof,cv)
end

function get_fe_odof_basis(f::SingleFieldFESpace,odof_to_dof)
  s = get_fe_dof_basis(f)
  dof_to_odof = invperm_table(odof_to_dof)
  data = _get_fe_odof_basis(get_data(s),dof_to_odof)
  CellDof(data,s.trian,s.domain_style)
end

function _get_fe_odof_basis(
  cell_dof::AbstractArray{<:Union{Dof,AbstractArray{<:Dof}}},
  dof_to_odof
  )

  map(_get_fe_odof_basis,cell_dof,dof_to_odof)
end

function _get_fe_odof_basis(dof::Union{Dof,AbstractArray{<:Dof}},dof_to_odof)
  @abstractmethod
end

function _get_fe_odof_basis(dof::LagrangianDofBasis,dof_to_odof)
  odof_to_node = dof.dof_to_node[dof_to_odof]
  odof_to_comp = dof.dof_to_comp[dof_to_odof]
  node_and_comp_to_odof = dof.node_and_comp_to_dof[dof_to_odof]
  LagrangianDofBasis(dof.nodes,odof_to_node,odof_to_comp,node_and_comp_to_odof)
end

function _get_fe_odof_basis(
  dof::LagrangianDofBasis{P,V},
  dof_to_odof
  ) where {P,V<:MultiValue}

  odof_to_node = similar(dof.dof_to_node)
  odof_to_comp = similar(dof.dof_to_comp)
  node_and_comp_to_odof = similar(dof.node_and_comp_to_dof)
  ncomps = num_indep_components(V)
  nnodes = Int(length(odof_to_node) / ncomps)
  m = zero(MVector{ncomps,Int})
  for node in 1:nnodes
    for comp in 1:ncomps
      o = nnodes*(comp-1)
      odof = dof_to_odof[node+o]
      odof_to_comp[odof] = comp
      odof_to_node[odof] = node
      m[comp] = odof
    end
    node_and_comp_to_odof[node] = Tuple(m)
  end
  LagrangianDofBasis(dof.nodes,odof_to_node,odof_to_comp,node_and_comp_to_odof)
end

function get_dirichlet_odof_tag(f::SingleFieldFESpace,cell_odofs)
  dof_to_tag = get_dirichlet_dof_tag(f)
  oddof_to_ddof = _get_oddof_to_ddof(f,cell_odofs)
  odof_to_tag = zeros(eltype(dof_to_tag),length(oddof_to_ddof))
  for (oddof,ddof) in enumerate(oddof_to_ddof)
    odof_to_tag[oddof] = dof_to_tag[ddof]
  end
  return odof_to_tag
end

function _get_oddof_to_ddof(f::SingleFieldFESpace,cell_odofs)
  oddof_to_ddof = zeros(Int32,num_dirichlet_dofs(f))
  cell_dofs = get_cell_dof_ids(f)
  cache = array_cache(cell_dofs)
  ocache = array_cache(cell_odofs)
  for cell in 1:length(cell_dofs)
    dofs = getindex!(cache,cell_dofs,cell)
    odofs = getindex!(ocache,cell_odofs,cell)
    for (iodof,odof) in enumerate(odofs)
      if odof < 0
        idof = odofs.terms[iodof]
        dof = dofs[idof]
        @check dof < 0
        oddof_to_ddof[-odof] = -dof
      end
    end
  end
  return oddof_to_ddof
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

function _fetch_from_cell!(periodic_cell,onodes,orders,cell,periodic)
  fill!(periodic_cell,false)
  D = length(orders)
  first_new_node = orders .* (Tuple(cell) .- 1) .+ 1
  onodes_range = ()
  @inbounds for (i,ni) in enumerate(first_new_node)
    onodes_range = (onodes_range...,ni:(ni+orders[i]))
    if periodic[i] && all(cell.I[setdiff(1:D,i)] .== 1) 
      selectdim(periodic_cell,D,i) .= true
    end
  end
  onodes_cell = view(onodes,onodes_range...)
  return onodes_cell
end

function _posneg_cumsum!(a)
  pos = 0
  neg = 0
  for i in eachindex(a)
    x = a[i]
    if x == 0
      continue
    elseif x > 0
      pos += 1
      a[i] = pos
    else
      neg -= 1
      a[i] = neg
    end
  end
  return a
end

function _add_periodicity!(a,periodic::NTuple{D,Bool}) where D
  for d in 1:D
    !periodic[d] || continue
    first_face = selectdim(a,d,1)
    last_face = selectdim(a,d,size(a,d))
    for i in CartesianIndices(first_face)
      if iszero(first_face[i])
        first_face[i] = last_face[i]
      end
    end
  end
end