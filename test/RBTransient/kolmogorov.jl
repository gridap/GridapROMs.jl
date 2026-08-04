using Gridap
using GridapROMs

using Gridap.Algebra
using Gridap.FESpaces
using Gridap.CellData
using Gridap.ReferenceFEs
using Gridap.Polynomials
using Gridap.Arrays
using Gridap.Geometry
using GridapROMs.DofMaps
using GridapROMs.Utils

# 1D domain definition and partitioning
L = 1.0
Ω = (-L,L,-L,L)
partition = (10,10)

# Cartesian model creation with periodicity (the bug trigger)
model = CartesianDiscreteModel(Ω, partition, isperiodic=(true,false))

# Reference element definition (1st-order Lagrangian)
reffe = ReferenceFE(lagrangian, Float64, 2)

V_ordered = OrderedFESpace(model, reffe)

space = FESpace(model, reffe)
cell_dofs_ids = get_cell_dof_ids(space)
cell_to_parent_cell = get_cell_to_bg_cell(space)
fe_dof_basis = get_data(get_fe_dof_basis(space))
orders = get_polynomial_orders(space)
trian = get_triangulation(space)
model = get_background_model(trian)

fe_dof_basis = testitem(fe_dof_basis)

desc = get_cartesian_descriptor(model)
periodic = desc.isperiodic
ncells = desc.partition
cells = CartesianIndices(ncells)
pcells = view(cells,cell_to_parent_cell)
onodes = LinearIndices(orders .* ncells .+ 1)

D = num_cell_dims(model)
V = Int
cache = array_cache(cell_dofs_ids)
p = DofMaps.cubic_polytope(Val(D))
node_to_i_onode = DofMaps._local_node_to_pnode(p,orders)
nnodes = length(onodes)
ncomps = num_components(V)
ndofs = nnodes*ncomps

o = one(eltype(V))
odofs = zeros(eltype(V),ndofs)
for (icell,cell) in enumerate(cells)
  onodes_range,periodic_onodes = DofMaps._onode_and_periodic_info(orders,cell,periodic)
  onodes_cell = view(onodes,onodes_range...)
  cell_dofs = getindex!(cache,cell_dofs_ids,icell)
  for node in 1:length(onodes_cell)
    comp_to_idof = fe_dof_basis.node_and_comp_to_dof[node]
    i_onode = node_to_i_onode[node]
    onode = onodes_cell[i_onode]
    isperiodic_onode = onode in periodic_onodes
    for comp in 1:ncomps
      idof = comp_to_idof[comp]
      dof = cell_dofs[idof]
      odof = onode + (comp-1)*nnodes
      odofs[odof] = isperiodic_onode ? 0 : (dof > 0 ? o : -o)
    end
  end
end

function _posneg_cumsum!(a)
  apos = 0
  aneg = 0
  azer = -1
  for i in eachindex(a)
    if a[i] > 0
      apos += 1
      a[i] = apos
    elseif a[i] < 0
      aneg -= 1
      a[i] = aneg 
    else
      azer += 1
      apos -= azer
      aneg += azer
    end
  end
end

_posneg_cumsum!(odofs)

for (i,odof) in enumerate(odofs)
  if odof > 0
    nfree += 1
    odofs[i] = nfree 
  elseif odof < 0
    ndiri -= 1
    odofs[i] = ndiri
  end
end

node_and_comps_to_odof = DofMaps._get_node_and_comps_to_odof(fe_dof_basis,odofs,onodes)
DofMaps._add_periodicity!(node_and_comps_to_odof,periodic)

# 
model = CartesianDiscreteModel(Ω, (10,5))
mx = CartesianDiscreteModel(Ω, (10,5), isperiodic=(true,false))
my = CartesianDiscreteModel(Ω, (10,5), isperiodic=(false,true))

space = FESpace(model, reffe)
space_x = FESpace(mx, reffe)
space_y = FESpace(my, reffe)