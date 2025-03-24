using Gridap
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using GridapEmbedded
using ROManifolds
using ROManifolds.Extensions
using ROManifolds.DofMaps
using SparseArrays
using DrWatson
using Test
using DrWatson

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 10
partition = (n,n)

dp = pmax - pmin

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

model = CartesianDiscreteModel(pmin,pmax,partition)
cutgeo = cut(model,geo2)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Ωbg = Triangulation(model)
Ωact = Triangulation(cutgeo,ACTIVE)
Ωactout = Triangulation(cutgeo,ACTIVE_OUT)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ωout = Triangulation(cutgeo,PHYSICAL_OUT)

order = 1
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩout = Measure(Ωout,degree)
dΩact = Measure(Ωact,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

reffe = ReferenceFE(lagrangian,Float64,order)

Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)

# EXPERIMENTAL
Voagg = OrderedAgFEMSpace(Vact,aggregates)

zero_free_values(Vagg)
zero_free_values(Voagg)

a(u,v) = ∫(u*v)dΩact

A = assemble_matrix(a,Vagg,Vagg)
Aact = assemble_matrix(a,Vact,Vact)
# Ao = assemble_matrix(a,Voagg,Voagg)

u,v = get_trial_fe_basis(Voagg),get_fe_basis(Voagg)
cellvals = (∫(u*v)dΩact)[Ωact]
cellrows = get_cell_dof_ids(Voagg,Ωact)
cellcols = get_cell_dof_ids(Voagg,Ωact)

# c = get_cell_constraints(Vagg)
# co = get_cell_constraints(Voagg)

# cell_to_mask,cell_to_cellcut = Extensions._get_cell_to_in_cut_info(f)
# for (cell,mask) in enumerate(cell_to_mask)
#   if mask
#     @assert co[cell] == c[cell] "$cell"
#   end
# end

f = Voagg
cell_mat_c = attach_constraints_cols(f,cellvals,Ωact)
cell_mat_rc = attach_constraints_rows(f,cell_mat_c,Ωact)
rows = get_cell_dof_ids(f,Ωact)
cols = get_cell_dof_ids(f,Ωact)

# vector
cellvals = (∫(v)dΩact)[Ωact]
cell_vec_r = attach_constraints_rows(f,cellvals,Ωact)



ccell_to_cdofs = f.cutcell_to_lmdof_to_mdof
dof_to_cells = f.incell_to_lmdof_to_mdof
cell_to_dofs = Extensions.inverse_table(dof_to_cells)
# vals = cell_mat_rc
vals = cell_vec_r

k = Extensions.MoveConstrainedVecVals(ccell_to_cdofs,dof_to_cells,cell_to_dofs,vals)

cell_to_mask,cell_to_cellcut = Extensions._get_cell_to_in_cut_info(f)
cell_to_cellin = zeros(eltype(cell_to_cellcut),length(cell_to_cellcut))
incell = findall(iszero,cell_to_mask)
cell_to_cellin[incell] .= 1:length(incell)
moved_vals = lazy_map(k,cell_to_cellin,cell_to_cellcut)

c1,c2,move_vals = return_cache(k,cell_to_cellin[23],cell_to_cellcut[23])
cell,ccell = cell_to_cellin[23],cell_to_cellcut[23]
# evaluate!(c,k,cell_to_cellin[23],cell_to_cellcut[23])
cdofs = k.ccell_to_cdofs[ccell]
cvals = getindex!(c1,k.vals,ccell)

vecdata = ([moved_vals,],[rows,])
assem = SparseMatrixAssembler(f,f)
b = assemble_vector(assem,vecdata)

f = Voagg
cell = 23
dof_to_cells = Extensions.inverse_table(f.cell_to_lmdof_to_mdof.values)
cdofs = f.cell_to_lmdof_to_mdof[cell]
cdof = cdofs[1]
cells = dof_to_cells[cdof]
_cell = first(cells)
dofs = f.cell_to_lmdof_to_mdof[_cell]

if iszero(cell)
  cdofs = k.ccell_to_cdofs[ccell]
  cvals = getindex!(c1,k.vals,ccell)
  for (ci,cdof) in enumerate(cdofs)
    cvi = cvals[ci]
    cells = k.dof_to_cells[cdof]
    _cell = first(cells) # it does not matter where we move the constrained values
    dofs = k.cell_to_dofs[_cell]
    vals = getindex!(c2,k.vals,_cell)
    for (i,dof) in enumerate(dofs)
      if dof == cdof
        move_vals[i] = vals[i] + cvi
        break
      end
    end
  end
else
  move_vals = getindex!(c1,k.vals,cell)
end
