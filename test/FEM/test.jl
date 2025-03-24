using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields
using Gridap.Geometry
using Gridap.FESpaces
using Test
using LinearAlgebra
using Gridap.CellData
using Gridap.ReferenceFEs
using GridapEmbedded
using ROManifolds
using ROManifolds.Extensions
using ROManifolds.DofMaps

# domain = (0,1,0,1)
# partition = (2,2)
# model = CartesianDiscreteModel(domain,partition)

# aggregates = [1,0,3,0]
# Ωact = view(Triangulation(model),[1,3])
# Vact = FESpace(Ωact,ReferenceFE(lagrangian,Float64,1),conformity=:H1)
# Vagg = AgFEMSpace(Vact,aggregates)

domain = (0,1,0,1)
partition = (3,3)
model = CartesianDiscreteModel(domain,partition)

aggregates = [1,2,3,0,0,0,0,0,0]
Ωact = view(Triangulation(model),[1,2,3])
Vact = FESpace(Ωact,ReferenceFE(lagrangian,Float64,1),conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)

s = SparsityPattern(Vagg,Vagg)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
partition = (4,4)
dp = pmax - pmin

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

model = CartesianDiscreteModel(pmin,pmax,partition)
cutgeo = cut(model,geo2)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

reffe = ReferenceFE(lagrangian,Float64,1)
Ωact = Triangulation(cutgeo,ACTIVE)
Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)

s = SparsityPattern(Vagg,Vagg)
sact = SparsityPattern(Vact,Vact)

f = OrderedAgFEMSpace(Vact,aggregates)

# alternative integration

Ω = Triangulation(cutgeo,PHYSICAL_IN)
dΩ = Measure(Ω,2)
dΩact = Measure(Ωact,2)
v = get_fe_basis(f)
u = get_trial_fe_basis(f)
cellrows = get_cell_dof_ids(f)
cellcols = get_cell_dof_ids(f)
trian = Ωact

# vector
cellvec = (∫(v)dΩact)[trian]
cell_vec_r = attach_constraints_rows(f,cellvec,trian)

# matrix
cellmat = (∫(u*v)dΩact)[trian]
cell_mat_c = attach_constraints_cols(f,cellmat,trian)
cell_mat_rc = attach_constraints_rows(f,cell_mat_c,trian)

assem = SparseMatrixAssembler(f,f)
m1 = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)))

# # symbolic_loop_matrix!
# cellmat = cell_mat_rc
# rows_cache = array_cache(cellrows)
# cols_cache = array_cache(cellcols)
# mat1 = first(cellmat)

ncell = 16
cell_to_incell = zeros(Int32,ncell)
cell_to_cutcell = zeros(Int32,ncell)
incell = unique(f.cell_to_incell)
cutcell = setdiff(1:ncell,incell)
cell_to_incell[incell] .= 1:length(incell)
cell_to_cutcell[cutcell] .= 1:length(cutcell)

cellvals = collect(cell_mat_rc)

dof_to_cells = Extensions.inverse_table(f.cell_to_lmdof_to_mdof.values)

# cellmat_copy = copy(cellvals)
incells = setdiff(unique(cell_to_incell),0)
cellvals = copy(cellmat_copy)
for cell in 1:length(cellvals)
  iszero(cell_to_cutcell[cell]) && continue
  cutvals = cellvals[cell]
  cutrows = cellrows[cell]
  cutcols = cellcols[cell]
  for k in CartesianIndices(cutvals)
    println(cell,k)
    cutval = cutvals[k]
    iszero(cutval) && continue
    cutrow = cutrows[k.I[1]]
    cutcol = cutcols[k.I[2]]
    _inrowcells = dof_to_cells[cutrow]
    _incolcells = dof_to_cells[cutcol]
    _incells = intersect(_inrowcells,_incolcells)
    incell = first(intersect(_incells,incells))

    invals = cellvals[incell]
    incols = cellcols[incell]
    inrows = cellrows[incell]
    iincol = findfirst(incols.==cutcol)
    iinrow = findfirst(inrows.==cutrow)
    invals[iinrow,iincol] += cutval
    cutvals[k] -= cutval
  end
end

cell = 5
cutcell = cell_to_cutcell[cell]
# iszero(cutcell) && continue
cutvals = cellvals[cell] #(valscache,cellvals,cutcell)
cutrows = cellrows[cell]
cutcols = cellcols[cell]
k = CartesianIndex(2,1)
cutval = cutvals[k]
cutrow = cutrows[k.I[1]]
cutcol = cutcols[k.I[2]]
inrowcells = dof_to_cells[cutrow]
incolcells = dof_to_cells[cutcol]
incells = intersect(inrowcells,incolcells)
incell = first(incells)
invals = cellvals[incell]
incols = cellcols[incell]
inrows = cellrows[incell]
iincol = findfirst(incols.==cutcol)
iinrow = findfirst(inrows.==cutrow)
invals[iinrow,iincol] += cutval
cutvals[k] -= cutval

CIAO
# # utils

# struct MoveConstrainedVecVals{A,B,C,D} <: Map
#   vals::A
#   cell_to_dofs::B
#   dof_to_cells::C
# end

# function Arrays.return_cache(k::MoveConstrainedVecVals,cell,ccell)
#   @assert iszero(cell) != iszero(ccell)
#   c1 = array_cache(k.vals)
#   c2 = array_cache(k.vals)
#   c3 = array_cache(k.vals)
#   Tvals = eltype(eltype(k.vals))
#   nvals = length(testitem(k.cell_to_dofs))
#   move_vals = zeros(Tvals,nvals)
#   c1,c2,move_vals
# end

# function Arrays.evaluate!(cache,k::MoveConstrainedVecVals,cell,ccell)
#   c1,c2,move_vals = cache
#   if iszero(cell)
#     cdofs = k.ccell_to_cdofs[ccell]
#     cvals = getindex!(c1,k.vals,ccell)
#     for (ci,cdof) in enumerate(cdofs)
#       cvi = cvals[ci]
#       cells = k.dof_to_cells[cdof]
#       _cell = first(cells) # it does not matter where we move the constrained values
#       dofs = k.cell_to_dofs[_cell]
#       vals = getindex!(c2,k.vals,_cell)
#       for (i,dof) in enumerate(dofs)
#         if dof == cdof
#           move_vals[i] = vals[i] + cvi
#           break
#         end
#       end
#     end
#   else
#     move_vals = getindex!(c1,k.vals,cell)
#   end
#   return move_vals
# end

# m1 = CartesianDiscreteModel((0,1),(4,))
# V1 = FESpace(m1,reffe,conformity=:H1)
# s1 = SparsityPattern(V1,V1)

# Vbg = FESpace(model,reffe,conformity=:H1)
# sbg = SparsityPattern(Vbg,Vbg,Ωact)
# sact = SparsityPattern(Vact,Vact,Ω)

# matrix = assemble_matrix((u,v)->∫(u*v)dΩ,Vagg,Vagg)
# sagg = SparsityPattern(matrix)
# to_keep = findall(!iszero,sagg.matrix.nzval)
# I,J,V = findnz(matrix)
# I′ = map(md -> Vagg.mDOF_to_DOF[md],I[to_keep])
# J′ = map(md -> Vagg.mDOF_to_DOF[md],J[to_keep])
# V′ = V[to_keep]
# matrix′ = sparse(I′,J′,V′,25,25)
# sagg′ = SparsityPattern(matrix′)

# tts = TProductSparsity(sagg,[s1,s1])
# DofMaps.get_d_sparse_dofs_to_full_dofs(Float64,Float64,tts)

# tts = TProductSparsity(sagg′,[s1,s1])
# DofMaps.get_d_sparse_dofs_to_full_dofs(Float64,Float64,tts)

# f = OrderedAgFEMSpace(Vact,aggregates)
# _f = Extensions.OrderedFESpaceWithLinearConstraints(
#   f.space,
#   f.n_fdofs,
#   f.n_fmdofs,
#   f.mDOF_to_DOF,
#   f.DOF_to_mDOFs,
#   f.DOF_to_coeffs,
#   f.cell_to_incell,
#   f.cell_to_lmdof_to_mdof.values,
#   f.cell_to_ldof_to_dof)

# sagg = SparsityPattern(_f,_f)
