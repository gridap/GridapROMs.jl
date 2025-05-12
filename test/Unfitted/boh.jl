using Gridap.FESpaces
using Gridap.Algebra
using GridapROMs.Utils
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.RBSteady

ncenters = 5
red = LocalReduction(state_reduction,ncenters)
loc_rbsolver = RBSolver(fesolver,red;nparams_res=nparams,nparams_jac=nparams)
red_trial,red_test = reduced_spaces(loc_rbsolver,feop,fesnaps)

ress = residual_snapshots(loc_rbsolver,feop,fesnaps)
res_red = RBSteady.get_residual_reduction(loc_rbsolver)
rhs = reduced_residual(res_red,red_test,ress)

jacs = jacobian_snapshots(loc_rbsolver,feop,fesnaps)

μ = realization(pspace;sampling=:uniform)

rhsμ = RBSteady.get_local(rhs,μ)[1]
b = RBSteady.allocate_hypred_cache(rhsμ,μ)
fill!(b,zero(eltype(b)))
RBSteady.inv_project!(b,rhsμ,μ)

x = get_param_data(fesnaps)
r = get_realization(fesnaps)
J = jacobian(feop,r,x)

CIAO
# using SparseArrays
# using SparseArrays.HigherOrderFns
# using GridapROMs.ParamDataStructures
# x = get_param_data(fesnaps)
# r = get_realization(fesnaps)
# jacobian(feop,r,x)

# op1 = feop.operators[1]
# op2 = feop.operators[2]
# op3 = feop.operators[3]
# op4 = feop.operators[4]

# using GridapROMs.DofMaps
# celldofs1 = op1.op.feop.feop.test.space.bg_cell_dof_ids
# celldofs2 = op2.op.feop.feop.test.space.bg_cell_dof_ids
# celldofs3 = op3.op.feop.feop.test.space.bg_cell_dof_ids
# celldofs4 = op4.op.feop.feop.test.space.bg_cell_dof_ids
# celldofs = [celldofs1,celldofs2,celldofs3,celldofs4]

# cell2cell1 = get_cell_to_bg_cell(op1.op.feop.feop.test.space)
# cell2cell2 = get_cell_to_bg_cell(op2.op.feop.feop.test.space)
# cell2cell3 = get_cell_to_bg_cell(op3.op.feop.feop.test.space)
# cell2cell4 = get_cell_to_bg_cell(op4.op.feop.feop.test.space)
# cell2cell = [cell2cell1,cell2cell2,cell2cell3,cell2cell4]

celldofs1234 = common_table(celldofs,cell2cell)

# c1 = array_cache(celldofs1)
# c2 = array_cache(celldofs2)
# c12 = array_cache(celldofs12)

# for i in eachindex(celldofs1)
#   a1 = getindex!(c1,celldofs1,i)
#   a2 = getindex!(c2,celldofs2,i)
#   a12 = getindex!(c12,celldofs12,i)
#   @check intersect(a1,a12) == a1
#   @check intersect(a2,a12) == a2
#   d1 = setdiff(a1,a12)
#   d2 = setdiff(a2,a12)
#   @check intersect(a2,d1) == d1
#   @check intersect(a1,d2) == d2
# end
