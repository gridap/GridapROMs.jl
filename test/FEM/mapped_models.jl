module ParamMappedModelTest

using GridapROMs
using Gridap
using Gridap.Geometry
using Test

domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)

μ = Realization([[1.0],[2.0]])
ϕ(μ) = x->VectorValue(x[2],μ[1]*x[1])
ϕμ(μ) = parameterize(ϕ,μ)
mmodel = MappedDiscreteModel(model,ϕμ(μ))

Ωm = Triangulation(mmodel)
Γm = BoundaryTriangulation(mmodel,tags=8)

dΩm = Measure(Ωm,4)
dΓm = Measure(Γm,4)

g(μ) = x->x[1]+μ[1]*x[2]
gμ(μ) = parameterize(g,μ)

reffe = ReferenceFE(lagrangian,Float64,2)
Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
Um = ParamTrialFESpace(Vm,gμ)

Umμ = Um(μ)

ν(μ) = x->x[1]+μ[1]*x[2]
νμ(μ) = parameterize(ν,μ)
f(μ) = x->x[1]+μ[1]*x[2]
fμ(μ) = parameterize(f,μ)

am(μ,u,v) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩm
bm(μ,u,v) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩm - ∫(fμ(μ)*v)dΩm #+ ∫(fμ(μ)*v)dΓm

pspace = ParamSpace((1.0,2.0))
opm = LinearParamOperator(bm,am,pspace,Um,Vm)

xm, = solve(LUSolver(),opm,μ)

function gridap_solution(μ)
  mmodel = MappedDiscreteModel(model,ϕ(μ))
  Ωm = Triangulation(mmodel)
  dΩm = Measure(Ωm,4)
  Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  Um = TrialFESpace(Vm,g(μ))
  am(u,v) = ∫(ν(μ)*∇(v)⋅∇(u))dΩm
  bm(v) = ∫(f(μ)*v)dΩm #+ ∫(f(μ)*v)dΓm
  opm = AffineFEOperator(am,bm,Um,Vm)
  xm = solve(LUSolver(),opm)
  get_free_dof_values(xm)
end

for (i,μi) in enumerate([1.0,2.0])
  @test xm[i] ≈ gridap_solution(μi)
end

end

using DrWatson
using LinearAlgebra
using Serialization

using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Geometry
using Gridap.Arrays
using Gridap.Algebra
using Gridap.CellData
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.ODEs
using GridapROMs.DofMaps
using GridapROMs.Uncommon
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapROMs.Utils

domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)

μ = Realization([[1.0],[2.0]])
ϕ(μ) = x->VectorValue(x[2],μ[1]*x[1])
ϕμ(μ) = parameterize(ϕ,μ)
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},2)
V = TestFESpace(model,reffe;conformity=:H1)#,dirichlet_tags="boundary")
U = ParamTrialFESpace(V)
ϕμh = interpolate(ϕμ(μ),U(μ))

mmodel = MappedDiscreteModel(model,ϕμh)

CIAO
# grid = mmodel.mapped_grid

# mmodelok = MappedDiscreteModel(model,ϕμ(μ))
# gridok = mmodelok.mapped_grid

# grid = get_grid(model)
# cell_node_ids = get_cell_node_ids(grid)
# old_nodes = get_node_coordinates(grid)
# node_coordinates = Vector{eltype(old_nodes)}(undef,length(old_nodes))
# cell_to_coords = get_cell_coordinates(grid)

# phys_map = get_data(ϕμh)
# cell_coords_map = lazy_map(evaluate,phys_map,cell_to_coords)

# phys_map_ok = Fill(GenericField(ϕμ(μ)),num_cells(grid))
# cell_coords_map_ok = lazy_map(evaluate,phys_map_ok,cell_to_coords)

# UU = U(μ)
# # obj = ϕμ(μ)
# # fv = zero_free_values(UU)
# # cell_vals = FESpaces._cell_vals(UU,obj)
# # gather_free_values!(fv,UU,cell_vals)

# # s = get_fe_dof_basis(UU)
# # trian = get_triangulation(s)
# # cf = CellField(obj,trian,DomainStyle(s))
# # # cell_vals = s(cf)
# # c = return_cache(s,cf)
# # # evaluate!(c,s,cf)
# # # lazy_map(evaluate,get_data(s),get_data(cf))
# # aa,bb = get_data(s)[1],get_data(cf)[1][1]
# # c = return_cache(aa,bb)
# # # evaluate!(c,aa,bb)

# # Create a FESpace for the geometrical description
# # The FEFunction that describes the coordinate field
# # or displacement can be a interpolation or a solution
# # of a mesh displacement problem
# T = eltype(get_node_coordinates(model))
# Ts = eltype(T)

# order = 2
# os = Fill(order,num_cells(model))
# _ps = get_polytopes(model)
# ct = get_cell_type(model)
# ps = lazy_map(Reindex(_ps), ct)

# f(a,b) = LagrangianRefFE(T,a,b)
# reffes = lazy_map(f,ps,os)
# Vₕ = ParamTrialFESpace(FESpace(model,reffes;conformity=:H1))(μ)

# fs(a,b) = LagrangianRefFE(Ts,a,b)
# s_reffes = lazy_map(f,ps,os)
# Vₕ_scal = FESpace(model,s_reffes;conformity=:H1)

# grid = get_grid(model)
# geo_map = phys_map

# cell_ctype = get_cell_type(grid)
# c_reffes = get_reffes(grid)

# # Create a fe_map using the cell_map that can be evaluated at the
# # vertices of the fe space (nodal type)
# # This returns a FEFunction initialised with the coordinates
# # But this is to build the FEFunction that will be inserted, it is
# # an advanced constructor, not needed at this stage

# c_dofs = get_fe_dof_basis(Vₕ)
# dof_basis = get_data(c_dofs)
# c_nodes = lazy_map(get_nodes,get_data(c_dofs))

# xh = zero(Vₕ)
# c_dofv = lazy_map(evaluate,dof_basis,geo_map)

# Uₕ = TrialFESpace(Vₕ)

# fv = get_free_dof_values(xh)
# dv = get_dirichlet_dof_values(get_fe_space(xh))
# gather_free_and_dirichlet_values!(fv,dv,Uₕ,c_dofv)

# c_xh = lazy_map(evaluate,get_data(xh),c_nodes)
# c_scal_ids = get_cell_dof_ids(Vₕ_scal)

# Tn = eltype(eltype(c_xh))
# # nodes_coords = Vector{Tn}(undef,num_free_dofs(Vₕ_scal))
# pnodes_coords = parameterize(zeros(eltype(Tn),num_free_dofs(Vₕ_scal)),2)
# Geometry._cell_vector_to_dof_vector!(pnodes_coords,c_scal_ids,c_xh)


#
grid = get_grid(model)
model_map = get_cell_map(grid)
phys_map = interpolate(x -> VectorValue(x[2],2*x[1]),V)
geo_map = lazy_map(∘,phys_map.cell_field.cell_field,model_map)

cell_node_ids = get_cell_node_ids(grid)
old_nodes = get_node_coordinates(grid)
node_coordinates = Vector{eltype(old_nodes)}(undef,length(old_nodes))
c_coor = get_cell_coordinates(grid)
map_c_coor = lazy_map(evaluate,phys_map.cell_field.cell_field,c_coor)
Geometry._cell_vector_to_dof_vector!(node_coordinates,cell_node_ids,map_c_coor)

grid_ok = MappedGrid(grid,x -> VectorValue(x[2],2*x[1]))
phys_map_ok = Fill(x -> VectorValue(x[2],2*x[1]),num_cells(model))
map_c_coor_ok = lazy_map(evaluate,phys_map_ok,c_coor)

f = phys_map_ok[1]
x = c_coor[1]
c = return_cache(f,x)
y = evaluate!(c,f,x)
f(x)

x1 = evaluate(phys_map.cell_field.cell_field[1],c_coor[1])
c = return_cache(phys_map.cell_field.cell_field[1],c_coor[1])
# evaluate!(c,phys_map.cell_field.cell_field[1],c_coor[1])
cf,ck = c
fx = evaluate!(cf,phys_map.cell_field.cell_field[1].fields,c_coor[1])

# dof_basis = get_data(get_fe_dof_basis(V))
dv = get_data(get_fe_basis(V))
# c_dofv = lazy_map(evaluate,dof_basis,phys_map.cell_dof_values)
# c_dofv = lazy_map(evaluate,dv,phys_map.cell_dof_values)
lazy_map(evaluate,dv,phys_map.cell_field.cell_field)
lazy_map(evaluate,dof_basis,phys_map.cell_field.cell_field)

x = get_cell_points(Triangulation(model))
y = phys_map(x)

c_coor = get_cell_coordinates(grid)
z = lazy_map(evaluate,get_data(phys_map),c_coor)
