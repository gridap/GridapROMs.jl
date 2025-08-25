module PoissonTests
using Gridap
using GridapROMs
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using Test

function main(distribute, parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  output = mkpath(joinpath(@__DIR__,"output"))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,cells)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet",[1,2,3,5,7])
  add_tag_from_tags!(labels,"neumann",[4,6,8])

  Ω = Triangulation(model)
  Γn = Boundary(model,tags="neumann")
  n_Γn = get_normal_vector(Γn)

  k = 2
  u((x,y)) = (x+y)^k
  f(x) = -Δ(u,x)
  g = n_Γn⋅∇(u)

  α(μ) = x -> sum(μ)
  αμ(μ) = parameterize(α,μ)

  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="dirichlet")
  U = TrialFESpace(u,V)

  dΩ = Measure(Ω,2*k)
  dΓn = Measure(Γn,2*k)

  a(μ,u,v) = ∫( αμ(μ)*∇(v)⋅∇(u) )dΩ
  l(μ,v) = a(μ,u,v) - ∫( αμ(μ)*v*f )dΩ - ∫( v*g )dΓn
  # assem = SparseMatrixAssembler(U,V,SubAssembledRows())
  op = LinearParamFEOperator(l,a,U,V)

  uh = solve(op)
  eh = u - uh
  @test sqrt(sum( ∫(abs2(eh))dΩ )) < 1.0e-9
end

end # module

using Gridap
using GridapROMs
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using PartitionedArrays
using Test

parts = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(parts),)))
end
# output = mkpath(joinpath(@__DIR__,"output"))

domain = (0,4,0,4)
cells = (4,4)
model = CartesianDiscreteModel(ranks,parts,domain,cells)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,5,7])
add_tag_from_tags!(labels,"neumann",[4,6,8])

Ω = Triangulation(model)
Γn = Boundary(model,tags="neumann")
n_Γn = get_normal_vector(Γn)

k = 2
u(μ) = x -> (μ[1]*x[1]+x[2])^k
f(μ) = x -> -Δ(u(μ),x)
fμ(μ) = parameterize(f,μ)
uμ(μ) = parameterize(u,μ)

reffe = ReferenceFE(lagrangian,Float64,k)
V = TestFESpace(model,reffe,dirichlet_tags="dirichlet")
U = ParamTrialFESpace(V,uμ)

pspace = ParamSpace((0,1))

dΩ = Measure(Ω,2*k)
dΓn = Measure(Γn,2*k)

a(μ,u,v) = ∫( ∇(v)⋅∇(u) )dΩ
l(μ,u,v) = a(μ,u,v) - ∫( v*fμ(μ) )dΩ - ∫( v*(n_Γn⋅∇(uμ(μ))) )dΓn
op = LinearParamOperator(l,a,pspace,U,V)

μ = realization(pspace)
uh, = solve(LUSolver(),op,μ)

μ1 = 0.5
U1 = TrialFESpace(V,u(μ1))
a1(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
l1(v) = ∫( v*f(μ1) )dΩ + ∫( v*(n_Γn⋅∇(u(μ1))) )dΓn
op1 = AffineFEOperator(a1,l1,U1,V)
u1 = solve(LUSolver(),op1.op)

UU = get_trial(op)(μ)
x = zero_free_values(UU)
# solve!(x,LUSolver(),op,μ)
nlop = parameterize(op,μ)
syscache = allocate_systemcache(nlop,x)
# solve!(x,LUSolver(),nlop,syscache)
residual!(syscache.b,nlop,x)
jacobian!(syscache.A,nlop,x)

b = syscache.b
b1 = op1.op.vector
A = syscache.A
A1 = op1.op.matrix

v,du = get_fe_basis(test),get_trial_fe_basis(test)
dc = ∫( ∇(v)⋅∇(du) )dΩ
matdata = collect_cell_matrix(U1,test,dc)
assem1 = SparseMatrixAssembler(test,test)
assem = parameterize(assem1,μ)

A = allocate_matrix(assem,matdata)

A1 = allocate_matrix(assem1,matdata)

A = local_views(A,get_rows(assem),get_cols(assem)).items[2]
assem = local_views(assem).items[2]
A1 = local_views(A1,get_rows(assem1),get_cols(assem1)).items[2]
assem1 = local_views(assem1).items[2]
matdata = matdata.items[2]

cellmat,cellidsrows,cellidscols = matdata[1][1],matdata[2][1],matdata[3][1]
rows_cache = array_cache(cellidsrows)
cols_cache = array_cache(cellidscols)
vals_cache = array_cache(cellmat)
mat1 = getindex!(vals_cache,cellmat,1)
rows1 = getindex!(rows_cache,cellidsrows,1)
cols1 = getindex!(cols_cache,cellidscols,1)
add! = AddEntriesMap(+)
add_cache = return_cache(add!,A,mat1,rows1,cols1)

# fill!(A.plids_to_value.data,0.0)
# fill!(A1.plids_to_value.nzval,0.0)
for cell in 1:length(cellidsrows)
  rows = getindex!(rows_cache,cellidsrows,cell)
  cols = getindex!(cols_cache,cellidscols,cell)
  vals = getindex!(vals_cache,cellmat,cell)
  evaluate!(add_cache,add!,A,vals,rows,cols)
  evaluate!(add_cache,add!,A1,vals,rows,cols)

  @assert A.plids_to_value.data[:,1] ≈ A1.plids_to_value.nzval "$(cell)"
end
cell = 3

# add_entries!(+,A,vals,rows,cols)
# Algebra._add_entries!(+,A,vals,rows,cols)
# add_entry!(+,A,vals[3,3],rows[3],cols[3])

function Base.setindex!(a::GridapDistributed.LocalView{T,2},v,i,j) where {T,N}
  ii,jj = map(_lid_to_plid,(i,j),a.d_to_lid_to_plid)
  @assert all(i->i>0,(ii,jj))
  println((ii,jj,v))
  a.plids_to_value[ii,jj] = v
end

@inline function Algebra._add_entries!(combine::Function,A,vs,is,js)
  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          vij = vs[li,lj]
          println(which(add_entry!,typeof.((combine,A,vij,i,j))))
          error("stop")
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end
