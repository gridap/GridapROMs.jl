using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays

using GridapROMs
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamDataStructures
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs
using GridapROMs.Distributed

using Test

parts = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(parts),)))
end

domain = (0,4,0,4)
cells = (4,4)
model = CartesianDiscreteModel(ranks,parts,domain,cells)
Ω = Triangulation(model)
Γ = Boundary(model)

pranges = fill([0,1],3)
pspace = ParamSpace(pranges)
μ = Realization([[1.0],[2.0],[3.0]])

f(μ) = x -> sum(μ)
fμ(μ) = parameterize(f,μ)
u(μ) = x -> (x[1]+x[2])*sum(μ)
uμ(μ) = parameterize(u,μ)
reffe = ReferenceFE(lagrangian,Float64,1)
V0 = TestFESpace(model,reffe,dirichlet_tags="boundary")
U = ParamTrialFESpace(V0,uμ)
@test isa(U,Distributed.DistributedUnEvalTrialFESpace)
Uμ = U(μ)
@test isa(Uμ,Distributed.DistributedSingleFieldParamFESpace)

free_values = zero_free_values(Uμ)
fh = FEFunction(Uμ,free_values)
zh = zero(Uμ)
uh = interpolate(uμ(μ),Uμ)
eh = uμ(μ) - uh

uh_dir = interpolate_dirichlet(uμ(μ),Uμ)
free_values = zero_free_values(Uμ)
dirichlet_values = get_dirichlet_dof_values(Uμ)
uh_dir2 = interpolate_dirichlet!(uμ(μ),free_values,dirichlet_values,Uμ)

uh_everywhere = interpolate_everywhere(uμ(μ),Uμ)
dirichlet_values0 = zero_dirichlet_values(Uμ)
uh_everywhere_ = interpolate_everywhere!(uμ(μ),free_values,dirichlet_values0,Uμ)
eh2 = uμ(μ) - uh_everywhere
eh2_ = uμ(μ) - uh_everywhere_

uh_everywhere2 = interpolate_everywhere(uh_everywhere,Uμ)
uh_everywhere2_ = interpolate_everywhere!(uh_everywhere,free_values,dirichlet_values,Uμ)
eh3 = uμ(μ) - uh_everywhere2

dofs = get_fe_dof_basis(Uμ)
cell_vals = dofs(uh)
gather_free_values!(free_values,Uμ,cell_vals)
gather_free_and_dirichlet_values!(free_values,dirichlet_values,Uμ,cell_vals)
uh4 = FEFunction(Uμ,free_values,dirichlet_values)
eh4 = uμ(μ) - uh4

dΩ = Measure(Ω,3)
cont   = ∫( abs2(eh) )dΩ
cont2  = ∫( abs2(eh2) )dΩ
cont2_ = ∫( abs2(eh2_) )dΩ
cont3  = ∫( abs2(eh3) )dΩ
cont4  = ∫( abs2(eh4) )dΩ

# Assembly
das = SubAssembledRows() #FullyAssembledRows()
Ωas  = Triangulation(das,model)
dΩa = Measure(Ωas,3)

a(μ,du,v) = ∫(fμ(μ)*du*v)dΩa
du = get_trial_fe_basis(Uμ)
v = get_fe_basis(V0)
matdata = collect_cell_matrix(Uμ,V0,a(μ,du,v))
assem = SparseMatrixAssembler(Uμ,V0)
assemμ = parameterize(assem,μ)

A = allocate_matrix(assemμ,matdata)
@test num_cols(assem) == size(A,2)
@test num_rows(assem) == size(A,1)
assemble_matrix!(A,a,matdata)
assemble_matrix_add!(A,a,matdata)
A = assemble_matrix(a,matdata)
@test num_cols(a) == size(A,2)
@test num_rows(a) == size(A,1)
b = allocate_vector(a,vecdata)
@test num_rows(a) == length(b)
assemble_vector!(b,a,vecdata)
assemble_vector_add!(b,a,vecdata)
b = assemble_vector(a,vecdata)
@test num_rows(a) == length(b)
A, b = allocate_matrix_and_vector(a,data)
assemble_matrix_and_vector!(A,b,a,data)
assemble_matrix_and_vector_add!(A,b,a,data)
@test num_cols(a) == size(A,2)
@test num_rows(a) == size(A,1)
@test num_rows(a) == length(b)
A, b = assemble_matrix_and_vector(a,data)
@test num_cols(a) == size(A,2)
@test num_rows(a) == size(A,1)
@test num_rows(a) == length(b)


U0 = HomogeneousTrialFESpace(U)
u0h = interpolate(0.0,U0)
cont  = ∫( abs2(u0h) )dΩ
@test sqrt(sum(cont)) < 1.0e-14

# OwnAndGhostVector partitions
V3 = FESpace(model,reffe,dirichlet_tags="boundary",split_own_and_ghost=true)
U3 = TrialFESpace(u,V3)
@test get_vector_type(V3) <: PVector{<:OwnAndGhostVectors}

free_values = zero_free_values(U3)
dirichlet_values = get_dirichlet_dof_values(U3)
uh = interpolate_everywhere(u,U3)
_uh = interpolate_everywhere(uh,U3)
__uh = interpolate_everywhere!(_uh,free_values,dirichlet_values,U3)

uh = interpolate(u,U3)
dofs      = get_fe_dof_basis(U3)
cell_vals = dofs(uh)
gather_free_values!(free_values,U3,cell_vals)
gather_free_and_dirichlet_values!(free_values,dirichlet_values,U3,cell_vals)
uh = FEFunction(U3,free_values,dirichlet_values)

# I need to use the square [0,2]² in the sequel so that
# when integrating over the interior facets, the entries
# of the vector which is assembled in assemble_tests(...)
# become one.
domain = (0,2,0,2)
cells = (4,4)
  dv = get_fe_basis(V0)
du = get_trial_fe_basis(U)
a(u,v) = ∫( fμ*∇(v)⋅∇(u) )dΩa
l(v) = ∫( fμ*v )dΩa
assem = SparseMatrixAssembler(U,V0,das)
zh = zero(U)
data = collect_cell_matrix_and_vector(U,V0,a(du,dv),l(dv),zh)
A1,b1 = assemble_matrix_and_vector(assem,data)

_a(u,v) = ∫( ∇(v)⋅∇(u) )dΩa
_l(v) = ∫( v )dΩa
_U = TrialFESpace(u([1.0]),V0)
_assem = SparseMatrixAssembler(_U,V0,das)
_zh = zero(_U)
_data = collect_cell_matrix_and_vector(_U,V0,_a(du,dv),_l(dv),_zh)
_A1,_b1 = assemble_matrix_and_vector(_assem,_data)

map(local_views(A1),local_views(_A1),local_views(b1),local_views(_b1)) do A,_A,b,_b
  for i = eachindex(A)
    @assert A[i] ≈ _A
    @assert b[i] ≈ _b
  end
end

x1 = A1\b1
r1 = A1*x1 -b1
uh1 = FEFunction(U,x1)
eh1 = uμ - uh1
@test all(sqrt.(sum(cont4)) .< 1.0e-9)
@test sqrt(sum(∫( abs2(eh1) )dΩ)) < 1.0e-9

map(A1.matrix_partition, A1.row_partition, A1.col_partition) do mat, rows, cols
    @test size(mat) == (local_length(rows),local_length(cols))
end

A2,b2 = allocate_matrix_and_vector(assem,data)
assemble_matrix_and_vector!(A2,b2,assem,data)
x2 = A2\b2
r2 = A2*x2 -b2
uh = FEFunction(U,x2)
eh2 = uμ - uh
sqrt(sum(∫( abs2(eh2) )dΩ)) < 1.0e-9

op = AffineFEOperator(a,l,U,V0,das)
solver = LinearFESolver(BackslashSolver())
uh = solve(solver,op)
eh = uμ - uh
@test sqrt(sum(∫( abs2(eh) )dΩ)) < 1.0e-9

data = collect_cell_matrix(U,V0,a(du,dv))
A3 = assemble_matrix(assem,data)
x3 = A3\op.op.vector
uh = FEFunction(U,x3)
eh3 = uμ - uh
sqrt(sum(∫( abs2(eh3) )dΩ)) < 1.0e-9

A4 = allocate_matrix(assem,data)
assemble_matrix!(A4,assem,data)
x4 = A4\op.op.vector
uh = FEFunction(U,x4)
eh4 = uμ - uh
sqrt(sum(∫( abs2(eh4) )dΩ)) < 1.0e-9

dv = get_fe_basis(V0)
l=∫(1*dv)dΩa
vecdata=collect_cell_vector(V0,l)
assem = SparseMatrixAssembler(U,V0,das)
b1=assemble_vector(assem,vecdata)
@test abs(sum(b1)-length(b1)) < 1.0e-12

b2=allocate_vector(assem,vecdata)
assemble_vector!(b2,assem,vecdata)
@test abs(sum(b2)-length(b2)) < 1.0e-12

m1 = nz_counter(get_matrix_builder(assemμ),(get_rows(assemμ),get_cols(assemμ)))
symbolic_loop_matrix!(m1,assemμ,matdata)
m2 = nz_allocation(m1)
symbolic_loop_matrix!(m2,assemμ,matdata)
# m3 = create_from_nz(m2)

I,J,V = GridapDistributed.get_allocations(m2)
plength = param_length(m2)
test_dofs_gids_prange = GridapDistributed.get_test_gids(m2)
trial_dofs_gids_prange = GridapDistributed.get_trial_gids(m2)

# convert I and J to global dof ids
GridapDistributed.to_global_indices!(I,test_dofs_gids_prange;ax=:rows)
GridapDistributed.to_global_indices!(J,trial_dofs_gids_prange;ax=:cols)

# Create the Prange for the rows
rows = GridapDistributed._setup_prange(test_dofs_gids_prange,I;ax=:rows)

# Move (I,J,V) triplets to the owner process of each row I.
# The triplets are accompanyed which Jo which is the process column owner
Jo = GridapDistributed.get_gid_owners(J,trial_dofs_gids_prange;ax=:cols)
t  = Distributed._assemble_param_coo!(I,J,V,rows,plength;owners=Jo)

# Wait the transfer to finish
wait(t)

# Create the Prange for the cols
cols = GridapDistributed._setup_prange(trial_dofs_gids_prange,J;ax=:cols,owners=Jo)

# Convert again I,J to local numeration
GridapDistributed.to_local_indices!(I,rows;ax=:rows)
GridapDistributed.to_local_indices!(J,cols;ax=:cols)

# Adjust local matrix size to linear system's index sets
asys = GridapDistributed.change_axes(m2,(rows,cols))

# Compress the local matrices
values = map(create_from_nz,local_views(asys))

#

_matdata = collect_cell_matrix(Uμ,V0,∫(du*v)dΩa)
_m1 = nz_counter(get_matrix_builder(assem),(get_rows(assem),get_cols(assem)))
symbolic_loop_matrix!(_m1,assem,_matdata)
_m2 = nz_allocation(_m1)
symbolic_loop_matrix!(_m2,assem,_matdata)

_I,_J,_V = GridapDistributed.get_allocations(_m2)

# convert I and J to global dof ids
GridapDistributed.to_global_indices!(_I,test_dofs_gids_prange;ax=:rows)
GridapDistributed.to_global_indices!(_J,trial_dofs_gids_prange;ax=:cols)

_Jo = GridapDistributed.get_gid_owners(_J,trial_dofs_gids_prange;ax=:cols)
_t  = GridapDistributed._assemble_coo!(_I,_J,_V,rows;owners=_Jo)

# Wait the transfer to finish
wait(_t)

# Convert again I,J to local numeration
GridapDistributed.to_local_indices!(_I,rows;ax=:rows)
GridapDistributed.to_local_indices!(_J,cols;ax=:cols)

# Adjust local matrix size to linear system's index sets
_asys = GridapDistributed.change_axes(_m2,(rows,cols))

# Compress the local matrices
_values = map(create_from_nz,local_views(_asys))

T = SparseMatrixCSC{Float64,Int}

# create_from_nz(asys.allocs.items[2])
alloc = asys.allocs.items[2]
m,n = map(length,alloc.counter.axes)
finalize_coo!(T,alloc.I,alloc.J,alloc.V,m,n)
M = reshape(alloc.V,:,alloc.plength)
sparse_from_coo(T,alloc.I,alloc.J,M,m,n)

# create_from_nz(_asys.allocs.items[2])
_alloc = _asys.allocs.items[2]
_m,_n = map(length,_alloc.counter.axes)
finalize_coo!(T,_alloc.I,_alloc.J,_alloc.V,_m,_n)
sparse_from_coo(T,_alloc.I,_alloc.J,_alloc.V,_m,_n)

row_partition = partition(rows)
Jown = Jo

function setup_snd(part,parts_snd,row_lids,plength,coo_entries_with_column_owner)
  global_to_local_row = global_to_local(row_lids)
  local_row_to_owner = local_to_owner(row_lids)
  owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
  ptrs = zeros(Int32,length(parts_snd)+1)
  k_gi,k_gj,k_jo,k_v = coo_entries_with_column_owner
  for k in 1:length(k_gi)
    gi = k_gi[k]
    li = global_to_local_row[gi]
    owner = local_row_to_owner[li]
    if owner != part
      ptrs[owner_to_i[owner]+1] +=1
    end
  end
  PartitionedArrays.length_to_ptrs!(ptrs)
  gi_snd_data = zeros(eltype(k_gi),ptrs[end]-1)
  gj_snd_data = zeros(eltype(k_gj),ptrs[end]-1)
  jo_snd_data = zeros(eltype(k_jo),ptrs[end]-1)
  v_snd_data = zeros(eltype(k_v),ptrs[end]-1,plength)
  δ = Int(length(v_snd_data)/plength)
  for k in 1:length(k_gi)
    gi = k_gi[k]
    li = global_to_local_row[gi]
    owner = local_row_to_owner[li]
    if owner != part
      gj = k_gj[k]
      p = ptrs[owner_to_i[owner]]
      gi_snd_data[p] = gi
      gj_snd_data[p] = gj
      jo_snd_data[p] = k_jo[k]
      for i = 1:plength
        v = k_v[k+(i-1)*δ]
        v_snd_data[p,i] = v
        k_v[k+(i-1)*δ] = zero(v)
      end
      ptrs[owner_to_i[owner]] += 1
    end
  end
  PartitionedArrays.rewind_ptrs!(ptrs)
  gi_snd = JaggedArray(gi_snd_data,ptrs)
  gj_snd = JaggedArray(gj_snd_data,ptrs)
  jo_snd = JaggedArray(jo_snd_data,ptrs)
  v_snd = JaggedArray(v_snd_data,ptrs)
  gi_snd,gj_snd,jo_snd,v_snd
end

function setup_rcv!(coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv,plength)
  k_gi,k_gj,k_jo,k_v = coo_entries_with_column_owner
  current_n = length(k_gi)
  new_n = current_n + length(gi_rcv.data)
  δ = Distributed._get_delta(v_rcv)
  resize!(k_gi,new_n)
  resize!(k_gj,new_n)
  resize!(k_jo,new_n)
  resize!(k_v,new_n*plength)
  for p in 1:length(gi_rcv.data)
    k_gi[current_n+p] = gi_rcv.data[p]
    k_gj[current_n+p] = gj_rcv.data[p]
    k_jo[current_n+p] = jo_rcv.data[p]
    for i in 1:plength
      k_v[current_n+p+(i-1)*new_n] = v_rcv.data[p+(i-1)*δ]
    end
  end
end
part = linear_indices(row_partition)
parts_snd,parts_rcv = assembly_neighbors(row_partition)
coo_entries_with_column_owner = map(tuple,I,J,Jown,V)
gi_snd,gj_snd,jo_snd,v_snd = map(setup_snd,part,parts_snd,row_partition,plength,coo_entries_with_column_owner) |> tuple_of_arrays
graph = ExchangeGraph(parts_snd,parts_rcv)
t1 = exchange(gi_snd,graph)
t2 = exchange(gj_snd,graph)
t3 = exchange(jo_snd,graph)
t4 = exchange(v_snd,graph)

gi_rcv = fetch(t1)
gj_rcv = fetch(t2)
jo_rcv = fetch(t3)
v_rcv = fetch(t4)
map(setup_rcv!,coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv,plength)
I,J,Jown,V
