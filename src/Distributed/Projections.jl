abstract type DistributedProjection <: Projection end

PartitionedArrays.partition(a::DistributedProjection) = partition(get_basis(a))
PartitionedArrays.local_values(a::DistributedProjection) = local_values(get_basis(a))
PartitionedArrays.own_values(a::DistributedProjection) = own_values(get_basis(a))
PartitionedArrays.ghost_values(a::DistributedProjection) = ghost_values(get_basis(a))
PartitionedArrays.consistent!(a::DistributedProjection) = consistent!(get_basis(a))

function RBSteady.galerkin_projection(Φl::GenericPMatrix,b::PVector)
  map(own_values(Φl),own_values(b)) do Φlo,bo
    galerkin_projection(Φlo,bo)
  end |> sreduce
end

function RBSteady.galerkin_projection(Φl::GenericPMatrix,A::PSparseMatrix,Φr::GenericPMatrix)
  TS = promote_type(eltype(Φl),eltype(Φr))
  nleft = size(Φl,2)
  n = getany(map(param_length,partition(A)))
  nright = size(Φr,2)
  Â = zeros(TS,nleft,nright,n)
  _galerkin_mul!(Â,Φl,A,Φr)
  return Â
end

function RBSteady.galerkin_projection(a::DistributedProjection,s::DistributedSnapshots)
  b̂ = galerkin_projection(get_basis(a),get_param_data(s))
  return ReducedProjection(b̂)
end

function RBSteady.galerkin_projection(a::DistributedProjection,s::DistributedSnapshots,c::DistributedProjection,args...)
  b̂ = galerkin_projection(get_basis(a),get_param_data(s),get_basis(c),args...)
  return ReducedProjection(b̂)
end

row_partition(a::DistributedProjection) = row_partition(get_basis(a))
col_partition(a::DistributedProjection) = col_partition(get_basis(a))
flat_row_partition(a::DistributedProjection) = flat_row_partition(get_basis(a))

RBSteady.fe_dof_ids(a::DistributedProjection) = axes(get_basis(a),1)

RBSteady.projection_type(a::DistributedProjection) = PVector{Vector{projection_eltype(a)}}

function Algebra.allocate_vector(::Type{<:PVector{V}},rows::AbstractVector) where V
  allocate_vector(V,rows)
end

function Algebra.allocate_in_domain(a::DistributedProjection,x::PVector{<:V}) where V<:AbstractParamVector
  x̂ = allocate_vector(PVector{eltype(V)},RBSteady.reduced_dof_ids(a))
  return parameterise(x̂,param_length(x))
end

function Algebra.allocate_in_range(a::DistributedProjection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(PVector{eltype(V)},RBSteady.fe_dof_ids(a))
  return parameterise(x,param_length(x̂))
end

function RBSteady.allocate_full_matrix(::Type{<:GenericPArray{M}},rows::PRange,cols::AbstractVector) where M
  GenericPArray{M}(undef,partition(rows),cols)
end

function RBSteady._allocate_projection(red::Reduction,s::DistributedBlockSnapshots{<:Any,N},args...) where N
  T = _distr_proj_type(red)
  block_basis = Array{T,N}(undef,size(s))
  BlockProjection(block_basis)
end

function RBTransient.kron_projection(red::KroneckerReduction,s::DistributedSparseSnapshots,args...)
  basis_space,basis_time = tucker(red.reductions,s,args...)
  basis_space′ = recast(basis_space,s)
  projection_space = PODProjection(basis_space′)
  projection_time = PODProjection(basis_time)
  return projection_space,projection_time
end

struct DistributedPODProjection <: DistributedProjection
  basis::AbstractMatrix
end

function RBSteady.PODProjection(basis::GenericPMatrix)
  DistributedPODProjection(basis)
end

function RBSteady.Projection(basis::GenericPMatrix,s::DistributedSparseSnapshots)
  basis′ = recast(basis,s)
  DistributedPODProjection(basis′)
end

function RBSteady.Projection(basis::GenericPMatrix,s::DistributedSnapshots)
  DistributedPODProjection(basis)
end

RBSteady.get_basis(a::DistributedPODProjection) = a.basis

function RBSteady.union_bases(a::DistributedPODProjection,b::DistributedPODProjection,args...) 
  union_bases(a,get_basis(b),args...)
end

function RBSteady.union_bases(a::DistributedPODProjection,basis_b::AbstractMatrix,args...)
  basis_a = get_basis(a)
  basis_ab = gram_schmidt(basis_b,basis_a,args...)
  DistributedPODProjection(basis_ab)
end

function GridapDistributed.local_views(a::DistributedPODProjection)
  map(local_views(get_basis(a))) do basis
    PODProjection(basis)
  end
end

struct DistributedNormedProjection <: DistributedProjection
  projection::DistributedProjection
  norm_matrix::PSparseMatrix
end

function RBSteady.NormedProjection(a::DistributedProjection,norm_matrix::PSparseMatrix)
  DistributedNormedProjection(a,norm_matrix)
end

RBSteady.get_projection(a::DistributedNormedProjection) = a.projection
RBSteady.get_norm_matrix(a::DistributedNormedProjection) = a.norm_matrix

RBSteady.get_basis(a::DistributedNormedProjection) = get_basis(a.projection)
RBSteady.num_fe_dofs(a::DistributedNormedProjection) = num_fe_dofs(a.projection)
RBSteady.num_reduced_dofs(a::DistributedNormedProjection) = num_reduced_dofs(a.projection)
RBSteady.projection_type(a::DistributedNormedProjection) = projection_type(a.projection)

function RBSteady.project!(x̂::AbstractArray,a::DistributedNormedProjection,x::AbstractArray)
  project!(x̂,a.projection,x,a.norm_matrix)
end

function RBSteady.inv_project!(x::AbstractArray,a::DistributedNormedProjection,x̂::AbstractArray)
  inv_project!(x,a.projection,x̂)
end

function RBSteady.union_bases(a::DistributedNormedProjection,b::DistributedNormedProjection,args...)
  projection′ = union_bases(a.projection,b.projection,args...)
  DistributedNormedProjection(projection′,a.norm_matrix)
end

function RBSteady.union_bases(a::DistributedNormedProjection,b::AbstractArray,args...)
  projection′ = union_bases(a.projection,b,args...)
  DistributedNormedProjection(projection′,a.norm_matrix)
end

function RBSteady.galerkin_projection(proj_left::DistributedNormedProjection,a::DistributedProjection)
  galerkin_projection(RBSteady.get_projection(proj_left),RBSteady.get_projection(a))
end

function RBSteady.galerkin_projection(
  proj_left::DistributedNormedProjection,
  a::DistributedProjection,
  proj_right::DistributedNormedProjection,
  args...
  )

  galerkin_projection(RBSteady.get_projection(proj_left),RBSteady.get_projection(a),RBSteady.get_projection(proj_right),args...)
end

for f in (:DEIM,:SOPT)
  @eval begin
    RBSteady.$f(a::DistributedNormedProjection) = $f(a.projection)
  end
end

function GridapDistributed.local_views(a::DistributedNormedProjection)
  map(local_views(a.projection),local_views(a.norm_matrix)) do projection,matrix
    NormedProjection(projection,matrix)
  end
end

function RBSteady.gram_solver(X::PSparseMatrix)
  solver = CGSolver(JacobiLinearSolver();maxiter=100,atol=1e-14,rtol=1e-10)
  ss = symbolic_setup(solver,X)
  numerical_setup(ss,X)
end

function LinearAlgebra.ldiv!(S::GenericPMatrix,ns,A::GenericPMatrix)
  for i in param_eachindex(S)
    Si = param_getindex(S,i)
    Ai = param_getindex(A,i)
    solve!(Si,ns,Ai)
    # consistent!(Si) |> wait
  end
end

function RBSteady.gram_schmidt(A::AbstractMatrix,ns::LinearSolvers.CGNumericalSetup,args...)
  gram_schmidt(A,ns.mat,args...)
end

function GridapDistributed.local_views(a::KroneckerProjection)
  map(local_views(a.projection_space)) do projection_space
    KroneckerProjection(projection_space,a.projection_time)
  end
end

# utils 

_distr_proj_type(red::Reduction) = _distr_proj_type(NormStyle(red),red)
_distr_proj_type(::NormStyle,::Reduction) = @abstractmethod
_distr_proj_type(::EuclideanNorm,::PODReduction) = DistributedPODProjection
_distr_proj_type(::AssembleOperator,::DirectReduction) = DistributedNormedProjection

function _galerkin_mul!(
  d::AbstractArray{<:Number,3},
  c::GenericPArray,
  a::PSparseMatrix,
  b::GenericPArray
  )

  @boundscheck @assert PartitionedArrays.matching_own_indices(axes(c,1),axes(a,1))
  @boundscheck @assert PartitionedArrays.matching_own_indices(axes(a,2),axes(b,1))
  if !PartitionedArrays.matching_ghost_indices(axes(a,2),axes(b,1))
    b = _change_layout(b,partition(axes(a,2)))
  end
  # Start the exchange
  t = consistent!(b)
  # Meanwhile, process the owned block into a per-rank local buffer.
  ld = map(own_values(c),own_values(a),own_values(b)) do co,aoo,bo
    dl = zeros(eltype(d),size(d))
    co1 = zeros(eltype(d),innersize(aoo)[1],size(bo,2))
    @inbounds for i in param_eachindex(aoo)
      mul!(co1,param_getindex(aoo,i),bo)
      mul!(view(dl,:,:,i),co',co1)
    end
    dl
  end
  # Wait for the exchange to finish
  wait(t)
  # process the ghost block, accumulating onto the same per-rank buffer
  map(ld,own_values(c),own_ghost_values(a),ghost_values(b)) do dl,co,aoh,bh
    co1 = zeros(eltype(d),innersize(aoh)[1],size(bh,2))
    @inbounds for i in param_eachindex(aoh)
      mul!(co1,param_getindex(aoh,i),bh)
      mul!(view(dl,:,:,i),co',co1,1,1)
    end
  end
  copyto!(d,sreduce(ld))
  d
end