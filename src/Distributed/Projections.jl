abstract type DistributedProjection <: Projection end

PartitionedArrays.partition(a::DistributedProjection) = partition(get_basis(a))
PartitionedArrays.local_values(a::DistributedProjection) = local_values(get_basis(a))
PartitionedArrays.own_values(a::DistributedProjection) = own_values(get_basis(a))
PartitionedArrays.ghost_values(a::DistributedProjection) = ghost_values(get_basis(a))
PartitionedArrays.consistent!(a::DistributedProjection) = consistent!(get_basis(a))

function RBSteady.galerkin_projection(a::DistributedProjection,b::DistributedProjection)
  lb̂ = map(own_values(a),own_values(b)) do ao,bo
    galerkin_projection(ao,bo)
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

function RBSteady.galerkin_projection(a::DistributedProjection,b::DistributedProjection,c::DistributedProjection,args...)
  lb̂ = map(own_values(a),own_values(b),own_values(c)) do ao,bo,co
    galerkin_projection(ao,bo,co,args...)
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

row_partition(a::DistributedProjection) = row_partition(get_basis(a))
col_partition(a::DistributedProjection) = col_partition(get_basis(a))
flat_row_partition(a::DistributedProjection) = flat_row_partition(get_basis(a))

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

function RBSteady._allocate_projection(red::Reduction,s::DistributedBlockSnapshots{N}) where N
  T = _distr_proj_type(red)
  block_basis = Array{T,N}(undef,size(s))
  BlockProjection(block_basis,s.touched)
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

RBSteady.get_basis(a::DistributedPODProjection) = a.basis
RBSteady.fe_dof_ids(a::DistributedPODProjection) = row_partition(a)

function RBSteady.union_bases(a::DistributedPODProjection,b::DistributedPODProjection,args...) 
  union_bases(a,get_basis(b),args...)
end

function RBSteady.union_bases(a::DistributedPODProjection,basis_b::AbstractMatrix,args...)
  basis_a = get_basis(a)
  basis_ab = gram_schmidt(basis_b,basis_a,args...)
  DistributedPODProjection(basis_ab)
end

function GridapDistributed.local_views(a::DistributedPODProjection)
  map(local_views(a.basis)) do basis
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

function RBSteady.galerkin_projection(proj_left::DistributedNormedProjection,a::Projection)
  galerkin_projection(RBSteady.get_projection(proj_left),RBSteady.get_projection(a))
end

function RBSteady.galerkin_projection(
  proj_left::DistributedNormedProjection,
  a::Projection,
  proj_right::DistributedNormedProjection,
  args...
  )

  galerkin_projection(
    RBSteady.get_projection(proj_left),
    RBSteady.get_projection(a),
    RBSteady.get_projection(proj_right),
    args...
  )
end

for f in (:DEIM,:SOPT)
  @eval begin
    RBSteady.$f(a::DistributedNormedProjection) = $f(a.projection)
  end
end

# utils 

_distr_proj_type(red::Reduction) = _distr_proj_type(NormStyle(red),red)
_distr_proj_type(::NormStyle,::Reduction) = @abstractmethod
_distr_proj_type(::EuclideanNorm,::PODReduction) = DistributedPODProjection
_distr_proj_type(::AssembleOperator,::DirectReduction) = DistributedNormedProjection