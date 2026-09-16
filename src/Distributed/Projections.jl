const DistributedProjection{A<:AbstractArray,B<:AbstractArray{<:AbstractDofMap}} = GenericProjection{A,B}
const DistributedPODProjection{A<:GenericPMatrix,B<:AbstractArray{<:AbstractDofMap}} = DistributedProjection{A,B}
const DistributedTTSVDProjection{A<:AbstractArray{<:GenericPArray},B<:AbstractArray{<:AbstractDofMap}} = DistributedProjection{A,B}

PartitionedArrays.partition(a::DistributedProjection) = partition(get_basis(a))
PartitionedArrays.local_values(a::DistributedProjection) = local_values(get_basis(a))
PartitionedArrays.own_values(a::DistributedProjection) = own_values(get_basis(a))
PartitionedArrays.ghost_values(a::DistributedProjection) = ghost_values(get_basis(a))
PartitionedArrays.consistent!(a::DistributedProjection) = consistent!(get_basis(a))

function GridapDistributed.local_views(a::DistributedProjection)
  map(local_views(get_basis(a)),local_views(get_dof_map(a))) do basis,dof_map
    GenericProjection(basis,dof_map)
  end
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

const DistributedNormedProjection{A<:DistributedProjection,B<:MatrixOrTensor} = NormedProjection{A,B}

function GridapDistributed.local_views(a::DistributedNormedProjection)
  map(local_views(a.projection),local_views(a.norm_matrix)) do projection,matrix
    NormedProjection(projection,matrix)
  end
end

function GridapDistributed.local_views(a::KroneckerProjection)
  map(local_views(a.projection_space)) do projection_space
    KroneckerProjection(projection_space,a.projection_time)
  end
end

# utils

function RBSteady._allocate_projection(red::Reduction,s::DistributedBlockSnapshots{<:Any,N},args...) where N
  T = _distr_proj_type(red)
  block_basis = Array{T,N}(undef,size(s))
  BlockProjection(block_basis)
end

_distr_proj_type(red::Reduction) = _distr_proj_type(NormStyle(red),red)
_distr_proj_type(::NormStyle,::Reduction) = @abstractmethod
_distr_proj_type(::EuclideanNorm,::PODReduction) = DistributedPODProjection
_distr_proj_type(::AssembleOperator,::DirectReduction) = DistributedNormedProjection