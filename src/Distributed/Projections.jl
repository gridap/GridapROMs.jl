PartitionedArrays.partition(a::Projection) = partition(get_basis(a))
PartitionedArrays.local_values(a::Projection) = local_values(get_basis(a))
PartitionedArrays.own_values(a::Projection) = own_values(get_basis(a))
PartitionedArrays.ghost_values(a::Projection) = ghost_values(get_basis(a))
PartitionedArrays.consistent!(a::Projection) = consistent!(get_basis(a))

row_partition(a::Projection) = row_partition(get_basis(a))
col_partition(a::Projection) = col_partition(get_basis(a))
flat_row_partition(a::Projection) = flat_row_partition(get_basis(a))

RBSteady.to_fe_blocks(x::BlockPArray,a::BlockProjection,args...) = x
RBSteady.to_reduced_blocks(x::BlockPArray,a::BlockProjection,args...) = x

function RBSteady.to_blocks(x::PVector,o,f=identity)
  n = length(o)-1
  map(1:n) do i
    vector_partition = map(partition(x)) do values
      f(view(x,o[i]:o[i+1]-1))
    end
    PVector(vector_partition,row_partition(x))
  end |> mortar
end

function RBSteady.to_blocks(x::PVector{<:AbstractParamVector},o,f=identity)
  n = length(o)-1
  map(1:n) do i
    vector_partition = map(partition(x)) do values
      f(get_param_entry(x,o[i]:o[i+1]-1))
    end
    PVector(vector_partition,row_partition(x))
  end |> mortar
end

for f in (:project!,:inv_project!)
  @eval begin
    function $f(
      y::Union{BlockArray,BlockParamArray},
      a::BlockProjection,
      x::BlockPArray
      )

      for i in eachindex(a)
        $f(blocks(y)[i],a[i],blocks(x)[i])
      end
    end
  end
end

function Algebra.allocate_in_domain(a::BlockProjection,x::BlockPArray)
  map(Algebra.allocate_in_domain,a.array,blocks(x)) |> mortar
end

const DistributedProjection{A<:AbstractArray,B<:AbstractArray{<:AbstractDofMap}} = GenericProjection{A,B}
const DistributedPODProjection{A<:GenericPMatrix,B<:AbstractArray{<:AbstractDofMap}} = DistributedProjection{A,B}
const DistributedTTSVDProjection{A<:AbstractArray{<:GenericPArray},B<:AbstractArray{<:AbstractDofMap}} = DistributedProjection{A,B}

function GridapDistributed.local_views(a::DistributedProjection)
  map(local_views(get_basis(a)),local_views(get_dof_map(a))) do basis,dof_map
    GenericProjection(basis,dof_map)
  end
end

RBSteady.fe_dof_ids(a::DistributedProjection) = axes(get_basis(a),1)

RBSteady.projection_type(a::DistributedProjection) = PVector{Vector{projection_eltype(a)}}

function Algebra.allocate_vector(::Type{<:PVector{V}},rows::AbstractVector) where V
  allocate_vector(V,rows)
end

function Algebra.allocate_vector(::Type{<:BlockPArray{V}},rows::AbstractVector) where V
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

RBSteady.fe_dof_ids(a::DistributedNormedProjection) = axes(get_basis(a),1)

RBSteady.projection_type(a::DistributedNormedProjection) = PVector{Vector{projection_eltype(a)}}

function Algebra.allocate_in_domain(a::DistributedNormedProjection,x::PVector{<:V}) where V<:AbstractParamVector
  x̂ = allocate_vector(PVector{eltype(V)},RBSteady.reduced_dof_ids(a))
  return parameterise(x̂,param_length(x))
end

function Algebra.allocate_in_range(a::DistributedNormedProjection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(PVector{eltype(V)},RBSteady.fe_dof_ids(a))
  return parameterise(x,param_length(x̂))
end

function GridapDistributed.local_views(a::DistributedNormedProjection)
  map(local_views(a.projection),local_views(a.norm_matrix)) do projection,matrix
    NormedProjection(projection,matrix)
  end
end

const DistributedKroneckerProjection{A<:Union{DistributedProjection,DistributedNormedProjection},B<:Projection} = KroneckerProjection{A,B}

function GridapDistributed.local_views(a::DistributedKroneckerProjection)
  map(local_views(a.projection_space)) do projection_space
    KroneckerProjection(projection_space,a.projection_time)
  end
end

function RBSteady.GalerkinProjectable(a::A) where A<:AbstractParamPVector
  GalerkinProjectable{A}(a)
end

function RBSteady.GalerkinProjectable(a::A) where A<:AbstractParamPSparseMatrix
  GalerkinProjectable{A}(a)
end

function RBSteady.GalerkinProjectable(a::BlockPArray)
  block_cache = map(GalerkinProjectable,blocks(a))
  return BlockProjection(block_cache)
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