PartitionedArrays.partition(a::Projection) = partition(get_basis(a))
PartitionedArrays.local_values(a::Projection) = local_values(get_basis(a))
PartitionedArrays.own_values(a::Projection) = own_values(get_basis(a))
PartitionedArrays.ghost_values(a::Projection) = ghost_values(get_basis(a))
PartitionedArrays.consistent!(a::Projection) = consistent!(get_basis(a))

row_partition(a::Projection) = row_partition(get_basis(a))
col_partition(a::Projection) = col_partition(get_basis(a))
flat_row_partition(a::Projection) = flat_row_partition(get_basis(a))

for f in (:project!,:inv_project!)
  @eval begin
    function RBSteady.$f(
      y::Union{BlockArray,BlockParamArray},
      a::BlockProjection,
      x::BlockPArray
      )

      for i in eachindex(a)
        $f(blocks(y)[i],a[i],blocks(x)[i])
      end
    end

    function RBSteady.$f(
      y::BlockPArray,
      a::BlockProjection,
      x::Union{BlockArray,BlockParamArray}
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

function Algebra.allocate_vector(::Type{<:PVector{V}},n::Integer) where V
  allocate_vector(V,n)
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

function GridapDistributed.local_views(a::BlockProjection)
  array = map(local_views,a.array) |> to_parray_of_arrays
  map(BlockProjection,array)
end

const DistributedKroneckerProjection{A<:Union{DistributedProjection,DistributedNormedProjection},B<:Projection} = KroneckerProjection{A,B}

"""
    struct DistributedTTSVDProjection{A<:Projection,B<:AbstractArray{<:Any,3}} <: Projection
      projection_space::A
      core_time::B
    end

TT-SVD of distributed transient snapshots, without tensor product quantities: a
row-partitioned spatial projection, and a temporal core replicated on every rank.
The reduced dimension is the rank of `core_time`
"""
struct DistributedTTSVDProjection{A<:Projection,B<:AbstractArray{<:Any,3}} <: Projection
  projection_space::A
  core_time::B
end

const DistributedSequentialProjection{A<:DistributedTTSVDProjection} = SequentialProjection{A}

const AbstractDistributedProjection = Union{
  DistributedProjection,
  DistributedNormedProjection,
  DistributedKroneckerProjection,
  DistributedSequentialProjection
}

function _ttsvd_projection(red::SequentialReduction,s::DistributedTransientSnapshots,args...)
  red_style = ReductionStyle(red)
  basis,Sr,Vr = tpod(red_style[1],RBTransient.first_unfold(s),args...)
  remainder = reshape(Diagonal(Sr)*Vr',length(Sr),num_params(s),num_times(s))
  cores = Array{eltype(remainder),3}[]
  RBTransient.add_temporal_core!(cores,red_style[end],remainder)
  GenericProjection(basis,get_dof_map(s)),only(cores)
end

function RBSteady.projection(red::SequentialReduction,s::DistributedTransientSnapshots)
  ps,core_time = _ttsvd_projection(red,s)
  SequentialProjection(DistributedTTSVDProjection(ps,core_time))
end

function RBSteady.projection(red::SequentialReduction,s::DistributedTransientSnapshots,X::MatrixOrTensor)
  ps,core_time = _ttsvd_projection(red,s,X)
  SequentialProjection(DistributedTTSVDProjection(NormedProjection(ps,X),core_time))
end

function GridapDistributed.local_views(a::DistributedTTSVDProjection)
  map(local_views(a.projection_space)) do projection_space
    DistributedTTSVDProjection(projection_space,a.core_time)
  end
end

function GridapDistributed.local_views(a::DistributedSequentialProjection)
  map(SequentialProjection,local_views(a.projection))
end

# the basis of the spatial projection: the space-time one is never assembled
RBSteady.get_basis(a::DistributedTTSVDProjection) = get_basis(a.projection_space)
RBSteady.get_cores(a::DistributedTTSVDProjection) = (get_basis(a),a.core_time)
RBSteady.get_norm_matrix(a::DistributedTTSVDProjection) = get_norm_matrix(a.projection_space)
DofMaps.get_dof_map(a::DistributedTTSVDProjection) = get_dof_map(a.projection_space)
RBSteady.num_reduced_dofs(a::DistributedTTSVDProjection) = size(a.core_time,3)
RBSteady.isnull(a::DistributedTTSVDProjection) = num_reduced_dofs(a) == 0 || isnull(a.projection_space)

function RBSteady.projection_eltype(a::DistributedTTSVDProjection)
  promote_type(RBSteady.projection_eltype(a.projection_space),eltype(a.core_time))
end

RBTransient.get_projection_space(a::DistributedSequentialProjection) = a.projection.projection_space
RBTransient.get_basis_space(a::DistributedSequentialProjection) = get_basis(a.projection)
ParamDataStructures.num_space_dofs(a::DistributedSequentialProjection) = size(get_basis(a.projection),1)
RBTransient.fe_space_dof_ids(a::DistributedSequentialProjection) = axes(get_basis(a.projection),1)

_flat_core_time(a::DistributedSequentialProjection) = _flat_core_time(a.projection)
_flat_core_time(a::DistributedTTSVDProjection) = reshape(a.core_time,:,size(a.core_time,3))

function RBSteady.project!(
  x̂::ConsecutiveParamVector,
  a::DistributedSequentialProjection,
  x::PVector
  )

  Nt = num_times(a)
  Np = param_length(x̂)
  @check Int(param_length(x) / Np) == Nt
  core = _flat_core_time(a)
  X̂s = get_all_data(project(RBTransient.get_projection_space(a),x))
  @inbounds @views for ip in param_eachindex(x̂)
    mul!(x̂.data[:,ip],transpose(core),vec(X̂s[:,ip:Np:Np*Nt]))
  end
  x̂
end

function RBSteady.inv_project!(
  x::AbstractVector{<:Number},
  a::DistributedTTSVDProjection,
  x̂::AbstractVector{<:Number}
  )

  basis = get_basis(a)
  X̂s = reshape(_flat_core_time(a)*x̂,size(basis,2),:)
  mul!(reshape(x,size(basis,1),:),basis,X̂s)
  x
end

function Algebra.allocate_in_range(a::DistributedSequentialProjection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(PVector{eltype(V)},RBTransient.fe_space_dof_ids(a))
  return parameterise(x,param_length(x̂)*num_times(a))
end

function RBSteady.galerkin_projection(proj_left::DistributedTTSVDProjection,a::DistributedTTSVDProjection)
  Φl = get_basis(proj_left)
  Φa = get_basis(a)
  M = galerkin_projection(Φl,Φa) # ns_left x ns_a
  core_a = reshape(a.core_time,size(Φa,2),:)
  proj_core_a = reshape(M*core_a,:,num_reduced_dofs(a))
  ReducedProjection(transpose(_flat_core_time(proj_left))*proj_core_a)
end

function RBSteady.galerkin_projection(
  proj_left::DistributedSequentialProjection,
  a::DistributedSequentialProjection,
  proj_right::DistributedSequentialProjection,
  combine
  )

  # space
  p_space = galerkin_projection(
    get_basis_space(proj_left),
    get_recast_basis(RBTransient.get_projection_space(a)),
    get_basis_space(proj_right)
  ) # ns_left x ns_right x ns_a
  p_space = permutedims(p_space,(1,3,2))
  p_space = reshape(p_space,1,1,1,size(p_space)...)

  # time
  pl_time = RBTransient.get_core_time(proj_left)
  a_time = RBTransient.get_core_time(a)
  pr_time = RBTransient.get_core_time(proj_right)
  p_time = RBSteady.contraction(pl_time,a_time,pr_time,combine)

  p = RBSteady.sequential_product(p_space,p_time)
  proj_cores = dropdims(p;dims=(1,2,3)) # n_left x n_a x n_right
  ReducedProjection(permutedims(proj_cores,(1,3,2))) # n_left x n_right x n_a
end

function GridapDistributed.local_views(a::DistributedKroneckerProjection)
  map(local_views(a.projection_space)) do projection_space
    KroneckerProjection(projection_space,a.projection_time)
  end
end

RBTransient.fe_space_dof_ids(a::DistributedKroneckerProjection) = axes(get_basis_space(a),1)

function RBSteady.project!(
  x̂::ConsecutiveParamVector,
  a::DistributedKroneckerProjection,
  x::PVector
  )

  Nt = num_times(a)
  Np = param_length(x̂)
  ns = num_reduced_dofs(a.projection_space)
  nt = num_reduced_dofs(a.projection_time)
  @check Int(param_length(x) / Np) == Nt

  x̂s = project(a.projection_space,x)
  X̂s = change_mode(get_all_data(x̂s),Np)
  @inbounds @views for ip in param_eachindex(x̂)
    isp = (ip-1)*ns + 1 : ip*ns
    X̂sp = X̂s[:,isp]
    x̂p = reshape(x̂.data[:,ip],ns,nt)
    mul!(transpose(x̂p),transpose(get_basis_time(a)),X̂sp)
  end

  x̂
end

function RBSteady.inv_project!(
  x::PVector{<:ConsecutiveParamVector},
  a::TransientProjection,
  x̂::ConsecutiveParamVector
  )

  map(local_values(x),local_views(a)) do x,a
    inv_project!(x,a,x̂)
  end
  x
end

function Algebra.allocate_in_domain(a::TransientProjection,x::PVector{<:V}) where V<:AbstractParamVector
  x̂ = allocate_vector(PVector{eltype(V)},RBSteady.reduced_dof_ids(a))
  nt = num_times(a)
  np = Int(param_length(x) / nt)
  return parameterise(x̂,np)
end

function Algebra.allocate_in_range(a::DistributedKroneckerProjection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(PVector{eltype(V)},RBSteady.fe_dof_ids(a.projection_space))
  nt = num_times(a)
  npt = param_length(x̂) * nt
  return parameterise(x,npt)
end

function RBTransient.allocate_in_space_domain(a::TransientProjection,x::PVector{<:V}) where V<:AbstractParamVector
  x̂ = allocate_vector(PVector{eltype(V)},RBTransient.num_reduced_dofs_space(a))
  return parameterise(x̂,param_length(x))
end

function RBTransient.allocate_in_space_range(a::TransientProjection,x̂::PVector{<:V}) where V<:AbstractParamVector
  x = allocate_vector(PVector{eltype(V)},RBTransient.num_fe_dofs_space(a))
  return parameterise(x,param_length(x̂))
end

for f in (:allocate_in_space_domain,:allocate_in_space_range)
  @eval begin
    function RBTransient.$f(a::BlockProjection,x::BlockPVector)
      @check length(a) == blocklength(x)
      mortar(map(i -> RBTransient.$f(a[Block(i)],x[Block(i)]),eachindex(a)))
    end
  end
end

for f in (:project_space!,:inv_project_space!)
  @eval begin
    function RBTransient.$f(
      y::Union{BlockArray,BlockParamArray},
      a::BlockProjection,
      x::BlockPArray
      )

      for i in eachindex(a)
        RBTransient.$f(blocks(y)[i],a[i],blocks(x)[i])
      end
    end
  end
end

function RBSteady.union_bases(a::DistributedPODProjection,basis_b::GenericPMatrix,args...)
  rows = axes(get_basis(a),1)
  basis_b′ = change_ghost(basis_b,rows;make_consistent=true)
  basis_a = get_basis(a)
  basis_ab = gram_schmidt(hcat(basis_b′,basis_a),args...)
  PODProjection(basis_ab,get_dof_map(a))
end

function RBSteady.GalerkinProjectable(a::A) where A<:PVector
  GalerkinProjectable{A}(a)
end

function RBSteady.GalerkinProjectable(a::A) where A<:PSparseMatrix
  GalerkinProjectable{A}(a)
end

function RBSteady.GalerkinProjectable(a::BlockPArray)
  block_cache = map(GalerkinProjectable,blocks(a))
  return BlockProjection(block_cache)
end