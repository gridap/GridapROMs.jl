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
  x̂ = allocate_vector(PVector{eltype(V)},reduced_dof_ids(a))
  return parameterise(x̂,param_length(x))
end

function Algebra.allocate_in_range(a::DistributedProjection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(PVector{eltype(V)},fe_dof_ids(a))
  return parameterise(x,param_length(x̂))
end

function RBSteady.allocate_full_matrix(::Type{<:GenericPArray{M}},rows::PRange,cols::AbstractVector) where M
  GenericPArray{M}(undef,partition(rows),cols)
end

const DistributedNormedProjection{A<:DistributedProjection,B<:MatrixOrTensor} = NormedProjection{A,B}

RBSteady.fe_dof_ids(a::DistributedNormedProjection) = axes(get_basis(a),1)

RBSteady.projection_type(a::DistributedNormedProjection) = PVector{Vector{projection_eltype(a)}}

function Algebra.allocate_in_domain(a::DistributedNormedProjection,x::PVector{<:V}) where V<:AbstractParamVector
  x̂ = allocate_vector(PVector{eltype(V)},reduced_dof_ids(a))
  return parameterise(x̂,param_length(x))
end

function Algebra.allocate_in_range(a::DistributedNormedProjection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(PVector{eltype(V)},fe_dof_ids(a))
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

"""
    const DistributedTransientProjection{A<:Union{DistributedProjection,DistributedNormedProjection}} = TransientProjection{A}

[`TransientProjection`](@ref) of distributed snapshots: a row-partitioned spatial
projection, and a temporal array replicated on every rank
"""
const DistributedTransientProjection{A<:Union{DistributedProjection,DistributedNormedProjection}} = TransientProjection{A}
const DistributedKroneckerProjection{A<:Union{DistributedProjection,DistributedNormedProjection}} = KroneckerProjection{A}
const DistributedSequentialProjection{A<:Union{DistributedProjection,DistributedNormedProjection}} = SequentialProjection{A}

const AbstractDistributedProjection = Union{
  DistributedProjection,
  DistributedNormedProjection,
  DistributedTransientProjection
}

function GridapDistributed.local_views(a::DistributedTransientProjection)
  map(local_views(get_projection_space(a))) do projection_space
    TransientProjection(projection_space,get_basis_time(a))
  end
end

# the space-time basis is never assembled
RBSteady.get_basis(a::DistributedTransientProjection) = get_basis_space(a)

function Algebra.allocate_in_domain(a::DistributedTransientProjection,x::PVector{<:V}) where V<:AbstractParamVector
  x̂ = allocate_vector(PVector{eltype(V)},reduced_dof_ids(a))
  nt = num_times(a)
  np = Int(param_length(x) / nt)
  return parameterise(x̂,np)
end

function Algebra.allocate_in_range(a::DistributedTransientProjection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(PVector{eltype(V)},RBTransient.fe_dof_ids_space(a))
  nt = num_times(a)
  npt = param_length(x̂) * nt
  return parameterise(x,npt)
end

function RBSteady.inv_project!(
  x::PVector{<:ConsecutiveParamVector},
  a::DistributedTransientProjection,
  x̂::ConsecutiveParamVector
  )

  map(local_values(x),local_views(a)) do x,a
    inv_project!(x,a,x̂)
  end
  x
end

function RBTransient.allocate_in_space_domain(a::DistributedTransientProjection,x::PVector{<:V}) where V<:AbstractParamVector
  x̂ = allocate_vector(PVector{eltype(V)},RBTransient.num_reduced_dofs_space(a))
  return parameterise(x̂,param_length(x))
end

function RBTransient.allocate_in_space_range(a::DistributedTransientProjection,x̂::PVector{<:V}) where V<:AbstractParamVector
  x = allocate_vector(PVector{eltype(V)},RBTransient.num_fe_dofs_space(a))
  return parameterise(x,param_length(x̂))
end

function RBSteady.project!(
  x̂::ConsecutiveParamVector,
  a::DistributedKroneckerProjection,
  x::PVector
  )

  Nt = num_times(a)
  Np = param_length(x̂)
  ps = get_projection_space(a)
  ns = num_reduced_dofs(ps)
  nt = RBTransient.num_reduced_dofs_time(a)
  @check Int(param_length(x) / Np) == Nt

  x̂s = project(ps,x)
  X̂s = change_mode(get_all_data(x̂s),Np)
  @inbounds @views for ip in param_eachindex(x̂)
    isp = (ip-1)*ns + 1 : ip*ns
    X̂sp = X̂s[:,isp]
    x̂p = reshape(x̂.data[:,ip],ns,nt)
    mul!(transpose(x̂p),transpose(get_basis_time(a)),X̂sp)
  end

  x̂
end

function RBSteady.Projection(red::SequentialReduction,s::DistributedTransientSnapshots,args...)
  red_style = ReductionStyle(red)
  space_basis,Sr,Vr = tpod(red_style[1],RBTransient.first_unfold(s),args...)
  remainder = reshape(Diagonal(Sr)*Vr',length(Sr),num_params(s),num_times(s))
  time_core, = RBSteady.ttsvd_loop(red_style,permutedims(remainder,(1,3,2)))
  GenericProjection(space_basis,get_dof_map(s)),time_core
end

function RBSteady.project!(
  x̂::ConsecutiveParamVector,
  a::DistributedSequentialProjection,
  x::PVector
  )

  Nt = num_times(a)
  Np = param_length(x̂)
  @check Int(param_length(x) / Np) == Nt
  core = cores2basis(RBTransient.get_core_time(a))
  X̂s = get_all_data(project(get_projection_space(a),x))
  @inbounds @views for ip in param_eachindex(x̂)
    mul!(x̂.data[:,ip],transpose(core),vec(X̂s[:,ip:Np:Np*Nt]))
  end
  x̂
end

function RBTransient.space_contractions(
  proj_left::DistributedSequentialProjection,
  a::DistributedSequentialProjection,
  proj_right::DistributedSequentialProjection
  )

  p_space = galerkin_projection(
    get_basis_space(proj_left),
    get_recast_basis(get_projection_space(a)),
    get_basis_space(proj_right)
  ) # ns_left x ns_right x ns_a
  p_space = permutedims(p_space,(1,3,2))
  (reshape(p_space,1,1,1,size(p_space)...),)
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