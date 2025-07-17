#TODO Functionalities are not yet implemented for a general high order PDE; eventually,
# it would be desirable to have something like:
# abstract type HighOrderProjection <: Projection end
# abstract type TransientProjection <: HighOrderProjection end

abstract type TransientProjection <: Projection end

get_projection_space(a::TransientProjection) = @abstractmethod
get_projection_time(a::TransientProjection) = @abstractmethod
get_basis_space(a::TransientProjection) = get_basis(get_projection_space(a))
get_basis_time(a::TransientProjection) = get_basis(get_projection_time(a))
ParamDataStructures.num_space_dofs(a::TransientProjection) = num_fe_dofs(get_projection_space(a))
ParamDataStructures.num_times(a::TransientProjection) = num_fe_dofs(get_projection_time(a))
RBSteady.num_fe_dofs(a::TransientProjection) = num_space_dofs(a)*num_times(a)

function RBSteady.project!(x̂::ConsecutiveParamVector,a::TransientProjection,x::ConsecutiveParamVector)
  nt = num_times(a)
  @check Int(param_length(x) / param_length(x̂)) == nt
  np = param_length(x̂)
  @inbounds for ip in eachindex(x̂)
    ipt = ip:np:np*nt
    xpt = vec(view(x.data,:,ipt))
    x̂p = x̂[ip]
    project!(x̂p,a,xpt)
  end
end

function RBSteady.inv_project!(x::AbstractParamVector,a::TransientProjection,x̂::AbstractParamVector)
  nt = num_times(a)
  @check Int(param_length(x) / param_length(x̂)) == nt
  np = param_length(x̂)
  @inbounds for ip in eachindex(x̂)
    ipt = ip:np:np*nt
    xpt = vec(view(x.data,:,ipt))
    x̂p = x̂[ip]
    inv_project!(xpt,a,x̂p)
  end
end

function Algebra.allocate_in_domain(a::TransientProjection,x::V) where V<:AbstractParamVector
  x̂ = allocate_vector(eltype(V),num_reduced_dofs(a))
  nt = num_times(a)
  np = Int(param_length(x) / nt)
  return global_parameterize(x̂,np)
end

function Algebra.allocate_in_range(a::TransientProjection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(eltype(V),num_space_dofs(a))
  nt = num_times(a)
  npt = param_length(x̂) * nt
  return global_parameterize(x,npt)
end

"""
    struct KroneckerProjection <: TransientProjection
      projection_space::Projection
      projection_time::Projection
    end

Projection operator for transient problems, containing a spatial projection and
a temporal one. The space-time projection operator is equal to

`projection_time ⊗ projection_space`

which, for efficiency reasons, is never explicitly computed
"""
struct KroneckerProjection <: TransientProjection
  projection_space::Projection
  projection_time::Projection
end

to_proj(basis::AbstractMatrix) = PODProjection(basis)
to_proj(basis::AbstractMatrix,X::AbstractSparseArray) = NormedProjection(to_proj(basis),X)

function KroneckerProjection(red::KroneckerReduction,s::TransientSnapshots,args...)
  basis_space,basis_time = tucker(red.reductions,s,args...)
  projection_space = to_proj(basis_space,args...)
  projection_time = to_proj(basis_time)
  KroneckerProjection(projection_space,projection_time)
end

function KroneckerProjection(red::KroneckerReduction,s::TransientSparseSnapshots,args...)
  basis_space,basis_time = tucker(red.reductions,s,args...)
  basis_space′ = recast(basis_space,s)
  projection_space = to_proj(basis_space′,args...)
  projection_time = to_proj(basis_time)
  KroneckerProjection(projection_space,projection_time)
end

function RBSteady.projection(red::KroneckerReduction,s::TransientSnapshots)
  KroneckerProjection(red,s)
end

function RBSteady.projection(red::KroneckerReduction,s::TransientSnapshots,X::MatrixOrTensor)
  KroneckerProjection(red,s,X)
end

get_projection_space(a::KroneckerProjection) = a.projection_space
get_projection_time(a::KroneckerProjection) = a.projection_time

RBSteady.get_basis(a::KroneckerProjection) = kron(get_basis_time(a),get_basis_space(a))
RBSteady.num_reduced_dofs(a::KroneckerProjection) = num_reduced_dofs(a.projection_space)*num_reduced_dofs(a.projection_time)
RBSteady.get_norm_matrix(a::KroneckerProjection) = get_norm_matrix(a.projection_space)

function RBSteady.project!(x̂::AbstractVector{<:Number},a::KroneckerProjection,x::AbstractVector{<:Number})
  ns = num_reduced_dofs(a.projection_space)
  nt = num_reduced_dofs(a.projection_time)
  X̂ = reshape(x̂,ns,nt)

  Ns = num_fe_dofs(a.projection_space)
  Nt = num_fe_dofs(a.projection_time)
  X = reshape(x,Ns,Nt)

  basis_time = get_basis(a.projection_time)

  project!(X̂,a.projection_space,X*basis_time)
end

function RBSteady.inv_project!(x::AbstractVector{<:Number},a::KroneckerProjection,x̂::AbstractVector{<:Number})
  Ns = num_fe_dofs(a.projection_space)
  Nt = num_fe_dofs(a.projection_time)
  X = reshape(x,Ns,Nt)

  ns = num_reduced_dofs(a.projection_space)
  nt = num_reduced_dofs(a.projection_time)
  X̂ = reshape(x̂,ns,nt)

  basis_time = get_basis(a.projection_time)

  inv_project!(X,a.projection_space,X̂*basis_time')
end

function RBSteady.galerkin_projection(
  proj_left::KroneckerProjection,
  a::KroneckerProjection)

  proj_basis_space = galerkin_projection(get_basis_space(proj_left),get_basis_space(a))
  proj_basis_time = galerkin_projection(get_basis_time(proj_left),get_basis_time(a))
  proj_basis = kron(proj_basis_time,proj_basis_space)
  return ReducedProjection(proj_basis)
end

function RBSteady.galerkin_projection(
  proj_left::KroneckerProjection,
  a::KroneckerProjection,
  proj_right::KroneckerProjection)

  @notimplemented "In unsteady problems, we need to provide a combining function"
end

function RBSteady.galerkin_projection(
  proj_left::KroneckerProjection,
  a::KroneckerProjection,
  proj_right::KroneckerProjection,
  combine)

  proj_basis_space = galerkin_projection(
    get_basis_space(proj_left),
    get_basis_space(a),
    get_basis_space(proj_right))

  proj_basis_time = galerkin_projection(
    get_basis_time(proj_left),
    get_basis_time(a),
    get_basis_time(proj_right),
    combine)

  nleft = num_reduced_dofs(proj_left)
  ns = num_reduced_dofs(a.projection_space)
  nt = num_reduced_dofs(a.projection_time)
  n = num_reduced_dofs(a)
  nright = num_reduced_dofs(proj_right)

  proj_basis = zeros(nleft,n,nright)
  @inbounds for is = 1:ns, it = 1:nt
    ist = (it-1)*ns+is
    @views proj_basis[:,ist,:] = kron(proj_basis_time[:,it,:],proj_basis_space[:,is,:])
  end

  return ReducedProjection(proj_basis)
end

function RBSteady.empirical_interpolation(a::KroneckerProjection)
  indices_space,interp_space = empirical_interpolation(get_basis_space(a))
  indices_time,interp_time = empirical_interpolation(get_basis_time(a))
  interp = kron(interp_time,interp_space)
  return (indices_space,indices_time),interp
end

# tt interface

struct SequentialProjection{A} <: TransientProjection
  projection::A
end

function RBSteady.projection(red::SequentialReduction,s::TransientSnapshots)
  proj = projection(get_reduction(red),s)
  SequentialProjection(proj)
end

function RBSteady.projection(red::SequentialReduction,s::TransientSnapshots,X::MatrixOrTensor)
  proj = projection(get_reduction(red),s,X)
  SequentialProjection(proj)
end

#TODO when new projection operators are implemented, this will have to change
RBSteady.get_cores(a::SequentialProjection) = get_cores(a.projection)
get_cores_space(a::SequentialProjection) = get_cores(a)[1:end-1]
get_core_time(a::SequentialProjection) = get_cores(a)[end]
get_basis_space(a::SequentialProjection) = cores2basis(get_cores_space(a)...)
get_basis_time(a::SequentialProjection) = @notimplemented

ParamDataStructures.num_space_dofs(a::SequentialProjection) = prod(map(c -> size(c,2),get_cores_space(a)))
ParamDataStructures.num_times(a::SequentialProjection) = size(get_core_time(a),2)

DofMaps.get_dof_map(a::SequentialProjection) = get_dof_map(a.projection)

RBSteady.get_basis(a::SequentialProjection) = get_basis(a.projection)
RBSteady.num_reduced_dofs(a::SequentialProjection) = num_reduced_dofs(a.projection)
RBSteady.get_norm_matrix(a::SequentialProjection) = get_norm_matrix(a.projection)
RBSteady.empirical_interpolation(a::SequentialProjection) = empirical_interpolation(a.projection)

function RBSteady.union_bases(a::SequentialProjection,b::AbstractArray,args...)
  projection′ = union_bases(a.projection,b,args...)
  SequentialProjection(projection′)
end

function RBSteady.galerkin_projection(proj_left::SequentialProjection,a::SequentialProjection)
  galerkin_projection(proj_left.projection,a.projection)
end

function RBSteady.rescale(op::Function,X::AbstractRankTensor,b::SequentialProjection)
  RBSteady.rescale(op,X,b.projection)
end

function RBSteady.galerkin_projection(
  proj_left::SequentialProjection,
  a::SequentialProjection,
  proj_right::SequentialProjection,
  combine)

  _galerkin_projection(get_dof_map(a),proj_left,a,proj_right,combine)
end

function _galerkin_projection(
  ::AbstractDofMap,
  proj_left::SequentialProjection,
  a::SequentialProjection,
  proj_right::SequentialProjection,
  combine)

  # space
  pl_space = get_cores_space(proj_left)
  a_space = get_cores_space(a)
  pr_space = get_cores_space(proj_right)
  p_space = unbalanced_contractions(pl_space,a_space,pr_space)

  # time
  pl_time = get_core_time(proj_left)
  a_time = get_core_time(a)
  pr_time = get_core_time(proj_right)
  p_time = contraction(pl_time,a_time,pr_time,combine)

  p = sequential_product(p_space...,p_time)
  proj_cores = dropdims(p;dims=(1,2,3))

  return ReducedProjection(proj_cores)
end

function _galerkin_projection(
  ::TrivialDofMap,
  proj_left::SequentialProjection,
  a::SequentialProjection,
  proj_right::SequentialProjection,
  combine)

  get_core_space(a) = RBSteady.basis2core(get_basis_space(a))

  # space
  pl_space = get_core_space(proj_left)
  a_space = first(get_cores(a))
  pr_space = get_core_space(proj_right)
  p_space = contraction(pl_space,a_space,pr_space)

  # time
  pl_time = get_core_time(proj_left)
  a_time = get_core_time(a)
  pr_time = get_core_time(proj_right)
  p_time = contraction(pl_time,a_time,pr_time,combine)

  p = sequential_product(p_space,p_time)
  proj_cores = dropdims(p;dims=(1,2,3))

  return ReducedProjection(proj_cores)
end

function RBSteady.project!(x̂::AbstractVector{<:Number},a::SequentialProjection,x::AbstractVector{<:Number})
  project!(x̂,a.projection,x)
end

function RBSteady.inv_project!(x::AbstractVector{<:Number},a::SequentialProjection,x̂::AbstractVector{<:Number})
  inv_project!(x,a.projection,x̂)
end

# multfield interface

function Arrays.return_type(::typeof(projection),::KroneckerReduction,::TransientSnapshots)
  KroneckerProjection
end

function Arrays.return_type(::typeof(projection),::KroneckerReduction,::TransientSnapshots,::AbstractMatrix)
  KroneckerProjection
end

function Arrays.return_type(::typeof(projection),::SequentialReduction,::TransientSnapshots)
  SequentialProjection
end

function Arrays.return_type(::typeof(projection),::SequentialReduction,::TransientSnapshots,::AbstractRankTensor)
  SequentialProjection
end

function RBSteady.enrich!(
  red::SupremizerReduction{A,<:KroneckerReduction},
  a::BlockProjection,
  norm_matrix::BlockMatrix,
  supr_matrix::BlockMatrix;
  kwargs...
  ) where A

  @check a.touched[1] "Primal field not defined"
  a_primal,a_dual... = a.array
  a_primal_space = a_primal.projection_space
  a_primal_time = a_primal.projection_time
  X_primal = norm_matrix[Block(1,1)]
  H_primal = cholesky(X_primal)
  for i = eachindex(a_dual)
    if a.touched[i]
      dual_i_space = get_basis_space(a_dual[i])
      C_primal_dual_i = supr_matrix[Block(1,i+1)]
      supr_space_i = H_primal \ C_primal_dual_i * dual_i_space
      a_primal_space = union_bases(a_primal_space,supr_space_i,H_primal)

      dual_i_time = get_basis_time(a_dual[i])
      a_primal_time = time_enrichment(red,a_primal_time,dual_i_time;kwargs...)
    end
  end
  a[1] = KroneckerProjection(a_primal_space,a_primal_time)
  return
end

function RBSteady.enrich!(
  red::SupremizerReduction{A,<:SequentialReduction},
  a::BlockProjection,
  norm_matrix::BlockRankTensor,
  supr_matrix::BlockRankTensor;
  kwargs...
  ) where A

  red′ = SupremizerReduction(red.reduction.reduction,red.supr_op,red.supr_tol)
  enrich!(red′,a,norm_matrix,supr_matrix;kwargs...)
end

"""
    time_enrichment(red::SupremizerReduction,a_primal::Projection,basis_dual) -> AbstractMatrix

Temporal supremizer enrichment. (Approximate) Procedure:

1. for every `b_dual ∈ Col(basis_dual)`
2. compute `Φ_primal_dual = get_basis(a_primal)'*get_basis(b_dual)`
3. compute `v = kernel(Φ_primal_dual)`
4. compute `v′ = orth_complement(v,a_primal)`
5. enrich `a_primal = [a_primal,v′]`
"""
function time_enrichment(red::SupremizerReduction,a_primal::Projection,basis_dual)
  tol = RBSteady.get_supr_tol(red)
  time_red = get_reduction_time(get_reduction(red))
  basis_primal′ = time_enrichment(get_basis(a_primal),basis_dual,tol)
  projection(time_red,basis_primal′)
end

function time_enrichment(basis_primal,basis_dual,tol)
  basis_pd = basis_primal'*basis_dual

  function enrich!(basis_primal,basis_pd,v)
    vnew = copy(v)
    orth_complement!(vnew,basis_primal)
    vnew /= norm(vnew)
    hcat(basis_primal,vnew),vcat(basis_pd,vnew'*basis_dual)
  end

  i = 1
  while i ≤ size(basis_pd,2)
    basis_pd_start = view(basis_pd,:,1:i-1)
    basis_pd_i = view(basis_pd,:,i)
    basis_d_i = view(basis_dual,:,i)
    proj = i == 1 ? zeros(size(basis_pd,1)) : orth_projection(basis_pd_i,basis_pd_start)
    dist = norm(basis_pd_i-proj)
    if dist ≤ tol
      basis_primal,basis_pd = enrich!(basis_primal,basis_pd,basis_d_i)
      i = 0
    else
      basis_pd_i .-= proj
    end
    i += 1
  end

  return basis_primal
end
