function _projection(red::TransientAffineReduction,s::TransientSnapshots,args...)
  s1 = flatten(select_snapshots(s,1,1))
  projection_space = projection(get_reduction_space(red),s1,args...)
  projection_time = PODProjection(I[1:num_times(s),1:1])
  TransientProjection(projection_space,projection_time)
end

function _projection(red::TransientReduction,s::TransientSnapshots,args...)
  s1 = flatten(s)
  projection_space = projection(get_reduction_space(red),s1,args...)
  proj_s1 = project(projection_space,s1)
  proj_s2 = change_mode(proj_s1,num_params(s))
  projection_time = projection(get_reduction_time(red),proj_s2)
  TransientProjection(projection_space,projection_time)
end

function RBSteady.projection(red::TransientReduction,s::TransientSnapshots)
  _projection(red,s)
end

function RBSteady.projection(red::TransientReduction,s::TransientSnapshots,X::MatrixOrTensor)
  _projection(red,s,X)
end

"""
    struct TransientProjection <: Projection
      projection_space::Projection
      projection_time::Projection
    end

Projection operator for transient problems, containing a spatial projection and
a temporal one. The space-time projection operator is equal to

`projection_time ⊗ projection_space`

which, for efficiency reasons, is never explicitly computed
"""
struct TransientProjection <: Projection
  projection_space::Projection
  projection_time::Projection
end

get_projection_space(a::TransientProjection) = a.projection_space
get_projection_time(a::TransientProjection) = a.projection_time
get_basis_space(a::TransientProjection) = get_basis(a.projection_space)
get_basis_time(a::TransientProjection) = get_basis(a.projection_time)

RBSteady.get_basis(a::TransientProjection) = kron(get_basis_time(a),get_basis_space(a))
RBSteady.num_fe_dofs(a::TransientProjection) = num_fe_dofs(a.projection_space)*num_fe_dofs(a.projection_time)
RBSteady.num_reduced_dofs(a::TransientProjection) = num_reduced_dofs(a.projection_space)*num_reduced_dofs(a.projection_time)
RBSteady.get_norm_matrix(a::TransientProjection) = get_norm_matrix(a.projection_space)

function RBSteady.project!(x̂::AbstractVector,a::TransientProjection,x::AbstractVector)
  ns = num_reduced_dofs(a.projection_space)
  nt = num_reduced_dofs(a.projection_time)
  X̂ = reshape(x̂,ns,nt)

  Ns = num_fe_dofs(a.projection_space)
  Nt = num_fe_dofs(a.projection_time)
  X = reshape(x,Ns,Nt)

  basis_time = get_basis(a.projection_time)

  project!(X̂,a.projection_space,X*basis_time)
end

function RBSteady.inv_project!(x::AbstractVector,a::TransientProjection,x̂::AbstractVector)
  Ns = num_fe_dofs(a.projection_space)
  Nt = num_fe_dofs(a.projection_time)
  X = reshape(x,Ns,Nt)

  ns = num_reduced_dofs(a.projection_space)
  nt = num_reduced_dofs(a.projection_time)
  X̂ = reshape(x̂,ns,nt)

  basis_time = get_basis(a.projection_time)

  inv_project!(X,a.projection_space,X̂*basis_time')
end

function RBSteady.project!(x̂::ConsecutiveParamVector,a::TransientProjection,x::ConsecutiveParamVector)
  nt = num_fe_dofs(a.projection_time)
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
  nt = num_fe_dofs(a.projection_time)
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
  nt = num_fe_dofs(a.projection_time)
  np = Int(param_length(x) / nt)
  return global_parameterize(x̂,np)
end

function Algebra.allocate_in_range(a::TransientProjection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(eltype(V),num_fe_dofs(a.projection_space))
  nt = num_fe_dofs(a.projection_time)
  npt = param_length(x̂) * nt
  return global_parameterize(x,npt)
end

function RBSteady.galerkin_projection(
  proj_left::TransientProjection,
  a::TransientProjection)

  proj_basis_space = galerkin_projection(get_basis_space(proj_left),get_basis_space(a))
  proj_basis_time = galerkin_projection(get_basis_time(proj_left),get_basis_time(a))
  proj_basis = kron(proj_basis_time,proj_basis_space)
  return ReducedProjection(proj_basis)
end

function RBSteady.galerkin_projection(
  proj_left::TransientProjection,
  a::TransientProjection,
  proj_right::TransientProjection)

  @notimplemented "In unsteady problems, we need to provide a combining function"
end

function RBSteady.galerkin_projection(
  proj_left::TransientProjection,
  a::TransientProjection,
  proj_right::TransientProjection,
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

function RBSteady.empirical_interpolation(a::TransientProjection)
  indices_space,interp_space = empirical_interpolation(get_basis_space(a))
  indices_time,interp_time = empirical_interpolation(get_basis_time(a))
  interp = kron(interp_time,interp_space)
  return (indices_space,indices_time),interp
end

# tt interface

get_cores_space(a::TTSVDProjection) = get_cores(a)[1:end-1]
get_core_time(a::TTSVDProjection) = get_cores(a)[end]

function RBSteady.galerkin_projection(
  proj_left::TTSVDProjection,
  a::TTSVDProjection,
  proj_right::TTSVDProjection,
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

# multfield interface

function Arrays.return_type(::typeof(projection),::TransientReduction,::TransientSnapshots)
  TransientProjection
end

function Arrays.return_type(::typeof(projection),::TransientReduction,::TransientSnapshots,::MatrixOrTensor)
  TransientProjection
end

function RBSteady.enrich!(
  red::SupremizerReduction,
  a::BlockProjection{<:TransientProjection},
  norm_matrix::BlockMatrix,
  supr_matrix::BlockMatrix;
  kwargs...)

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
      a_primal_space = RBSteady.union_bases(a_primal_space,supr_space_i,X_primal)

      dual_i_time = get_basis_time(a_dual[i])
      a_primal_time = time_enrichment(red,a_primal_time,dual_i_time;kwargs...)
    end
  end
  a[1] = TransientProjection(a_primal_space,a_primal_time)
  return
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
