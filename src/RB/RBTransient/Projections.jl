"""
    struct TransientProjection{A<:Projection,B<:AbstractArray} <: Projection
      projection_space::A
      array_time::B
    end

Space-time projection, made of a spatial `Projection` and a temporal array: a basis
for a [`KroneckerProjection`](@ref), the last TT core for a [`SequentialProjection`](@ref)

# Interface
- `get_projection_space(a)`: return the spatial `Projection`.
- `get_basis_space(a)` / `get_basis_time(a)`: retrieve the spatial basis / the
  temporal array.
"""
struct TransientProjection{A<:Projection,B<:AbstractArray} <: Projection
  projection_space::A
  array_time::B
end

"""
    get_projection_space(a::Projection) -> Projection

Spatial projection of `a` (`a` itself, if not transient)
"""
get_projection_space(a::Projection) = a
get_projection_space(a::TransientProjection) = a.projection_space

"""
    get_basis_space(a::Projection) -> AbstractMatrix

Spatial basis of `a`
"""
get_basis_space(a::Projection) = get_basis(a)
get_basis_space(a::TransientProjection) = get_basis_space(get_projection_space(a))

"""
    get_basis_time(a::Projection) -> AbstractArray

Temporal array of `a`: a matrix, or a TT core for a [`SequentialProjection`](@ref)
"""
get_basis_time(a::Projection) = @notimplemented "No available temporal basis $(typeof(a)) projections"
get_basis_time(a::TransientProjection) = a.array_time

ParamDataStructures.num_times(a::Projection) = @notimplemented "No available temporal basis $(typeof(a)) projections"
ParamDataStructures.num_space_dofs(a::Projection) = num_fe_dofs(get_projection_space(a))

RBSteady.num_fe_dofs(a::TransientProjection) = num_space_dofs(a)*num_times(a)
RBSteady.isnull(a::TransientProjection) = isnull(get_projection_space(a)) || isnull(get_basis_time(a))
RBSteady.get_norm_matrix(a::TransientProjection) = get_norm_matrix(get_projection_space(a))
DofMaps.get_dof_map(a::TransientProjection) = get_dof_map(get_projection_space(a))

function RBSteady.projection_eltype(a::TransientProjection)
  promote_type(projection_eltype(get_projection_space(a)),eltype(get_basis_time(a)))
end

num_fe_dofs_space(a::Projection) = num_fe_dofs(a)
num_reduced_dofs_space(a::Projection) = num_reduced_dofs(a)
reduced_dof_ids_space(a::Projection) = reduced_dof_ids(a)
fe_dof_ids_space(a::Projection) = fe_dof_ids(a)

num_fe_dofs_space(a::TransientProjection) = num_fe_dofs(get_projection_space(a))
num_reduced_dofs_space(a::TransientProjection) = num_reduced_dofs(get_projection_space(a))
reduced_dof_ids_space(a::TransientProjection) = reduced_dof_ids(get_projection_space(a))
fe_dof_ids_space(a::TransientProjection) = fe_dof_ids(get_projection_space(a))

function RBSteady.project!(
  x̂::ConsecutiveParamVector,
  a::TransientProjection,
  x::ConsecutiveParamVector
  )

  nt = num_times(a)
  np = param_length(x̂)
  @check Int(param_length(x) / np) == nt
  @inbounds @views for ip in param_eachindex(x̂)
    ipt = ip:np:np*nt
    xpt = vec(x.data[:,ipt])
    x̂p = x̂.data[:,ip]
    project!(x̂p,a,xpt)
  end
end

function RBSteady.inv_project!(
  x::ConsecutiveParamVector,
  a::TransientProjection,
  x̂::ConsecutiveParamVector
  )

  nt = num_times(a)
  np = param_length(x̂)
  @check Int(param_length(x) / np) == nt
  @inbounds @views for ip in param_eachindex(x̂)
    ipt = ip:np:np*nt
    xpt = vec(x.data[:,ipt])
    x̂p = x̂.data[:,ip]
    inv_project!(xpt,a,x̂p)
  end
end

function Algebra.allocate_in_domain(
  a::TransientProjection,
  x::V
  ) where V<:AbstractParamVector

  x̂ = allocate_vector(eltype(V),reduced_dof_ids(a))
  nt = num_times(a)
  np = Int(param_length(x) / nt)
  return parameterise(x̂,np)
end

function Algebra.allocate_in_range(
  a::TransientProjection,
  x̂::V
  ) where V<:AbstractParamVector

  x = allocate_vector(eltype(V),fe_dof_ids_space(a))
  nt = num_times(a)
  npt = param_length(x̂) * nt
  return parameterise(x,npt)
end

"""
    const KroneckerProjection{A<:Projection} = TransientProjection{A,<:AbstractMatrix}

[`TransientProjection`](@ref) with a temporal basis. The space-time projection
operator is equal to

`basis_time ⊗ projection_space`

which, for efficiency reasons, is never explicitly computed
"""
const KroneckerProjection{A<:Projection} = TransientProjection{A,<:AbstractMatrix}

KroneckerProjection(args...) = TransientProjection(args...)

function RBSteady.Projection(red::KroneckerReduction,s::TransientSnapshots,args...)
  basis_space,basis_time = tucker(red.reductions,s,args...)
  projection_space = GenericProjection(basis_space,get_dof_map(s))
  return projection_space,basis_time
end

function RBSteady.projection(red::KroneckerReduction,s::TransientSnapshots)
  ps,bt = Projection(red,s)
  return TransientProjection(ps,bt)
end

function RBSteady.projection(
  red::KroneckerReduction,
  s::TransientSnapshots,
  X::MatrixOrTensor
  )

  ps,bt = Projection(red,s,X)
  psX = NormedProjection(ps,X)
  return TransientProjection(psX,bt)
end

ParamDataStructures.num_times(a::KroneckerProjection) = size(get_basis_time(a),1)
num_reduced_dofs_time(a::KroneckerProjection) = size(get_basis_time(a),2)

RBSteady.get_basis(a::KroneckerProjection) = kron(get_basis_time(a),get_basis_space(a))
RBSteady.num_reduced_dofs(a::KroneckerProjection) = num_reduced_dofs_space(a)*num_reduced_dofs_time(a)

function RBSteady.project!(
  x̂::AbstractVector{<:Number},
  a::KroneckerProjection,
  x::AbstractVector{<:Number}
  )

  X̂ = reshape(x̂,num_reduced_dofs_space(a),num_reduced_dofs_time(a))
  X = reshape(x,num_space_dofs(a),num_times(a))
  project!(X̂,get_projection_space(a),X*get_basis_time(a))
end

function RBSteady.inv_project!(
  x::AbstractVector{<:Number},
  a::KroneckerProjection,
  x̂::AbstractVector{<:Number}
  )

  X = reshape(x,num_space_dofs(a),num_times(a))
  X̂ = reshape(x̂,num_reduced_dofs_space(a),num_reduced_dofs_time(a))
  inv_project!(X,get_projection_space(a),X̂*get_basis_time(a)')
end

function RBSteady.galerkin_projection(proj_left::KroneckerProjection,a::KroneckerProjection)
  proj_basis_space = galerkin_projection(get_basis_space(proj_left),get_basis_space(a))
  proj_basis_time = galerkin_projection(get_basis_time(proj_left),get_basis_time(a))
  proj_basis = kron(proj_basis_time,proj_basis_space)
  return ReducedProjection(proj_basis)
end

function RBSteady.galerkin_projection(
  proj_left::KroneckerProjection,
  a::KroneckerProjection,
  proj_right::KroneckerProjection,
  combine
  )

  proj_basis_space = galerkin_projection(
    get_basis_space(proj_left),
    get_recast_basis(get_projection_space(a)),
    get_basis_space(proj_right)
  )

  proj_basis_time = galerkin_projection(
    get_basis_time(proj_left),
    get_basis_time(a),
    get_basis_time(proj_right),
    combine
  )

  nleft = num_reduced_dofs(proj_left)
  ns = num_reduced_dofs_space(a)
  nt = num_reduced_dofs_time(a)
  n = num_reduced_dofs(a)
  nright = num_reduced_dofs(proj_right)

  T = projection_eltype(proj_left)
  S = projection_eltype(a)
  TS = promote_type(T,S)
  proj_basis = zeros(TS,nleft,nright,n)
  @inbounds @views for it = 1:nt, is = 1:ns
    ist = (it-1)*ns+is
    cache = proj_basis[:,:,ist]
    kron!(cache,proj_basis_time[:,:,it],proj_basis_space[:,:,is])
  end

  return ReducedProjection(proj_basis)
end

function RBSteady.galerkin_projection(proj_left::KroneckerProjection,a::GalerkinProjectable,args...)
  nt = num_times(proj_left)
  np = Int(param_length(a.array) / nt)
  proj_a_space = galerkin_projection(get_basis_space(proj_left),get_basis(a))
  proj_a_spacetime = galerkin_projection(get_basis_time(proj_left),change_mode(proj_a_space,np))
  proj_a_time_space = permutedims(reshape(proj_a_spacetime,size(proj_a_spacetime,1),:,np),(2,1,3))
  proj = reshape(proj_a_time_space,:,np)
  return ReducedProjection(proj)
end

function RBSteady.galerkin_projection(
  proj_left::KroneckerProjection,
  a::GalerkinProjectable,
  proj_right::KroneckerProjection,
  combine
  )

  nt = num_times(proj_left)
  np = Int(param_length(a.array) / nt)
  ns_left = num_reduced_dofs_space(proj_left)
  ns_right = num_reduced_dofs_space(proj_right)
  nt_left = num_reduced_dofs_time(proj_left)
  nt_right = num_reduced_dofs_time(proj_right)

  proj_a_space = galerkin_projection(
    get_basis_space(proj_left),
    get_basis(a),
    get_basis_space(proj_right)
  ) # ns_left x ns_right x Nμ*Nt

  a2 = reshape(proj_a_space,ns_left,ns_right,np,nt) # ns_left x ns_right x Nμ x Nt
  a3 = permutedims(a2,(4,1,2,3)) # Nt x ns_left x ns_right x Nμ
  a4 = reshape(a3,nt,ns_left*ns_right*np) # Nt x ns_left*ns_right*Nμ
  proj_a_spacetime = galerkin_projection(
    get_basis_time(proj_left),
    a4,
    get_basis_time(proj_right),
    combine
  ) # nt_left x nt_right x ns_left*ns_right*Nμ

  a5 = reshape(proj_a_spacetime,nt_left,nt_right,ns_left,ns_right,np) # nt_left x nt_right x ns_left x ns_right x Nμ
  a6 = permutedims(a5,(3,1,4,2,5)) # ns_left x nt_left x ns_right x nt_right x Nμ
  proj = reshape(a6,ns_left*nt_left,ns_right*nt_right,np)
  return ReducedProjection(proj)
end

for f in (:DEIM,:SOPT)
  @eval begin
    function RBSteady.$f(a::KroneckerProjection)
      indices_space,interp_space = RBSteady.$f(get_projection_space(a))
      indices_time,interp_time = RBSteady.$f(get_basis_time(a))
      interp = kron(interp_time,interp_space)
      return (indices_space,indices_time),interp
    end
  end
end

# tt interface

"""
    const SequentialProjection{A<:Projection} = TransientProjection{A,<:AbstractArray{<:Any,3}}

[`TransientProjection`](@ref) computed with a TT-SVD of the snapshots, whose temporal
array is the last TT core. The reduced dimension is the rank of this core. Output
of a [`SequentialReduction`](@ref)
"""
const SequentialProjection{A<:Projection} = TransientProjection{A,<:AbstractArray{<:Any,3}}

SequentialProjection(args...) = TransientProjection(args...)

function RBSteady.Projection(red::SequentialReduction,s::TransientSnapshots,args...)
  cores = reduction(red,s,args...)
  projection_space = TTSVDProjection(cores[1:end-1],get_dof_map(s))
  return projection_space,last(cores)
end

function RBSteady.projection(red::SequentialReduction,s::TransientSnapshots)
  ps,ct = Projection(red,s)
  return TransientProjection(ps,ct)
end

function RBSteady.projection(
  red::SequentialReduction,
  s::TransientSnapshots,
  X::MatrixOrTensor
  )

  ps,ct = Projection(red,s,X)
  psX = NormedProjection(ps,X)
  return TransientProjection(psX,ct)
end

get_cores_space(a::SequentialProjection) = get_cores(get_projection_space(a))
get_core_time(a::SequentialProjection) = get_basis_time(a)
RBSteady.get_cores(a::SequentialProjection) = [get_cores_space(a)...,get_core_time(a)]

ParamDataStructures.num_times(a::SequentialProjection) = size(get_core_time(a),2)

RBSteady.get_basis(a::SequentialProjection) = cores2basis(get_recast_cores(a)...)
RBSteady.num_reduced_dofs(a::SequentialProjection) = size(get_core_time(a),3)

RBSteady.DEIM(a::SequentialProjection) = DEIM(_to_ttsvd(a))
RBSteady.SOPT(a::SequentialProjection) = SOPT(_to_ttsvd(a))

function RBSteady.union_bases(a::SequentialProjection,b::AbstractArray,args...)
  ab = union_bases(_to_ttsvd(a),b,args...)
  _to_seq(a,ab)
end

function RBSteady.project!(
  x̂::AbstractVector{<:Number},
  a::SequentialProjection,
  x::AbstractVector{<:Number}
  )

  X = reshape(x,:,num_times(a))
  X̂s = project(get_projection_space(a),X)
  mul!(x̂,transpose(cores2basis(get_core_time(a))),vec(X̂s))
end

function RBSteady.inv_project!(
  x::AbstractVector{<:Number},
  a::SequentialProjection,
  x̂::AbstractVector{<:Number}
  )

  X = reshape(x,:,num_times(a))
  X̂s = reshape(cores2basis(get_core_time(a))*x̂,num_reduced_dofs_space(a),num_times(a))
  inv_project!(X,get_projection_space(a),X̂s)
end

function RBSteady.galerkin_projection(proj_left::SequentialProjection,a::SequentialProjection)
  ps_left = get_projection_space(proj_left)
  ps = get_projection_space(a)
  p_space = get_basis(galerkin_projection(ps_left,ps)) # ns_left x ns_a

  nt = num_times(a)
  n = num_reduced_dofs(a)
  core = reshape(get_core_time(a),size(p_space,2),nt*n)
  p_core = reshape(p_space*core,size(p_space,1)*nt,n)
  ReducedProjection(transpose(cores2basis(get_core_time(proj_left)))*p_core)
end

function RBSteady.galerkin_projection(
  proj_left::SequentialProjection,
  a::SequentialProjection,
  proj_right::SequentialProjection,
  combine
  )

  p_space = space_contractions(proj_left,a,proj_right)

  pl_time = get_core_time(proj_left)
  a_time = get_core_time(a)
  pr_time = get_core_time(proj_right)
  p_time = contraction(pl_time,a_time,pr_time,combine)

  p = sequential_product(p_space...,p_time)
  proj_cores = dropdims(p;dims=(1,2,3)) # n_test x n_a x n_trial
  proj_cores = permutedims(proj_cores,(1,3,2)) # n_test x n_trial x n_a

  return ReducedProjection(proj_cores)
end

"""
    space_contractions(proj_left,a,proj_right) -> Tuple{Vararg{AbstractArray}}

Spatial factors of the Galerkin projection of the [`SequentialProjection`](@ref) `a`
"""
function space_contractions(
  proj_left::SequentialProjection,
  a::SequentialProjection{<:TTSVDProjection{A,<:AbstractDofMap}},
  proj_right::SequentialProjection
  ) where A

  pl_space = get_cores_space(proj_left)
  a_space = recast(get_cores_space(a),get_dof_map(a))
  pr_space = get_cores_space(proj_right)
  unbalanced_contractions(pl_space,a_space,pr_space)
end

function space_contractions(
  proj_left::SequentialProjection,
  a::SequentialProjection{<:TTSVDProjection{A,<:TrivialDofMap}},
  proj_right::SequentialProjection
  ) where A

  pl_space = RBSteady.basis2core(get_basis_space(proj_left))
  a_space = first(get_recast_cores(a))
  pr_space = RBSteady.basis2core(get_basis_space(proj_right))
  (contraction(pl_space,a_space,pr_space),)
end

function RBSteady.galerkin_projection(proj_left::SequentialProjection,a::GalerkinProjectable)
  nt = num_times(proj_left)
  np = Int(param_length(a.array) / nt)
  proj_a_space = galerkin_projection(get_basis_space(proj_left),get_basis(a))
  proj_core_space = permutedims(reshape(proj_a_space,:,np,nt),(1,3,2)) # ns_left x Nt x Nμ
  pl_time = get_core_time(proj_left) # ns_left x Nt x nt_left
  proj = _contraction(pl_time,proj_core_space) # nt_left x Nμ
  return ReducedProjection(proj)
end

function RBSteady.galerkin_projection(
  proj_left::SequentialProjection,
  a::GalerkinProjectable,
  proj_right::SequentialProjection,
  combine
  )

  nt = num_times(proj_left)
  np = Int(param_length(a.array) / nt)
  ns_left = num_reduced_dofs_space(proj_left)
  ns_right = num_reduced_dofs_space(proj_right)

  proj_a_space = galerkin_projection(
    get_basis_space(proj_left),
    get_basis(a),
    get_basis_space(proj_right)
  ) # ns_left x ns_right x Nμ*Nt

  a2 = reshape(proj_a_space,ns_left,ns_right,np,nt) # ns_left x ns_right x Nμ x Nt
  a4 = permutedims(a2,(1,4,3,2)) # ns_left x Nt x Nμ x ns_right

  pl_time = get_core_time(proj_left) # ns_left x Nt x nt_left
  pr_time = get_core_time(proj_right) # ns_right x Nt x nt_right
  proj_cores = _contraction(pl_time,a4,pr_time,combine) # nt_left x nt_right x Nμ
  return ReducedProjection(proj_cores)
end

# multifield interface

function RBSteady.enrich!(
  red::SupremizerReduction{A,B,<:TransientReduction},
  a::BlockProjection,
  norm_matrix,
  supr_matrix;
  kwargs...
  ) where {A,B}

  red′ = SupremizerReduction(red.reduction.reduction,red.coupling,red.supr_tol)
  enrich!(red′,a,norm_matrix,supr_matrix;kwargs...)
end

function RBSteady.enrich!(
  red::SupremizerReduction{A,B,<:KroneckerReduction},
  a::BlockProjection,
  norm_matrix,
  supr_matrix
  ) where {A,B}

  tol = RBSteady.get_supr_tol(red)
  a_primal,a_dual... = a.array
  a_primal_space = get_projection_space(a_primal)
  a_primal_time = get_basis_time(a_primal)
  X_primal = norm_matrix[1]
  H_primal = gram_solver(X_primal)
  for i = eachindex(a_dual)
    dual_i_space = get_basis_space(a_dual[i])
    C_primal_dual_i = supr_matrix[i]
    supr_space_i = supremizers(H_primal,C_primal_dual_i,dual_i_space)
    a_primal_space = union_bases(a_primal_space,supr_space_i,H_primal)

    dual_i_time = get_basis_time(a_dual[i])
    a_primal_time = time_enrichment(a_primal_time,dual_i_time;tol)
  end
  a[1] = KroneckerProjection(a_primal_space,a_primal_time)
  return
end

"""
    time_enrichment(basis_primal::AbstractMatrix,basis_dual::AbstractMatrix;tol=1e-2) -> AbstractMatrix

Temporal supremizer enrichment. (Approximate) Procedure:

1. for every `b_dual ∈ Col(basis_dual)`
2. compute `Φ_primal_dual = basis_primal'*b_dual`
3. compute `v = kernel(Φ_primal_dual)`
4. compute `v′ = orth_complement(v,basis_primal)`
5. enrich `basis_primal = [basis_primal,v′]`
"""
function time_enrichment(basis_primal,basis_dual;tol=1e-2)
  basis_pd = basis_primal'*basis_dual
  T = eltype(basis_pd)

  i = 1
  while i ≤ size(basis_pd,2)
    basis_pd_start = view(basis_pd,:,1:i-1)
    basis_pd_i = view(basis_pd,:,i)
    proj = i == 1 ? zeros(T,size(basis_pd,1)) : orth_projection(basis_pd_i,basis_pd_start)
    dist = norm(basis_pd_i-proj)
    if dist ≤ tol
      basis_primal,basis_pd = tenrich(basis_primal,basis_pd,basis_dual,i)
      i = 0
    else
      basis_pd_i .-= proj
    end
    i += 1
  end

  return basis_primal
end

function tenrich(basis_primal,basis_pd,basis_dual,i)
  vi = copy(view(basis_dual,:,i))
  orth_complement!(vi,basis_primal)
  vi ./= norm(vi)
  hcat(basis_primal,vi),vcat(basis_pd,vi'*basis_dual)
end

# space-only projections

function space_project(a::Projection,x::AbstractArray,args...)
  x̂ = allocate_in_space_domain(a,x)
  project_space!(x̂,a,x,args...)
  return x̂
end

function inv_space_project(a::Projection,x̂::AbstractArray)
  x = allocate_in_space_range(a,x̂)
  inv_project_space!(x,a,x̂)
  return x
end

function project_space!(x̂::AbstractArray,a::Projection,x::AbstractArray)
  basis = get_basis_space(a)
  mul!(x̂,basis',x)
end

function inv_project_space!(x::AbstractArray,a::Projection,x̂::AbstractArray)
  basis = get_basis_space(a)
  mul!(x,basis,x̂)
end

function allocate_in_space_domain(a::Projection,x::V) where V<:AbstractVector
  x̂ = allocate_vector(V,reduced_dof_ids_space(a))
  return x̂
end

function allocate_in_space_range(a::Projection,x̂::V) where V<:AbstractVector
  x = allocate_vector(V,fe_dof_ids_space(a))
  return x
end

function allocate_in_space_domain(a::Projection,X::M) where M<:AbstractMatrix
  X̂ = zeros(eltype(M),num_reduced_dofs_space(a),size(X,2))
  return X̂
end

function allocate_in_space_range(a::Projection,X̂::M) where M<:AbstractMatrix
  X = Matrix{eltype(M)}(undef,num_fe_dofs_space(a),size(X̂,2))
  return X
end

function allocate_in_space_domain(a::Projection,x::V) where V<:AbstractParamVector
  x̂ = allocate_vector(eltype(V),reduced_dof_ids_space(a))
  return parameterise(x̂,param_length(x))
end

function allocate_in_space_range(a::Projection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(eltype(V),fe_dof_ids_space(a))
  return parameterise(x,param_length(x̂))
end

function num_fe_dofs_space(a::BlockProjection)
  dofs = 0
  for i in eachindex(a)
    dofs += num_fe_dofs_space(a[i])
  end
  return dofs
end

function num_reduced_dofs_space(a::BlockProjection)
  dofs = 0
  for i in eachindex(a)
    dofs += num_reduced_dofs_space(a[i])
  end
  return dofs
end

for f in (:allocate_in_space_domain,:allocate_in_space_range)
  @eval begin
    function $f(a::BlockProjection)
      mortar(map($f,a.array))
    end

    function $f(a::BlockProjection,x::Union{BlockArray,BlockParamArray})
      @check length(a) == blocklength(x)
      mortar(map(i -> $f(a[Block(i)],blocks(x)[i]),eachindex(a)))
    end
  end
end

for f in (:project_space!,:inv_project_space!)
  @eval begin
    function $f(
      y::Union{BlockArray,BlockParamArray},
      a::BlockProjection,
      x::Union{BlockArray,BlockParamArray}
      )

      for i in eachindex(a)
        $f(blocks(y)[i],a[i],blocks(x)[i])
      end
    end
  end
end

# utils

_to_ttsvd(a::SequentialProjection) = TTSVDProjection(get_cores(a),get_dof_map(a))

function _to_seq(a::SequentialProjection,b::TTSVDProjection)
  cores_space...,core_time = get_cores(b)
  proj_space = _to_seq(get_projection_space(a),TTSVDProjection(cores_space,get_dof_map(b)))
  TransientProjection(proj_space,core_time)
end

_to_seq(a::TTSVDProjection,b) = b
_to_seq(a::NormedProjection,b) = NormedProjection(b,get_norm_matrix(a))