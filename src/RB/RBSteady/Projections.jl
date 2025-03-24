"""
    abstract type Projection <: Map end

Represents a basis for a `n`-dimensional vector subspace of a `N`-dimensional
vector space (where `N` ≫ `n`), to be used as a Galerkin projection operator.
The kernel of a Projection is `n`-dimensional, whereas its image is
`N`-dimensional.

Subtypes:

- [`PODProjection`](@ref)
- [`TTSVDProjection`](@ref)
- [`NormedProjection`](@ref)
- [`BlockProjection`](@ref)
- [`InvProjection`](@ref)
- [`ReducedProjection`](@ref)
- [`HRProjection`](@ref)
"""
abstract type Projection <: Map end

"""
    get_basis(a::Projection) -> AbstractMatrix

Returns the basis spanning the reduced subspace represented by the projection `a`
"""
get_basis(a::Projection) = @abstractmethod

"""
    num_fe_dofs(a::Projection) -> Int

For a projection map `a` from a low dimensional space `n` to a high dimensional
one `N`, returns `N`
"""
num_fe_dofs(a::Projection) = @abstractmethod

"""
    num_reduced_dofs(a::Projection) -> Int

For a projection map `a` from a low dimensional space `n` to a high dimensional
one `N`, returns `n`
"""
num_reduced_dofs(a::Projection) = @abstractmethod

"""
    project(a::Projection,x::AbstractArray,args...) -> AbstractArray

Projects a high-dimensional object `x` onto the subspace represented by `a`
"""
function project(a::Projection,x::AbstractArray,args...)
  x̂ = allocate_in_domain(a,x)
  project!(x̂,a,x,args...)
  return x̂
end

"""
    project!(x̂::AbstractArray,a::Projection,x::AbstractArray,args...) -> Nothing

In-place projection of a high-dimensional object `x` onto the subspace represented by `a`
"""
function project!(x̂::AbstractArray,a::Projection,x::AbstractArray)
  basis = get_basis(a)
  mul!(x̂,basis',x)
end

function project!(x̂::AbstractArray,a::Projection,x::AbstractArray,norm_matrix::AbstractMatrix)
  project!(x̂,a,norm_matrix*x)
end

"""
    inv_project(a::Projection,x::AbstractArray) -> AbstractArray

Recasts a low-dimensional object `x` onto the high-dimensional space in which `a`
is immersed
"""
function inv_project(a::Projection,x̂::AbstractArray)
  x = allocate_in_range(a,x̂)
  inv_project!(x,a,x̂)
  return x
end

"""
    inv_project!(x::AbstractArray,a::Projection,x̂::AbstractArray) -> Nothing

In-place recasting of a low-dimensional object `x̂` the high-dimensional space
in which `a` is immersed
"""
function inv_project!(x::AbstractArray,a::Projection,x̂::AbstractArray)
  basis = get_basis(a)
  mul!(x,basis,x̂)
end

function Algebra.allocate_in_domain(a::Projection,x::V) where V<:AbstractVector
  x̂ = allocate_vector(V,num_reduced_dofs(a))
  return x̂
end

function Algebra.allocate_in_range(a::Projection,x̂::V) where V<:AbstractVector
  x = allocate_vector(V,num_fe_dofs(a))
  return x
end

function Algebra.allocate_in_domain(a::Projection,X::M) where M<:AbstractMatrix
  X̂ = zeros(eltype(M),num_reduced_dofs(a),size(X,2))
  return X̂
end

function Algebra.allocate_in_range(a::Projection,X̂::M) where M<:AbstractMatrix
  X = allocate_vector(V,num_fe_dofs(a),size(X̂,2))
  return X
end

function Algebra.allocate_in_domain(a::Projection,x::V) where V<:AbstractParamVector
  x̂ = allocate_vector(eltype(V),num_reduced_dofs(a))
  return global_parameterize(x̂,param_length(x))
end

function Algebra.allocate_in_range(a::Projection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(eltype(V),num_fe_dofs(a))
  return global_parameterize(x,param_length(x̂))
end

"""
    galerkin_projection(a::Projection,b::Projection) -> ReducedProjection
    galerkin_projection(a::Projection,b::Projection,c::Projection,args...) -> ReducedProjection

(Petrov) Galerkin projection of a projection map `b` onto the subspace `a` (row
projection) and, if applicable, onto the subspace `c` (column projection)
"""
function galerkin_projection(a::Projection,b::Projection)
  b̂ = galerkin_projection(get_basis(a),get_basis(b))
  return ReducedProjection(b̂)
end

function galerkin_projection(a::Projection,b::Projection,c::Projection,args...)
  b̂ = galerkin_projection(get_basis(a),get_basis(b),get_basis(c),args...)
  return ReducedProjection(b̂)
end

"""
    empirical_interpolation(a::Projection) -> (AbstractVector,AbstractMatrix)

Computes the EIM of `a`. The outputs are:

- a vector of integers `i`, corresponding to a list of interpolation row indices
- a matrix `Φi = view(Φ,i)`, where `Φ = get_basis(a)`. This quantity represents
  the restricted basis on the set of interpolation rows `i`
"""
empirical_interpolation(a::Projection) = @abstractmethod

"""
    union_bases(a::Projection,b::Projection,args...) -> Projection

Computes the projection corresponding to the union of `a` and `b`. In essence this
operation performs as

  `gram_schmidt(union(get_basis(a),get_basis(b)))`
"""
union_bases(a::Projection,b::Projection,args...) = @abstractmethod

gram_schmidt(a::Projection,b::Projection,args...) = gram_schmidt(get_basis(a),get_basis(b),args...)

get_norm_matrix(a::Projection) = I(num_fe_dofs(a))

rescale(op::Function,x::AbstractArray,b::Projection) = @abstractmethod

Base.:+(a::Projection,b::Projection) = union_bases(a,b)
Base.:-(a::Projection,b::Projection) = union_bases(a,b)
Base.:*(a::Projection,b::Projection) = galerkin_projection(a,b)
Base.:*(a::Projection,b::Projection,c::Projection) = galerkin_projection(a,b,c)
Base.:*(a::Projection,x::AbstractArray) = inv_project(a,x)
Base.:*(x::AbstractArray,b::Projection) = rescale(*,x,b)

function Base.:*(b::Projection,y::ConsecutiveParamArray)
  item = zeros(num_reduced_dofs(b))
  plength = param_length(y)
  x = global_parameterize(item,plength)
  mul!(x,b,y)
end

function LinearAlgebra.mul!(x::AbstractArray,b::Projection,y::AbstractArray,α::Number,β::Number)
  mul!(x,get_basis(b),y,α,β)
end

function LinearAlgebra.mul!(x::ConsecutiveParamArray,b::Projection,y::ConsecutiveParamArray,α::Number,β::Number)
  mul!(x.data,get_basis(b),y.data,α,β)
end

# constructors

"""
    projection(red::Reduction,s::AbstractArray) -> Projection
    projection(red::Reduction,s::AbstractArray,X::MatrixOrTensor) -> Projection

Constructs a [`Projection`](@ref) from a collection of snapshots `s`. An inner product
represented by the quantity `X` can be provided, in which case the resulting
`Projection` will be `X`-orthogonal
"""
function projection(red::Reduction,s::AbstractArray)
  Projection(red,s)
end

function projection(red::Reduction,s::AbstractArray,X::MatrixOrTensor)
  proj = Projection(red,s,X)
  NormedProjection(proj,X)
end

function Projection(red::Reduction,s::AbstractArray,args...)
  @abstractmethod
end

"""
    struct InvProjection <: Projection
      projection::Projection
    end

Represents the inverse map of a [`Projection`](@ref) `projection`
"""
struct InvProjection <: Projection
  projection::Projection
end

Base.adjoint(a::Projection) = InvProjection(a)

get_basis(a::InvProjection) = adjoint(get_basis(a.projection))
num_fe_dofs(a::InvProjection) = num_reduced_dofs(a.projection)
num_reduced_dofs(a::InvProjection) = num_fe_dofs(a.projection)
get_norm_matrix(a::InvProjection) = get_norm_matrix(a.projection)

"""
    abstract type ReducedProjection{A<:AbstractArray} <: Projection end

Type representing a Galerkin projection of a [`Projection`](@ref) onto a reduced subspace
represented by another `Projection`.

Subtypes:

- [`ReducedAlgebraicProjection`](@ref)
"""
abstract type ReducedProjection{A<:AbstractArray} <: Projection end

const ReducedVecProjection = ReducedProjection{<:AbstractMatrix}
const ReducedMatProjection = ReducedProjection{<:AbstractArray{T,3} where T}

function project!(x̂::AbstractVector,a::ReducedProjection,x::AbstractVector)
  @notimplemented
end

function inv_project!(x::AbstractVector,a::ReducedMatProjection,x̂::AbstractVector)
  basis = get_basis(a)
  contraction!(x,basis,x̂)
end

function LinearAlgebra.mul!(
  x::AbstractArray,
  b::ReducedMatProjection,
  y::AbstractArray,
  α::Number,β::Number)

  contraction!(x,get_basis(b),y,α,β)
end

function LinearAlgebra.mul!(
  x::ConsecutiveParamArray,
  b::ReducedMatProjection,
  y::ConsecutiveParamArray,
  α::Number,β::Number)

  contraction!(x.data,get_basis(b),y.data,α,β)
end

"""
"""
struct ReducedAlgebraicProjection{A} <: ReducedProjection{A}
  basis::A
end

ReducedProjection(basis::AbstractArray) = ReducedAlgebraicProjection(basis)

get_basis(a::ReducedAlgebraicProjection) = a.basis
num_reduced_dofs(a::ReducedAlgebraicProjection) = size(get_basis(a),2)
num_reduced_dofs_left_projector(a::ReducedAlgebraicProjection) = size(get_basis(a),1)
num_reduced_dofs_right_projector(a::ReducedMatProjection) = size(get_basis(a),3)

function Projection(red::AffineReduction,s::AbstractMatrix,args...)
  podred = PODReduction(ReductionStyle(red),NormStyle(red))
  Projection(podred,s,args...)
end

"""
    struct PODProjection <: Projection
      basis::AbstractMatrix
    end

Projection stemming from a truncated proper orthogonal decomposition [`tpod`](@ref)
"""
struct PODProjection <: Projection
  basis::AbstractMatrix
end

function Projection(red::PODReduction,s::AbstractArray,args...)
  basis = reduction(red,s,args...)
  PODProjection(basis)
end

function Projection(red::PODReduction,s::SparseSnapshots,args...)
  basis = reduction(red,s,args...)
  basis′ = recast(basis,s)
  PODProjection(basis′)
end

get_basis(a::PODProjection) = a.basis
num_fe_dofs(a::PODProjection) = size(get_basis(a),1)
num_reduced_dofs(a::PODProjection) = size(get_basis(a),2)

union_bases(a::PODProjection,b::PODProjection,args...) = union_bases(a,get_basis(b),args...)

function union_bases(a::PODProjection,basis_b::AbstractMatrix,args...)
  basis_a = get_basis(a)
  basis_ab, = gram_schmidt(basis_b,basis_a,args...)
  PODProjection(basis_ab)
end

function empirical_interpolation(a::PODProjection)
  empirical_interpolation(get_basis(a))
end

function rescale(op::Function,x::AbstractArray,b::PODProjection)
  PODProjection(op(x,get_basis(b)))
end

# TT interface

"""
    struct TTSVDProjection <: Projection
      cores::AbstractVector{<:AbstractArray{T,3} where T}
      dof_map::AbstractDofMap
    end

Projection stemming from a tensor train SVD [`ttsvd`](@ref). For reindexing purposes
a field `dof_map` is provided along with the tensor train cores `cores`
"""
struct TTSVDProjection <: Projection
  cores::AbstractVector{<:AbstractArray{T,3} where T}
  dof_map::AbstractDofMap
end

function Projection(red::TTSVDReduction,s::AbstractMatrix,args...)
  Projection(first(red),s,args...)
end

function Projection(red::TTSVDReduction,s::AbstractArray,args...)
  cores = reduction(red,s,args...)
  dof_map = get_dof_map(s)
  TTSVDProjection(cores,dof_map)
end

function Projection(red::TTSVDReduction,s::SparseSnapshots,args...)
  cores = reduction(red,s,args...)
  cores′ = recast(cores,s)
  dof_map = get_dof_map(s)
  TTSVDProjection(cores′,dof_map)
end

get_cores(a::Projection) = @notimplemented
get_cores(a::TTSVDProjection) = a.cores

DofMaps.get_dof_map(a::Projection) = @notimplemented
DofMaps.get_dof_map(a::TTSVDProjection) = a.dof_map

get_basis(a::TTSVDProjection) = cores2basis(get_cores(a)...)
num_fe_dofs(a::TTSVDProjection) = prod(map(c -> size(c,2),get_cores(a)))
num_reduced_dofs(a::TTSVDProjection) = size(last(get_cores(a)),3)

#TODO this needs to be fixed; for now it's ok, since it's only used to generate
# zero free reduced dof values
function project!(x̂::AbstractArray,a::TTSVDProjection,x::AbstractArray,norm_matrix::AbstractRankTensor)
  # a′ = rescale(_sparse_rescaling,norm_matrix,a)
  # basis′ = get_basis(a′)
  # mul!(x̂,basis′',x)
  x̂
end

function union_bases(a::TTSVDProjection,b::TTSVDProjection,args...)
  @check get_dof_map(a) == get_dof_map(b)
  union_bases(a,get_cores(b),args...)
end

function union_bases(
  a::TTSVDProjection,
  cores_b::AbstractVector{<:AbstractArray},
  args...
  )

  red_style = TTSVDReduction(fill(1e-4,length(cores_a)))
  union_bases(a,cores_b,red_style,args...)
end

function union_bases(
  a::TTSVDProjection,
  cores_b::AbstractVector{<:AbstractArray},
  red_style::ReductionStyle,
  args...
  )

  cores_a = get_cores(a)
  @check length(cores_a) == length(cores_b)

  cores_ab = block_cores([cores_a,cores_b])
  orthogonalize!(red_style,cores_ab,args...)
  TTSVDProjection(cores_ab,get_dof_map(a))
end

function galerkin_projection(proj_left::TTSVDProjection,a::TTSVDProjection)
  cores_left = get_cores(proj_left)
  cores = get_cores(a)
  proj_basis = galerkin_projection(cores_left,cores)
  return ReducedProjection(proj_basis)
end

function galerkin_projection(proj_left::TTSVDProjection,a::TTSVDProjection,proj_right::TTSVDProjection)
  cores_left = get_cores(proj_left)
  cores = get_cores(a)
  cores_right = get_cores(proj_right)
  proj_basis = galerkin_projection(cores_left,cores,cores_right)
  return ReducedProjection(proj_basis)
end

function empirical_interpolation(a::TTSVDProjection)
  cores = get_cores(a)
  dof_map = get_dof_map(a)

  ptrs = Vector{Int32}(undef,length(cores)+1)
  for i in eachindex(cores)
    ptrs[i+1] = size(cores[i],3)
  end
  length_to_ptrs!(ptrs)

  interp = ones(1,1)
  data = fill(zero(Int32),ptrs[end]-1)
  for i = eachindex(cores)
    interp_core = reshape(interp,1,size(interp)...)
    c = cores2basis(interp_core,cores[i])
    inds,interp = empirical_interpolation(c)
    pini = ptrs[i]
    pend = ptrs[i+1]-1
    for (k,pk) in enumerate(pini:pend)
      data[pk] = inds[k]
    end
  end
  linds = Table(data,ptrs)
  ginds = get_basis_indices(linds,dof_map)

  return ginds,interp
end

function rescale(op::Function,X::AbstractRankTensor{D},b::TTSVDProjection) where D
  cores = get_cores(b)
  dof_map = get_dof_map(b)
  if D == ndims(dof_map)
    TTSVDProjection(op(X,cores),dof_map)
  else
    c1 = op(X,cores[1:D])
    c2 = cores[D+1:end]
    TTSVDProjection([c1...,c2...],dof_map)
  end
end

"""
    struct NormedProjection <: Projection
      projection::Projection
      norm_matrix::MatrixOrTensor
    end

Represents a `Projection` `projection` spanning a space equipped with an inner
product represented by the quantity `norm_matrix`
"""
struct NormedProjection <: Projection
  projection::Projection
  norm_matrix::MatrixOrTensor
end

get_projection(a::Projection) = a
get_projection(a::NormedProjection) = a.projection
get_norm_matrix(a::NormedProjection) = a.norm_matrix

get_basis(a::NormedProjection) = get_basis(a.projection)
num_fe_dofs(a::NormedProjection) = num_fe_dofs(a.projection)
num_reduced_dofs(a::NormedProjection) = num_reduced_dofs(a.projection)

get_cores(a::NormedProjection) = get_cores(a.projection)
DofMaps.get_dof_map(a::NormedProjection) = get_dof_map(a.projection)

function project!(x̂::AbstractArray,a::NormedProjection,x::AbstractArray)
  project!(x̂,a.projection,x,a.norm_matrix)
end

function inv_project!(x::AbstractArray,a::NormedProjection,x̂::AbstractArray)
  inv_project!(x,a.projection,x̂)
end

function union_bases(a::NormedProjection,b::NormedProjection,args...)
  projection′ = union_bases(a.projection,b.projection,args...)
  NormedProjection(projection′,a.norm_matrix)
end

function union_bases(a::NormedProjection,b::AbstractArray,args...)
  projection′ = union_bases(a.projection,b,args...)
  NormedProjection(projection′,a.norm_matrix)
end

function galerkin_projection(proj_left::NormedProjection,a::Projection)
  galerkin_projection(proj_left.projection,get_projection(a))
end

function galerkin_projection(proj_left::NormedProjection,a::Projection,proj_right::NormedProjection,args...)
  galerkin_projection(proj_left.projection,get_projection(a),proj_right.projection,args...)
end

function empirical_interpolation(a::NormedProjection)
  empirical_interpolation(a.projection)
end

function rescale(op::Function,x::Any,b::NormedProjection)
  projection′ = rescale(op,x,b.projection)
  NormedProjection(projection′,a.norm_matrix)
end

# multi field interface

function Arrays.return_type(::typeof(projection),::PODReduction,::Snapshots)
  PODProjection
end

function Arrays.return_type(::typeof(projection),::TTSVDReduction,::Snapshots)
  TTSVDProjection
end

function Arrays.return_type(::typeof(projection),::Reduction,::Snapshots,::MatrixOrTensor)
  NormedProjection
end

function Arrays.return_cache(::typeof(projection),red::Reduction,s::BlockSnapshots)
  i = findfirst(s.touched)
  @notimplementedif isnothing(i)
  A = return_type(projection,red,s[i])
  block_basis = Array{A,ndims(s)}(undef,size(s))
  touched = s.touched
  return BlockProjection(block_basis,touched)
end

function Arrays.return_cache(::typeof(projection),red::Reduction,s::BlockSnapshots,X::MatrixOrTensor)
  i = findfirst(s.touched)
  @notimplementedif isnothing(i)
  A = return_type(projection,red,s[i],X[Block(i,i)])
  block_basis = Array{A,ndims(s)}(undef,size(s))
  touched = s.touched
  return BlockProjection(block_basis,touched)
end

function projection(red::Reduction,s::BlockSnapshots)
  basis = return_cache(projection,red,s)
  for i in eachindex(basis)
    if basis.touched[i]
      basis[i] = projection(red,s[i])
    end
  end
  return basis
end

function projection(red::Reduction,s::BlockSnapshots,X::MatrixOrTensor)
  basis = return_cache(projection,red,s,X)
  for i in eachindex(basis)
    if basis.touched[i]
      basis[i] = projection(red,s[i],X[Block(i,i)])
    end
  end
  return basis
end

"""
    struct BlockProjection{A,N} <: Projection end

Block container for Projection of type `A` in a `MultiField` setting. This
type is conceived similarly to `ArrayBlock` in `Gridap`
"""
struct BlockProjection{A<:Projection,N} <: Projection
  array::Array{A,N}
  touched::Array{Bool,N}

  function BlockProjection(
    array::Array{A,N},
    touched::Array{Bool,N}
    ) where {A<:Projection,N}

    @check size(array) == size(touched)
    new{A,N}(array,touched)
  end
end

function BlockProjection(a::AbstractArray{A},touched::Array{Bool,N}) where {A<:Projection,N}
  array = Array{A,N}(undef,k.size)
  for i in touched
    if touched[i]
      array[i] = a[i]
    end
  end
  BlockProjection(array,touched)
end

Base.ndims(a::BlockProjection) = ndims(a.touched)
Base.size(a::BlockProjection,args...) = size(a.touched,args...)
Base.axes(a::BlockProjection,args...) = axes(a.touched,args...)
Base.length(a::BlockProjection) = length(a.touched)
Base.eachindex(a::BlockProjection) = eachindex(a.touched)

function Base.getindex(a::BlockProjection,i...)
  if !a.touched[i...]
    return nothing
  end
  a.array[i...]
end

function Base.setindex!(a::BlockProjection,v,i...)
  @check a.touched[i...] "Only touched entries can be set"
  a.array[i...] = v
end

Base.getindex(a::BlockProjection,i::Block) = getindex(a,i.n...)
Base.setindex!(a::BlockProjection,v,i::Block) = setindex!(a,v,i.n...)

function Arrays.testitem(a::BlockProjection)
  i = findall(a.touched)
  @notimplementedif length(i) == 0
  a.array[first(i)]
end

function num_fe_dofs(a::BlockProjection)
  dofs = zeros(Int,length(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_fe_dofs(a[i])
    end
  end
  return dofs
end

function num_reduced_dofs(a::BlockProjection)
  dofs = zeros(Int,size(a))
  for i in eachindex(a)
    if a.touched[i]
      dofs[i] = num_reduced_dofs(a[i])
    end
  end
  return dofs
end

for f in (:(Algebra.allocate_in_domain),:(Algebra.allocate_in_range))
  @eval begin
    function $f(a::BlockProjection,x::Union{BlockVector,BlockParamVector})
      @check length(a) == blocklength(x)
      @notimplementedif !all(a.touched)
      mortar(map(i -> $f(a[Block(i)],x[Block(i)]),eachindex(a)))
    end
  end
end

for f in (:project!,:inv_project!)
  @eval begin
    function $f(
      y::Union{BlockArray,BlockParamArray},
      a::BlockProjection,
      x::Union{BlockArray,BlockParamArray})

      for i in eachindex(a)
        if a.touched[i]
          yi = y[Block(i)]
          $f(yi,a[i],x[Block(i)])
        end
      end
    end
  end
end

function Arrays.return_cache(::typeof(get_norm_matrix),a::BlockProjection)
  i = findfirst(s.touched)
  @notimplementedif isnothing(i)
  A = typeof(get_norm_matrix(a[i]))
  norm_matrix = Array{A,ndims(s)}(undef,size(s))
  return norm_matrix
end

function get_norm_matrix(a::BlockProjection)
  norm_matrix = return_cache(get_norm_matrix,a)
  for i in eachindex(a)
    if a.touched[i]
      norm_matrix[Block(i,i)] = get_norm_matrix(a[i])
    end
  end
  return norm_matrix
end

"""
    enrich!(
      red::SupremizerReduction,
      a::BlockProjection,
      norm_matrix::MatrixOrTensor,
      supr_matrix::MatrixOrTensor) -> Nothing

In-place augmentation of the primal block of a [`BlockProjection`](@ref) `a`.
This function has the purpose of stabilizing the reduced equations stemming from
a saddle point problem
"""
function enrich!(
  red::SupremizerReduction,
  a::BlockProjection,
  norm_matrix::BlockMatrix,
  supr_matrix::BlockMatrix)

  @check a.touched[1] "Primal field not defined"
  red_style = ReductionStyle(red)
  a_primal,a_dual... = a.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = cholesky(X_primal)
  for i = eachindex(a_dual)
    dual_i = get_basis(a_dual[i])
    C_primal_dual_i = supr_matrix[Block(1,i+1)]
    supr_i = H_primal \ C_primal_dual_i * dual_i
    a_primal = union_bases(a_primal,supr_i,X_primal,red_style)
  end
  a[1] = a_primal
  return
end

function enrich!(
  red::SupremizerReduction{A,<:TTSVDReduction},
  a::BlockProjection,
  norm_matrix::BlockRankTensor,
  supr_matrix::BlockRankTensor
  ) where A

  @check a.touched[1] "Primal field not defined"
  red_style = ReductionStyle(red)
  a_primal,a_dual... = a.array
  primal_map = get_dof_map(a_primal)
  X_primal = norm_matrix[Block(1,1)]
  for i = eachindex(a_dual)
    dual_i = get_cores(a_dual[i])
    C_primal_dual_i = supr_matrix[Block(1,i+1)]
    supr_i = tt_supremizer(X_primal,C_primal_dual_i,dual_i)
    a_primal = union_bases(a_primal,supr_i,red_style,X_primal)
  end
  a[1] = a_primal
  return
end
