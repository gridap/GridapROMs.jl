"""
    abstract type Projection <: Map end

Represents a basis for a `n`-dimensional vector subspace of a `N`-dimensional
vector space (where `N >> n`), to be used as a Galerkin projection operator.
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
num_fe_dofs(a::Projection) = size(get_basis(a),1)

fe_dof_ids(a::Projection) = Base.OneTo(num_fe_dofs(a))

"""
    num_reduced_dofs(a::Projection) -> Int

For a projection map `a` from a low dimensional space `n` to a high dimensional
one `N`, returns `n`
"""
num_reduced_dofs(a::Projection) = size(get_basis(a),2)

reduced_dof_ids(a::Projection) = Base.OneTo(num_reduced_dofs(a))

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

function project!(
  x̂::AbstractArray,
  a::Projection,
  x::AbstractArray,
  norm_matrix::AbstractMatrix
  )

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

projection_type(a::Projection) = Vector{projection_eltype(a)}

"""
    projection_eltype(a::Projection) -> DataType

Returns the eltype of the projection `a`
"""
projection_eltype(a::Projection) = eltype(get_basis(a))

function Algebra.allocate_in_domain(a::Projection) 
  V = projection_type(a)
  x̂ = allocate_vector(V,reduced_dof_ids(a))
  return x̂
end

function Algebra.allocate_in_range(a::Projection) 
  V = projection_type(a)
  x = allocate_vector(V,fe_dof_ids(a))
  return x
end

function Algebra.allocate_in_domain(a::Projection,x::V) where V<:AbstractVector
  x̂ = allocate_vector(V,reduced_dof_ids(a))
  return x̂
end

function Algebra.allocate_in_range(a::Projection,x̂::V) where V<:AbstractVector
  x = allocate_vector(V,fe_dof_ids(a))
  return x
end

function Algebra.allocate_in_domain(a::Projection,X::M) where M<:AbstractMatrix
  X̂ = allocate_full_matrix(M,reduced_dof_ids(a),Base.OneTo(size(X,2)))
  return X̂
end

function Algebra.allocate_in_range(a::Projection,X̂::M) where M<:AbstractMatrix
  X = allocate_full_matrix(M,fe_dof_ids(a),Base.OneTo(size(X̂,2)))
  return X
end

function Algebra.allocate_in_domain(a::Projection,x::V) where V<:AbstractParamVector
  x̂ = allocate_vector(eltype(V),reduced_dof_ids(a))
  return parameterise(x̂,param_length(x))
end

function Algebra.allocate_in_range(a::Projection,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(eltype(V),fe_dof_ids(a))
  return parameterise(x,param_length(x̂))
end

function allocate_full_matrix(::Type{M},rows::AbstractVector,cols::AbstractVector) where M
  zeros(eltype(M),length(rows),length(cols))
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
    DEIM(a::Projection) -> (AbstractVector,AbstractMatrix)

Computes the EIM of `a`. The outputs are:

- a vector of integers `i`, corresponding to a list of interpolation row indices
- a matrix `Φi = view(Φ,i)`, where `Φ = get_basis(a)`. This quantity represents
  the restricted basis on the set of interpolation rows `i`
"""
DEIM(a::Projection) = DEIM(get_basis(a))

"""
    SOPT(a::Projection) -> (AbstractVector,AbstractMatrix)

Computes the S-OPT hyper-reduction of `a`. Check [this](https://arxiv.org/abs/2203.16494)
reference for more information
"""
SOPT(a::Projection) = SOPT(get_basis(a))

"""
    union_bases(a::Projection,b::Projection,args...) -> Projection

Computes the projection corresponding to the union of `a` and `b`. In essence this
operation performs as

  `gram_schmidt(union(get_basis(a),get_basis(b)))`
"""
union_bases(a::Projection,b::Projection,args...) = @abstractmethod

gram_schmidt(a::Projection,b::Projection,args...) = gram_schmidt(get_basis(a),get_basis(b),args...)

get_norm_matrix(a::Projection) = I(num_fe_dofs(a))

Base.:+(a::Projection,b::Projection) = union_bases(a,b)
Base.:-(a::Projection,b::Projection) = union_bases(a,b)
Base.:*(a::Projection,b::Projection) = galerkin_projection(a,b)
Base.:*(a::Projection,b::Projection,c::Projection) = galerkin_projection(a,b,c)
Base.:*(a::Projection,x::AbstractArray) = inv_project(a,x)

function Base.:*(b::Projection,y::ConsecutiveParamArray{T}) where T
  S = promote_type(projection_eltype(b),T)
  item = zeros(S,num_reduced_dofs(b))
  plength = param_length(y)
  x = parameterise(item,plength)
  mul!(x,b,y)
end

function LinearAlgebra.mul!(
  x::AbstractArray,
  b::Projection,
  y::AbstractArray,
  α::Number,
  β::Number
  )

  mul!(x,get_basis(b),y,α,β)
end

function LinearAlgebra.mul!(
  x::ConsecutiveParamArray,
  b::Projection,
  y::ConsecutiveParamArray,
  α::Number,
  β::Number
  )
  
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

function projection(red::Reduction,s::Snapshots)
  Projection(red,s)
end

function projection(red::Reduction,s::Snapshots,X::MatrixOrTensor)
  proj = Projection(red,s,X)
  NormedProjection(proj,X)
end

function Projection(red::Reduction,s::AbstractArray,args...)
  ŝ = reduction(red,s,args...)
  Projection(ŝ,s)
end

function Projection(basis::AbstractArray,s::AbstractArray)
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
const ReducedMatProjection = ReducedProjection{<:AbstractArray{<:Any,3}}

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
  α::Number,β::Number
  )

  contraction!(x,get_basis(b),y,α,β)
end

function LinearAlgebra.mul!(
  x::ConsecutiveParamArray,
  b::ReducedMatProjection,
  y::ConsecutiveParamArray,
  α::Number,β::Number
  )

  contraction!(get_all_data(x),get_basis(b),get_all_data(y),α,β)
end

"""
"""
struct ReducedAlgebraicProjection{A} <: ReducedProjection{A}
  basis::A
end

ReducedProjection(basis::AbstractArray) = ReducedAlgebraicProjection(basis)

Base.eltype(::Type{<:ReducedAlgebraicProjection{A}}) where A = eltype(A)
Base.ndims(::Type{<:ReducedAlgebraicProjection{A}}) where A = ndims(A)

get_basis(a::ReducedAlgebraicProjection) = a.basis
num_reduced_dofs(a::ReducedAlgebraicProjection) = size(get_basis(a),2)
num_reduced_dofs_left_projector(a::ReducedAlgebraicProjection) = size(get_basis(a),1)
num_reduced_dofs_right_projector(a::ReducedMatProjection) = size(get_basis(a),3)

"""
    struct PODProjection <: Projection
      basis::AbstractMatrix
    end

Projection stemming from a truncated proper orthogonal decomposition [`tpod`](@ref)
"""
struct PODProjection <: Projection
  basis::AbstractMatrix
end

function Projection(basis::AbstractMatrix,s::AbstractArray)
  PODProjection(basis)
end

function Projection(basis::AbstractMatrix,s::SparseSnapshots)
  basis′ = recast(basis,s)
  PODProjection(basis′)
end

get_basis(a::PODProjection) = a.basis

union_bases(a::PODProjection,b::PODProjection,args...) = union_bases(a,get_basis(b),args...)

function union_bases(a::PODProjection,basis_b::AbstractMatrix,args...)
  basis_a = get_basis(a)
  basis_ab = gram_schmidt(basis_b,basis_a,args...)
  PODProjection(basis_ab)
end

# TT interface

"""
    struct TTSVDProjection <: Projection
      cores::AbstractVector{<:AbstractArray{<:Any,3}}
      dof_map::AbstractDofMap
    end

Projection stemming from a tensor train SVD [`ttsvd`](@ref). For reindexing purposes
a field `dof_map` is provided along with the tensor train cores `cores`
"""
struct TTSVDProjection <: Projection
  cores::AbstractVector{<:AbstractArray{<:Any,3}}
  dof_map::AbstractDofMap
end

function Projection(cores::AbstractVector{<:AbstractArray},s::AbstractArray)
  dof_map = get_dof_map(s)
  TTSVDProjection(cores,dof_map)
end

function Projection(cores::AbstractVector{<:AbstractArray},s::SparseSnapshots)
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

function project!(
  x̂::AbstractArray,
  a::TTSVDProjection,
  x::AbstractArray,
  norm_matrix::AbstractRankTensor
  )

  a′ = PODProjection(get_basis(a))
  k_norm_matrix = kron(norm_matrix)
  k_norm_matrix′ = _make_compatible(k_norm_matrix,a′)
  k_a′ = NormedProjection(a′,k_norm_matrix′)
  project!(x̂,k_a′,x)
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

  cores_a = get_cores(a)
  @check length(cores_a) == length(cores_b)

  cores_ab = block_cores([cores_a,cores_b])
  orthogonalize!(cores_ab,args...)
  TTSVDProjection(cores_ab,get_dof_map(a))
end

function galerkin_projection(proj_left::TTSVDProjection,a::TTSVDProjection)
  cores_left = get_cores(proj_left)
  cores = get_cores(a)
  proj_basis = galerkin_projection(cores_left,cores)
  return ReducedProjection(proj_basis)
end

function galerkin_projection(
  proj_left::TTSVDProjection,
  a::TTSVDProjection,
  proj_right::TTSVDProjection
  )

  _galerkin_projection(get_dof_map(a),proj_left,a,proj_right)
end

function projection_eltype(a::TTSVDProjection)
  promote_type(map(eltype,get_cores(a))...)
end

for f in (:DEIM,:SOPT)
  @eval begin
    function $f(a::TTSVDProjection)
      cores = get_cores(a)
      dof_map = get_dof_map(a)

      ptrs = Vector{Int32}(undef,length(cores)+1)
      for i in eachindex(cores)
        ptrs[i+1] = size(cores[i],3)
      end
      length_to_ptrs!(ptrs)

      T = projection_eltype(a)
      interp = ones(T,1,1)
      data = fill(zero(Int32),ptrs[end]-1)
      for i = eachindex(cores)
        interp_core = reshape(interp,1,size(interp)...)
        c = cores2basis(interp_core,cores[i])
        inds,interp = $f(c)
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
projection_type(a::NormedProjection) = projection_type(a.projection)

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
  galerkin_projection(get_projection(proj_left),get_projection(a))
end

function galerkin_projection(
  proj_left::NormedProjection,
  a::Projection,
  proj_right::NormedProjection,
  args...
  )

  galerkin_projection(get_projection(proj_left),get_projection(a),get_projection(proj_right),args...)
end

for f in (:DEIM,:SOPT)
  @eval begin
    $f(a::NormedProjection) = $f(a.projection)
  end
end

# multi field interface

function projection(red::Reduction,s::AbstractBlockSnapshots)
  basis = _allocate_projection(red,s)
  for i in eachindex(basis)
    basis[i] = projection(red,s[i])
  end
  return basis
end

function projection(red::Reduction,s::AbstractBlockSnapshots,X::MatrixOrTensor)
  basis = _allocate_projection(red,s)
  for i in eachindex(basis)
    basis[i] = projection(red,s[i],X[Block(i,i)])
  end
  return basis
end

"""
    struct BlockProjection{A<:Projection,N} <: Projection
      array::Array{A,N}
    end

Block container for Projection of type `A` in a `MultiField` setting. This
type is conceived similarly to `ArrayBlock` in [`Gridap`](@ref). Every block is
always populated.
"""
struct BlockProjection{A<:Projection,N} <: Projection
  array::Array{A,N}
end

Base.ndims(a::BlockProjection) = ndims(a.array)
Base.size(a::BlockProjection,args...) = size(a.array,args...)
Base.axes(a::BlockProjection,args...) = axes(a.array,args...)
Base.length(a::BlockProjection) = length(a.array)
Base.eachindex(a::BlockProjection) = eachindex(a.array)

Base.getindex(a::BlockProjection,i...) = a.array[i...]
Base.setindex!(a::BlockProjection,v,i...) = (a.array[i...] = v)

Base.getindex(a::BlockProjection,i::Block) = getindex(a,i.n...)
Base.setindex!(a::BlockProjection,v,i::Block) = setindex!(a,v,i.n...)

Arrays.testitem(a::BlockProjection) = first(a.array)

function get_basis(a::BlockProjection{A,N}) where {A,N}
  return map(get_basis,a.array)
end

function num_fe_dofs(a::BlockProjection)
  dofs = 0
  for i in eachindex(a)
    dofs += num_fe_dofs(a[i])
  end
  return dofs
end

function num_reduced_dofs(a::BlockProjection)
  dofs = 0
  for i in eachindex(a)
    dofs += num_reduced_dofs(a[i])
  end
  return dofs
end

for (f,g) in zip((:to_fe_blocks,:to_reduced_blocks),(:num_fe_dofs,:num_reduced_dofs))
  @eval begin
    function $f(x::Union{BlockVector,BlockParamVector},a::BlockProjection,args...)
      x
    end

    function $f(x,a::BlockProjection,args...)
      ids = map($g,a.array)
      pushfirst!(ids,1)
      to_blocks(x,cumsum(ids),args...)
    end
  end
end

function to_blocks(x::AbstractVector,o,f=identity)
  n = length(o)-1
  mortar(map(i -> f(view(x,o[i]:o[i+1]-1)),1:n))
end

function to_blocks(x::AbstractParamVector,o,f=identity)
  n = length(o)-1
  mortar(map(i -> f(get_param_entry(x,o[i]:o[i+1]-1)),1:n))
end

for f in (:allocate_in_domain,:allocate_in_range)
  @eval begin
    function Algebra.$f(a::BlockProjection)
      mortar(map(Algebra.$f,a.array))
    end

    function Algebra.$f(a::BlockProjection,x::BlockVector)
      @check length(a) == blocklength(x)
      mortar(map(i -> Algebra.$f(a[Block(i)],x[Block(i)]),eachindex(a)))
    end

    function Algebra.$f(a::BlockProjection,x::BlockParamVector)
      @check length(a) == blocklength(x)
      mortar(map(i -> Algebra.$f(a[Block(i)],x[Block(i)]),eachindex(a)))
    end
  end
end

for (f,g) in zip((:project!,:inv_project!),(:to_fe_blocks,:to_reduced_blocks))
  ginv = g == :to_fe_blocks ? :to_reduced_blocks : :to_fe_blocks
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

    function $f(
      y::Union{AbstractArray,AbstractParamArray},
      a::BlockProjection,
      x::Union{AbstractArray,AbstractParamArray}
      )

      $f($ginv(y,a),a,$g(x,a))
    end
  end
end

function galerkin_projection(
  proj_left::BlockProjection{A,1},
  a::BlockProjection{B,1},
  args...
  ) where {A,B}

  @check length(proj_left) == size(a,1)
  block_cache = Vector{Projection}(undef,length(a))
  for i in eachindex(a)
    block_cache[i] = galerkin_projection(proj_left[i],a[i],args...)
  end
  return BlockProjection(block_cache)
end

function galerkin_projection(
  proj_left::BlockProjection{A,1},
  a::BlockProjection{B,2},
  proj_right::BlockProjection{A,1},
  args...
  ) where {A,B}

  @check length(proj_left) == size(a,1)
  @check length(proj_right) == size(a,2)
  block_cache = Matrix{Projection}(undef,size(a))
  for i in axes(a,1), j in axes(a,2)
    block_cache[i,j] = galerkin_projection(proj_left[i],a[i,j],proj_right[j],args...)
  end
  return BlockProjection(block_cache)
end

function ReducedProjection(basis::VectorBlock)
  block_cache = map(ReducedProjection,basis.array)
  return BlockProjection(block_cache)
end

function get_norm_matrix(a::BlockProjection)
  norm_matrix = _allocate_norm_matrix(a)
  for i in eachindex(a)
    norm_matrix[Block(i,i)] = get_norm_matrix(a[i])
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
  ::SupremizerReduction,
  a::BlockProjection,
  norm_matrix::BlockMatrix,
  supr_matrix::BlockMatrix
  )

  a_primal,a_dual... = a.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = symcholesky(X_primal)
  for i = eachindex(a_dual)
    dual_i = get_basis(a_dual[i])
    C_primal_dual_i = supr_matrix[Block(1,i+1)]
    supr_i = supremizers(H_primal,C_primal_dual_i,dual_i)
    a_primal = union_bases(a_primal,supr_i,H_primal)
  end
  a[1] = a_primal
  return
end

function enrich!(
  ::SupremizerReduction{A,B,<:TTSVDReduction},
  a::BlockProjection,
  norm_matrix::BlockRankTensor,
  supr_matrix::BlockRankTensor
  ) where {A,B}

  a_primal,a_dual... = a.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = symcholesky(X_primal)
  for i = eachindex(a_dual)
    dual_i = get_cores(a_dual[i])
    C_primal_dual_i = supr_matrix[Block(1,i+1)]
    supr_i = tt_supremizers(H_primal,C_primal_dual_i,dual_i)
    a_primal = union_bases(a_primal,supr_i,X_primal)
  end
  a[1] = a_primal
  return
end

function supremizers(f::Factorization,C::AbstractMatrix,ϕ::AbstractMatrix)
  c1 = similar(ϕ,size(C,1),size(ϕ,2))
  c2 = similar(ϕ,size(C,1),size(ϕ,2))
  mul!(c1,C,ϕ)
  ldiv!(c2,f,c1)
  return c2
end

# galerkin projections

struct GalerkinProjectable{A<:AbstractParamArray} <: Projection
  array::A
end

get_basis(a::GalerkinProjectable) = a.array

function GalerkinProjectable(s::AbstractSnapshots)
  GalerkinProjectable(get_param_data(s))
end

function GalerkinProjectable(a::BlockParamArray{T,N}) where {T,N}
  block_cache = map(GalerkinProjectable,blocks(a))
  return BlockProjection(block_cache)
end

function GalerkinProjectable(a::AbstractArray{<:AbstractArray})
  block_cache = map(GalerkinProjectable,a)
  return BlockProjection(block_cache)
end

function galerkin_projection(a::Projection,b,args...)
  galerkin_projection(a,GalerkinProjectable(b),args...)
end

function galerkin_projection(a::Projection,b,c::Projection,args...)
  galerkin_projection(a,GalerkinProjectable(b),c,args...)
end

function copy_projection!(cache,a::Projection)
  copy_projection!(cache,get_basis(a))
end

# utils

function _allocate_projection(red::Reduction,s::AbstractBlockSnapshots{<:Any,N}) where N
  T = _proj_type(red)
  block_basis = Array{T,N}(undef,size(s))
  BlockProjection(block_basis)
end

function _allocate_norm_matrix(a::BlockProjection{A,N}) where {A,N}
  ai = testitem(a)
  T = typeof(get_norm_matrix(ai))
  Array{T,N}(undef,size(a))
end

function _galerkin_projection(
  ::AbstractDofMap,
  proj_left::TTSVDProjection,
  a::TTSVDProjection,
  proj_right::TTSVDProjection
  )

  cores_left = get_cores(proj_left)
  cores = get_cores(a)
  cores_right = get_cores(proj_right)
  proj_basis = galerkin_projection(cores_left,cores,cores_right)
  return ReducedProjection(proj_basis)
end

function _galerkin_projection(
  ::TrivialDofMap,
  proj_left::TTSVDProjection,
  a::TTSVDProjection,
  proj_right::TTSVDProjection
  )

  proj_basis = galerkin_projection(get_basis(proj_left),get_basis(a),get_basis(proj_right))
  return ReducedProjection(proj_basis)
end

_proj_type(red::Reduction) = _proj_type(NormStyle(red),red)
_proj_type(::NormStyle,::Reduction) = @abstractmethod
_proj_type(::EuclideanNorm,::PODReduction) = PODProjection
_proj_type(::EuclideanNorm,::TTSVDReduction) = TTSVDProjection
_proj_type(::AssembleOperator,::DirectReduction) = NormedProjection
_proj_type(::AssembleOperator,::LocalReduction) = LocalProjection

function _make_compatible(X::AbstractMatrix,a::Projection)
  size(X,1) == num_fe_dofs(a) && return X 
  d = Int(num_fe_dofs(a) / size(X,1))
  kron(I(d),X)
end