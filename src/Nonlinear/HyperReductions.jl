"""
    struct NNOperator{A,N} <: HRProjection{…,NNOperatorReduction}
      model::A
      reduced_sizes::NTuple{N,Int}
    end

[`HRProjection`](@ref) for operator regression (`NNOperatorReduction` strategy). The
`model` maps parameter matrices `(d×k)` directly to projected operator values,
bypassing FE assembly during the online phase.

`reduced_sizes` stores the output shape per parameter sample:
- `(nrows, 1)` for a projected residual vector of length `nrows`
- `(nrows, 1, ncols)` for a projected Jacobian matrix of size `nrows×ncols`

The middle entry `1` is a dummy so that `N=2` (vector) and `N=3` (matrix) give
distinct type parameter `N` while `last(reduced_sizes)` always returns the column
count (1 for vectors, `ncols` for matrices).

Constructed automatically by [`HRProjection(::NNOperatorReduction, …)`](@ref).
"""
struct NNOperator{A,N} <: HRProjection{ReducedProjection{AbstractArray{Number,N}},NNOperatorReduction}
  model::A
  reduced_sizes::NTuple{N,Int}
end

function NNOperator(model::NeuralNetwork,test::RBSpace) 
  nrows = num_reduced_dofs(test)
  NNOperator(model,(nrows,1))
end

function NNOperator(model::NeuralNetwork,trial::RBSpace,test::RBSpace) 
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  NNOperator(model,(nrows,1,ncols))
end

RBSteady.get_interpolation(a::NNOperator) = EmptyInterpolation()
RBSteady.projection_eltype(a::NNOperator) = eltype(get_weights(a.model))

RBSteady.num_reduced_dofs(a::NNOperator) = 1
RBSteady.num_reduced_dofs_left_projector(a::NNOperator) = first(a.reduced_sizes)
RBSteady.num_reduced_dofs_right_projector(a::NNOperator) = last(a.reduced_sizes)

function FESpaces.interpolate!(
  b̂::AbstractArray,
  cache,
  a::NNOperator,
  r::AbstractRealisation
  )

  b̂r = evaluate!(cache,a.model,matrix_of_params(r))
  o = one(eltype2(b̂))
  _axpy!(o,b̂r,b̂)
  return b̂
end

function RBSteady.allocate_coefficient(a::NNOperator,r::AbstractRealisation)
  x = matrix_of_params(r)
  return_cache(a.model,x)
end

const NNHRProjection{A<:Projection} = HRProjection{A,<:NNHyperReduction}

"""
    const NNContribution = AffineContribution{<:NNOperator}

[`AffineContribution`](@ref) backed by [`NNOperator`](@ref)s.
"""
const NNContribution = AffineContribution{<:NNOperator}

function FESpaces.interpolate!(
  hypred::AbstractArray,
  cache,
  a::NNContribution,
  r::AbstractRealisation
  )

  fill!(hypred,zero(eltype(hypred)))
  for aval in get_contributions(a)
    interpolate!(hypred,cache,aval,r)
  end
  return hypred
end

function RBSteady.allocate_coefficient(a::NNContribution,args...)
  allocate_coefficient(first(get_contributions(a)),args...)
end

function RBSteady.allocate_hypred_cache(a::NNContribution,args...)
  fecache = allocate_coefficient(a,args...)
  coeffs = fecache
  hypred = allocate_hyper_reduction(a,args...)
  return HRParamArray(fecache,coeffs,hypred)
end

function RBSteady.HRProjection(
  red::NNOperatorReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  r = get_realisation(s)
  b = GalerkinProjectable(s)
  y = galerkin_projection(test,b)
  model = TrainedNeuralNetwork(get_strategy(red),r,get_basis(y))
  return NNOperator(model,test)
end

function RBSteady.HRProjection(
  red::NNOperatorReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  r = get_realisation(s)
  A = GalerkinProjectable(s)
  y = galerkin_projection(test,A,trial)
  model = TrainedNeuralNetwork(get_strategy(red),r,get_basis(y))
  return NNOperator(model,trial,test)
end

function RBSteady.HRProjection(
  red::NNHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis)
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.HRProjection(
  red::NNHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial)
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

# utils 

_axpy!(α,a,b) = @abstractmethod 

function _axpy!(α,a::AbstractMatrix,b::AbstractParamVector) 
  axpy!(α,a,get_all_data(b))
end

function _axpy!(α,a::AbstractMatrix,b::AbstractParamMatrix)
  nrows,ncols = innersize(b)
  k = param_length(b)
  a′ = reshape(a,nrows,ncols,k)
  axpy!(α,a′,get_all_data(b))
end
