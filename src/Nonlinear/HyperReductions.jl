const NNHRProjection{A<:Projection,B<:AbstractNNHyperReduction} = HRProjection{A,B}

struct NNOperator{A,N} <: NNHRProjection{ReducedProjection{AbstractArray{Number,N}},NNOperatorReduction}
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

function FESpaces.interpolate!(
  b̂::AbstractArray,
  cache,
  a::NNHRProjection{<:Projection,<:NNHyperReduction},
  x::Any
  )

  o = one(eltype2(b̂))
  coeff = interpolate!(cache,get_interpolation(a),x)
  _mul!(b̂,a,coeff,o,o)
  return b̂
end

function RBSteady.allocate_coefficient(a::NNHRProjection{<:Projection,<:NNHyperReduction},r::AbstractRealisation)
  x = matrix_of_params(r)
  i = get_interpolation(a)
  return_cache(i.interpolation,x)
end

const NNContribution = AffineContribution{<:NNHRProjection}

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
  ϕ = get_basis(y)
  model = TrainedNeuralNetwork(get_strategy(red),r,ϕ)
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
  ϕ = permutedims(get_basis(y),(1,3,2))
  model = TrainedNeuralNetwork(get_strategy(red),r,ϕ)
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

_mul!(a,b,c,α,β) = @abstractmethod 

function _mul!(a::AbstractParamArray,b,c::AbstractMatrix,α,β) 
  mul!(get_all_data(a),b,c,α,β)
end