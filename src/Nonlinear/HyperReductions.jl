struct NNProjection{A,N} <: HRProjection{ReducedProjection{AbstractArray{Number,N}},NNRegression}
  model::A
  reduced_sizes::NTuple{N,Int}
end

function NNProjection(model::NeuralNetwork,test::RBSpace) 
  nrows = num_reduced_dofs(test)
  NNProjection(model,(nrows,1))
end

function NNProjection(model::NeuralNetwork,trial::RBSpace,test::RBSpace) 
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  NNProjection(model,(nrows,1,ncols))
end

RBSteady.get_interpolation(a::NNProjection) = EmptyInterpolation()
RBSteady.projection_eltype(a::NNProjection) = eltype(get_weights(a.model))

RBSteady.num_reduced_dofs(a::NNProjection) = 1
RBSteady.num_reduced_dofs_left_projector(a::NNProjection) = first(a.reduced_sizes)
RBSteady.num_reduced_dofs_right_projector(a::NNProjection) = last(a.reduced_sizes)

function FESpaces.interpolate!(
  b̂::AbstractArray,
  cache,
  a::NNProjection,
  r::AbstractRealisation
  )

  b̂r = evaluate!(cache,a.model,r)
  o = one(eltype2(b̂))
  axpy!(o,b̂r,b̂)
  return b̂
end

function RBSteady.allocate_coefficient(a::NNProjection,r::AbstractRealisation)
  x = matrix_of_params(r)
  return_cache(a.model,x)
end

"""
"""
const NNContribution = AffineContribution{<:NNProjection}

function FESpaces.interpolate!(
  hypred::AbstractArray,
  coeff::AbstractArray,
  a::NNContribution,
  r::AbstractRealisation
  )

  fill!(hypred,zero(eltype(hypred)))
  for aval in get_contributions(a)
    interpolate!(hypred,coeff,aval,r)
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

# HRProjection constructors for NNRegression

function RBSteady.HRProjection(
  red::NNRegression,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  r = get_realisation(s)
  b = GalerkinProjectable(s)
  y = galerkin_projection(test,b)
  model = TrainedNeuralNetwork(get_strategy(red),r,y)
  return NNProjection(model,test)
end

function RBSteady.HRProjection(
  red::NNRegression,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  r = get_realisation(s)
  A = GalerkinProjectable(s)
  y = galerkin_projection(test,A,trial)
  model = TrainedNeuralNetwork(get_strategy(red),r,y)
  return NNProjection(model,trial,test)
end

# HRProjection constructors for NNHyperReduction

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
