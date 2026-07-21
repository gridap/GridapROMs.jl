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

function RBSteady.get_basis(a::NNOperator)
  T = projection_eltype(a)
  nrows = num_reduced_dofs_left_projector(a)
  ncols = num_reduced_dofs_right_projector(a)
  ncols == 1 ? ReducedProjection(zeros(T,nrows,1)) : ReducedProjection(zeros(T,nrows,1,ncols))
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

# multi-field interface 

function RBSteady.allocate_coefficient(
  a::BlockHRProjection{N,<:Any,<:AbstractNNHyperReduction},
  r::AbstractRealisation
  ) where N

  i0 = findfirst(a.touched)
  A = typeof(allocate_coefficient(a.array[i0],r))
  block_cache = Array{A,N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      block_cache[i] = allocate_coefficient(a.array[i],r)
    end
  end
  return ArrayBlock(block_cache,a.touched)
end

# transient interface

function RBSteady.HRProjection(
  red::HighDimNNOperatorReduction,
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
  red::HighDimNNOperatorReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  r = get_realisation(s)
  A = GalerkinProjectable(s)
  y = galerkin_projection(test,A,trial,get_time_combination(red))
  ϕ = permutedims(get_basis(y),(1,3,2))
  model = TrainedNeuralNetwork(get_strategy(red),r,ϕ)
  return NNOperator(model,trial,test)
end

function RBSteady.HRProjection(
  red::HighDimNNHyperReduction,
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
  red::HighDimNNHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial,get_time_combination(red))
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

const TupOfNNContribution = Tuple{Vararg{NNContribution}}

function FESpaces.interpolate!(
  b̂::AbstractArray,
  coeff::Tuple,
  a::TupOfNNContribution,
  r::AbstractRealisation
  )

  fill!(b̂,zero(eltype(b̂)))
  for (ai,ci) in zip(a,coeff)
    for aval in get_contributions(ai)
      interpolate!(b̂,ci,aval,r)
    end
  end
  return b̂
end

function FESpaces.interpolate!(
  cache::HRParamArray,
  a::TupOfNNContribution,
  r::AbstractRealisation
  )

  interpolate!(cache.hypred,cache.coeff,a,r)
end

# local interface

struct LocalNNHRProjection{B<:AbstractNNHyperReduction} <: HRProjection{Projection,B}
  reductions::AbstractMatrix
  k::NTuple{2,KmeansResult}
end

local_vals(a::LocalNNHRProjection) = a.reductions
get_clusters(a::LocalNNHRProjection) = a.k

RBSteady.get_basis(a::LocalNNHRProjection) = LocalProjection(map(get_basis,local_vals(a)),get_clusters(a))
RBSteady.get_interpolation(a::LocalNNHRProjection) = EmptyInterpolation()
RBSteady.projection_eltype(a::LocalNNHRProjection) = projection_eltype(first(local_vals(a)))
RBSteady.num_reduced_dofs(a::LocalNNHRProjection) = 1
RBSteady.num_reduced_dofs_left_projector(a::LocalNNHRProjection) = num_reduced_dofs_left_projector(first(local_vals(a)))
RBSteady.num_reduced_dofs_right_projector(a::LocalNNHRProjection) = num_reduced_dofs_right_projector(first(local_vals(a)))

function get_local(a::LocalNNHRProjection,μ::AbstractVector)
  k1,k2 = get_clusters(a)
  lab1 = get_label(k1,μ)
  lab2 = get_label(k2,μ)
  local_vals(a)[lab1,lab2]
end

function RBSteady.allocate_coefficient(a::LocalNNHRProjection,r::AbstractRealisation)
  allocate_coefficient(first(local_vals(a)),r)
end

function RBSteady.allocate_hyper_reduction(a::LocalNNHRProjection)
  allocate_hyper_reduction(first(local_vals(a)))
end

function RBSteady.reduced_form(
  lred::LocalReduction{<:Any,EuclideanNorm,<:AbstractNNHyperReduction},
  s,
  trian::Triangulation,
  test::SingleFieldRBSpace
  )

  red = get_reduction(lred)
  ks = compute_clusters(lred,s)
  kr, = get_clusters(test)
  cs = cluster(s,ks)
  B = typeof(red)

  hr = Matrix{HRProjection}(undef,length(ks.counts),length(kr.counts))
  for i in eachindex(ks.counts)
    si = cs[i]
    for (j,centerj) in enumerate(eachcol(kr.centers))
      testj = get_local(test,centerj)
      hyper_redij, = reduced_form(red,si,trian,testj)
      hr[i,j] = hyper_redij
    end
  end

  hyper_red = LocalNNHRProjection{B}(hr,(ks,kr))
  red_trian = reduced_triangulation(trian,hyper_red)
  return hyper_red,red_trian
end

function RBSteady.reduced_form(
  lred::LocalReduction{<:Any,EuclideanNorm,<:AbstractNNHyperReduction},
  s,
  trian::Triangulation,
  trial::SingleFieldRBSpace,
  test::SingleFieldRBSpace
  )

  red = get_reduction(lred)
  ks = compute_clusters(lred,s)
  kr, = get_clusters(test)
  cs = cluster(s,ks)
  B = typeof(red)

  hr = Matrix{HRProjection}(undef,length(ks.counts),length(kr.counts))
  for i in eachindex(ks.counts)
    si = cs[i]
    for (j,centerj) in enumerate(eachcol(kr.centers))
      trialj = get_local(trial,centerj)
      testj = get_local(test,centerj)
      hyper_redij, = reduced_form(red,si,trian,trialj,testj)
      hr[i,j] = hyper_redij
    end
  end

  hyper_red = LocalNNHRProjection{B}(hr,(ks,kr))
  red_trian = reduced_triangulation(trian,hyper_red)
  return hyper_red,red_trian
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