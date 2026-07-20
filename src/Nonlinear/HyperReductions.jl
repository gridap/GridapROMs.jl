"""
    const NNHRProjection{A} = HRProjection{A,<:NNHyperReduction}

[`HRProjection`](@ref) using a neural network for coefficient interpolation.
The linear reduced basis is obtained from POD/TTSVD snapshots; only the EIM
coefficient prediction step is replaced by a NN.
"""
const NNHRProjection{A} = HRProjection{A,<:NNHyperReduction}

"""
    const NNContribution = AffineContribution{<:NNHRProjection}
"""
const NNContribution = AffineContribution{<:NNHRProjection}

"""
    struct NNProjection{A<:ReducedProjection,M} <: HRProjection{A,NNRegression}

[`HRProjection`](@ref) for operator regression. The model `m` maps parameter
matrices (d×k) directly to projected-operator vectors (n_reduced×k), replacing
both the coefficient interpolation and the basis projection steps.

`basis` is a dummy [`ReducedProjection`](@ref) kept only to carry
dimensional information (number of reduced dofs). The `mul!` step in the
default [`interpolate!`](@ref) is bypassed entirely.
"""
struct NNProjection{A<:ReducedProjection,M} <: HRProjection{A,NNRegression}
  basis::A
  model::M
end

RBSteady.get_basis(a::NNProjection) = a.basis
RBSteady.get_interpolation(a::NNProjection) = EmptyInterpolation()
RBSteady.projection_eltype(a::NNProjection) = projection_eltype(get_basis(a))

"""
    const NNOpContribution = AffineContribution{<:NNProjection}
"""
const NNOpContribution = AffineContribution{<:NNProjection}

# NNProjection bypasses the default two-step (coeff interpolate + mat-vec).
# The NN output is added directly into b̂, matching the accumulate pattern.
function FESpaces.interpolate!(
  b̂::AbstractArray,
  _coeff::AbstractArray,
  a::NNProjection,
  r::AbstractRealisation
  )

  x = matrix_of_params(r)
  out = a.model(x) # (n_reduced × k)
  _axpy_nn_output!(b̂,out)
  return b̂
end

function _axpy_nn_output!(b̂::ConsecutiveParamVector,out::AbstractMatrix)
  axpy!(one(eltype2(b̂)),out,get_all_data(b̂))
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

function RBSteady.HRProjection(
  red::NNHyperReduction,
  s::Nothing,
  trian::Triangulation,
  test::RBSpace
  )

  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))
  interp = Interpolation(red)
  return HRProjection(basis,red,interp)
end

function RBSteady.HRProjection(
  red::NNHyperReduction,
  s::Nothing,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  T = get_dof_value_type(trial)
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  basis = ReducedProjection(zeros(T,nrows,1,ncols))
  interp = Interpolation(red)
  return HRProjection(basis,red,interp)
end

# HRProjection constructors for NNRegression

function RBSteady.HRProjection(
  red::NNRegression,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))

  r = get_realisation(s)
  b = GalerkinProjectable(s)
  y = galerkin_projection(test,b)
  model = train_model(get_strategy(red),r,y)

  return NNProjection(basis,model)
end

function RBSteady.HRProjection(
  red::NNRegression,
  s::Nothing,
  trian::Triangulation,
  test::RBSpace
  )

  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))

  model = 
  NNProjection(basis,_untrained_nn_model)
end

_untrained_nn_model(_) = error("NNProjection used without training (s=nothing was passed)")
