"""
    struct NNOpRegression <: TrivialHyperReduction
      nparams::Int
      strategy::NNStrategy
    end

A hyper-reduction strategy for **operator regression**: the NN directly maps
parameter values to the Galerkin-projected residual vector, bypassing FE
assembly entirely during the online phase. Only suitable for residual
(vector-valued) operators.

The offline phase projects the residual snapshots onto the test space and
trains the NN to reproduce the projected vectors. The online phase calls
the NN forward pass, producing the projected residual without any assembly.

`strategy` controls the MLP architecture and training.
`nparams` controls how many parameter samples to use for NN training.
"""
struct NNOpRegression <: TrivialHyperReduction
  nparams::Int
  strategy::NNStrategy
end

"""
    NNOpRegression(; nparams=20, strategy=NNStrategy()) -> NNOpRegression
"""
function NNOpRegression(;nparams::Int=20,strategy=NNStrategy())
  NNOpRegression(nparams,strategy)
end

ParamDataStructures.num_params(r::NNOpRegression) = r.nparams
get_strategy(r::NNOpRegression) = r.strategy

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
    struct NNOpProjection{A<:ReducedProjection,M} <: HRProjection{A,NNOpRegression}

[`HRProjection`](@ref) for operator regression. The model `m` maps parameter
matrices (d×k) directly to projected-operator vectors (n_reduced×k), replacing
both the coefficient interpolation and the basis projection steps.

`basis` is a dummy [`ReducedProjection`](@ref) kept only to carry
dimensional information (number of reduced dofs). The `mul!` step in the
default [`interpolate!`](@ref) is bypassed entirely.
"""
struct NNOpProjection{A<:ReducedProjection,M} <: HRProjection{A,NNOpRegression}
  basis::A
  model::M
end

RBSteady.get_basis(a::NNOpProjection) = a.basis
RBSteady.get_interpolation(a::NNOpProjection) = EmptyInterpolation()
RBSteady.projection_eltype(a::NNOpProjection) = projection_eltype(get_basis(a))

"""
    const NNOpContribution = AffineContribution{<:NNOpProjection}
"""
const NNOpContribution = AffineContribution{<:NNOpProjection}

# NNOpProjection bypasses the default two-step (coeff interpolate + mat-vec).
# The NN output is added directly into b̂, matching the accumulate pattern.
function FESpaces.interpolate!(
  b̂::AbstractArray,
  _coeff::AbstractArray,
  a::NNOpProjection,
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

# HRProjection constructors for NNOpRegression

function RBSteady.HRProjection(
  red::NNOpRegression,
  s::Snapshots,
  trian::Triangulation,
  test::RBSpace
  )

  T = get_dof_value_type(test)
  n_reduced = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,n_reduced,1))

  r = get_realisation(s)
  snap_data = get_all_data(get_param_data(s))  # (n_dofs × k)
  test_basis = get_basis(test)                  # (n_dofs × n_reduced)
  y_data = test_basis' * snap_data              # (n_reduced × k)
  y_train = ConsecutiveParamArray(y_data)

  model = train_model(get_strategy(red),r,y_train)
  NNOpProjection(basis,model)
end

function RBSteady.HRProjection(
  red::NNOpRegression,
  s::Nothing,
  trian::Triangulation,
  test::RBSpace
  )

  T = get_dof_value_type(test)
  n_reduced = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,n_reduced,1))
  NNOpProjection(basis,_untrained_nn_model)
end

_untrained_nn_model(_) = error("NNOpProjection used without training (s=nothing was passed)")
