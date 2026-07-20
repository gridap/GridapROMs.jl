"""
    struct NNInterpolation{M<:AbstractNNModel} <: Interpolation

An [`Interpolation`](@ref) backed by a neural network model. During the
online phase, `interpolate!` replaces the EIM linear solve with a NN forward
pass: `coeff[:, i] = model(μ_i)`.

Constructed automatically by `Interpolation(red::NNHyperReduction, basis, s)`.
"""
struct NNInterpolation{M<:AbstractNNModel} <: Interpolation
  model::M
end

function FESpaces.interpolate!(
  cache::AbstractArray,
  a::NNInterpolation,
  r::AbstractRealisation
  )

  x = matrix_of_params(r)
  out = a.model(x) # (n_modes × k)
  n = prod(innersize(cache))
  k = param_length(cache)
  copyto!(reshape(get_all_data(cache),n,k),out)
  cache
end

"""
    Interpolation(red::NNHyperReduction, basis::Projection, s::Snapshots)
      -> NNInterpolation

Offline training step for [`NNHyperReduction`](@ref):

1. Runs empirical interpolation on `basis` to determine integration points
   and the EIM matrix
2. Solves the EIM system for each snapshot to obtain coefficient vectors
3. Trains the NN from `red.factory` on the `(parameter, coefficient)` pairs

Returns an [`NNInterpolation`](@ref) ready for online use.
"""
function Interpolation(
  red::NNHyperReduction,
  a::Projection,
  s::Snapshots
  )

  inds,interp = empirical_interpolation(a)
  factor = lu(interp)
  r = get_realisation(s)
  red_data = get_at_domain(s,inds)
  coeff = parameterise(allocate_in_domain(a),r)
  ldiv!(coeff,factor,red_data)
  model = nn_factory(red)(r,coeff)
  NNInterpolation(model)
end
