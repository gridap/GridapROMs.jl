# Reconstruct Gridap FEFunction from neural operator predictions.
#
# This closes the loop: the neural operator outputs a raw DOF vector,
# and we wrap it back into a Gridap FEFunction so it can be used for
# visualization (writevtk), error computation, and postprocessing
# with the full Gridap ecosystem.

"""
    evaluate_rom(result::TrainingResult, μ::AbstractVector) -> Vector{Float64}

Predict the DOF vector for a new parameter μ using the trained neural operator.
Returns denormalized DOF values ready for FEFunction reconstruction.
"""
function evaluate_rom(result::TrainingResult,μ::AbstractVector)
  mu_col = Float32.(reshape(μ,length(μ),1))
  mu_n = normalize(result.input_norm,mu_col)

  u_hat_n,_ = Lux.apply(
    result.model,(mu_n,result.trunk_matrix),result.params,result.state
  )
  u_hat = denormalize(result.output_norm,vec(u_hat_n))
  return Float64.(u_hat)
end

"""
    reconstruct_fe_function(
      result::TrainingResult,
      μ::AbstractVector,
      trial::FESpace
    ) -> FEFunction

Evaluate the neural operator at parameter μ and reconstruct a Gridap
FEFunction. The trial space provides the Dirichlet DOF values and the
mesh/basis function information needed to build a proper FEFunction.

This is the key integration point with Gridap: predicted DOFs go in via
`FEFunction(trial, free_values)`, the same constructor Gridap uses
internally after solving a linear system.
"""
function reconstruct_fe_function(
  result::TrainingResult,
  μ::AbstractVector,
  trial::FESpace
)
  predicted_dofs = evaluate_rom(result,μ)
  return FEFunction(trial,predicted_dofs)
end
