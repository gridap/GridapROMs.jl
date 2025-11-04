"""
    const TransientTrialParamFESpace = UnEvalTrialFESpace
"""
const TransientTrialParamFESpace = UnEvalTrialFESpace

function ODEs.allocate_space(U::UnEvalTrialFESpace,μ::Realization,t)
  HomogeneousTrialParamFESpace(U.space,length(μ)*length(t))
end

function ODEs.allocate_space(U::UnEvalTrialFESpace,r::TransientRealization)
  allocate_space(U,get_params(r),get_times(r))
end

function Arrays.evaluate!(
  Upt::TrialParamFESpace,
  U::UnEvalTrialFESpace,
  μ::Realization,t)

  dir(f) = f(μ,t)
  dir(f::Vector) = dir.(f)
  TrialParamFESpace!(Upt,dir(U.dirichlet))
  Upt
end

function Arrays.evaluate!(
  Upt::TrialParamFESpace,
  U::UnEvalTrialFESpace,
  r::TransientRealization)

  evaluate!(Upt,U,get_params(r),get_times(r))
end

Arrays.evaluate(U::UnEvalTrialFESpace,μ::Nothing,t::Nothing) = U.space0

(U::UnEvalTrialFESpace)(μ,t) = evaluate(U,μ,t)
(U::TrialFESpace)(μ,t) = U
(U::ZeroMeanFESpace)(μ,t) = U

function ODEs.time_derivative(U::UnEvalTrialFESpace)
  ∂tdir(f) = (μ,t) -> time_derivative(f(μ,t))
  ∂tdir(f::Vector) = ∂tdir.(f)
  UnEvalTrialFESpace(U.space,∂tdir(U.dirichlet))
end

# Define the UnEvalTrialFESpace interface for stationary spaces

ODEs.allocate_space(U::FESpace,μ,t) = U
Arrays.evaluate!(Upt::FESpace,U::FESpace,μ,t) = U
Arrays.evaluate(U::FESpace,μ,t) = U
(space::FESpace)(μ,t) = evaluate(space,μ,t)

# Define the interface for MultiField

"""
    const TransientMultiFieldParamFESpace = MultiFieldFESpace
"""
const TransientMultiFieldParamFESpace = MultiFieldFESpace

function has_unevaluated(U::MultiFieldFESpace)
  (
    any(space -> space isa TransientTrialFESpace,U.spaces) ||
    any(space -> space isa UnEvalTrialFESpace,U.spaces)
  )
end

function has_param_transient(U::MultiFieldFESpace)
  any(space -> space isa UnEvalTrialFESpace,U.spaces)
end

function ODEs.allocate_space(U::MultiFieldFESpace,μ,t)
  if !has_param_transient(U)
    return U
  end
  spaces = map(U->allocate_space(U,μ,t),U.spaces)
  style = MultiFieldStyle(U)
  MultiFieldParamFESpace(spaces;style)
end

function ODEs.allocate_space(U::MultiFieldFESpace,r::TransientRealization)
  allocate_space(U,get_params(r),get_times(r))
end

function Arrays.evaluate!(
  Upt::MultiFieldFESpace,
  U::MultiFieldFESpace,
  μ,t)

  if !has_param_transient(U)
    return U
  end
  for (Upti,Ui) in zip(Upt,U)
    evaluate!(Upti,Ui,μ,t)
  end
  Upt
end

function ODEs.evaluate!(Upt::MultiFieldFESpace,U::MultiFieldFESpace,r::TransientRealization)
  evaluate!(Upt,U,get_params(r),get_times(r))
end

function Arrays.evaluate(U::MultiFieldFESpace,μ::Nothing,t::Nothing)
  if !has_param_transient(U)
    return U
  end
  spaces = map(space -> evaluate(space,μ,t),U.spaces)
  style = MultiFieldStyle(U)
  MultiFieldFESpace(spaces;style)
end

function Arrays.evaluate(U::MultiFieldFESpace,μ,t)
  if !has_param_transient(U)
    return U
  end
  Upt = allocate_space(U,μ,t)
  evaluate!(Upt,U,μ,t)
  Upt
end

function Arrays.evaluate(U::MultiFieldFESpace,r::TransientRealization)
  evaluate(U,get_params(r),get_times(r))
end
