"""
    struct UncommonParamTrialFESpace{A<:SingleFieldFESpace} <: SingleFieldParamFESpace
      space::A
      dirichlet::AbstractVector
    end
"""
struct UncommonParamTrialFESpace{A<:SingleFieldFESpace} <: SingleFieldParamFESpace{A}
  space::A
  dirichlet::AbstractVector
end

# FE space interface

ParamDataStructures.param_length(f::UncommonParamTrialFESpace) = length(f.dirichlet)

FESpaces.get_fe_space(f::UncommonParamTrialFESpace) = f.space

# Evaluations

function ODEs.allocate_space(f::UncommonParamTrialFESpace,r::Realization)
  HomogeneousTrialParamFESpace(f.space,length(r))
end

function ODEs.allocate_space(f::UncommonParamTrialFESpace,μ::Realization,t)
  HomogeneousTrialParamFESpace(f.space,length(μ)*length(t))
end

function ODEs.allocate_space(f::UncommonParamTrialFESpace,r::TransientRealization)
  allocate_space(f,get_params(r),get_times(r))
end

function Arrays.evaluate(f::UncommonParamTrialFESpace,args...)
  fpt = allocate_space(f,args...)
  evaluate!(fpt,f,args...)
  fpt
end

function Arrays.evaluate!(fpt::TrialParamFESpace,f::UncommonParamTrialFESpace,r::Realization)
  @check param_length(fpt) ≤ length(r)
  TrialParamFESpace!(fpt,f.dirichlet[1:param_length(fpt)])
  fpt
end

function Arrays.evaluate!(
  fpt::TrialParamFESpace,
  f::UncommonParamTrialFESpace,
  r::TransientRealization)

  evaluate!(fpt,f,get_params(r),get_times(r))
end

function Arrays.evaluate!(
  fpt::TrialParamFESpace,
  f::UncommonParamTrialFESpace,
  r::Realization,t)

  @check param_length(fpt) ≤ length(μ)*length(t)
  TrialParamFESpace!(fpt,f.dirichlet[1:param_length(fpt)])
  fpt
end

(U::UncommonParamTrialFESpace)(r) = evaluate(U,r)
(U::UncommonParamTrialFESpace)(μ,t) = evaluate(U,μ,t)

Arrays.evaluate(U::UncommonParamTrialFESpace,r::Nothing) = evaluate(U.space,r)
Arrays.evaluate(U::UncommonParamTrialFESpace,μ::Nothing,t::Nothing) = evaluate(U.space,μ,t)
