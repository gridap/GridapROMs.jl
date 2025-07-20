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

function Arrays.evaluate(f::UncommonParamTrialFESpace,μ::Realization)
  diri = ParamArray(f.dirichlet[1:param_length(μ)])
  TrialParamFESpace(diri,f.space)
end

function Arrays.evaluate(f::UncommonParamTrialFESpace,args...)
  fp = allocate_space(f,args...)
  evaluate!(fp,f,args...)
  fp
end

function ODEs.allocate_space(f::UncommonParamTrialFESpace,μ::Realization,t)
  HomogeneousTrialParamFESpace(f.space,length(μ)*length(t))
end

function ODEs.allocate_space(f::UncommonParamTrialFESpace,r::TransientRealization)
  allocate_space(f,get_params(r),get_times(r))
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
  @check isa(f.dirichlet,AbstractVector{<:Function})
  dirif = TransientParamFunction(first(f.dirichlet),r,t)
  TrialParamFESpace!(fpt,dirif)
  fpt
end

(U::UncommonParamTrialFESpace)(r) = evaluate(U,r)
(U::UncommonParamTrialFESpace)(μ,t) = evaluate(U,μ,t)

Arrays.evaluate(U::UncommonParamTrialFESpace,r::Nothing) = evaluate(U.space,r)
Arrays.evaluate(U::UncommonParamTrialFESpace,μ::Nothing,t::Nothing) = evaluate(U.space,μ,t)
