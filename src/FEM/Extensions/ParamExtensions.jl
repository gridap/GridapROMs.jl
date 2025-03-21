struct ParamExtension{E<:ExtensionStyle} <: ExtensionStyle
  style::E
end

ParamExtension() = ParamExtension(HarmonicExtension())

struct UnEvalExtension{E} <: Extension{ParamExtension{E}}
  style::ParamExtension{E}
  space::SingleFieldFESpace
  dof_to_bg_dofs::AbstractVector
  metadata
end

function Extension(
  style::ParamExtension,
  space::SingleFieldFESpace,
  dof_to_bg_dofs::AbstractVector,
  args...)

  metadata = args
  UnEvalExtension(style,space,dof_to_bg_dofs,metadata)
end

function Arrays.evaluate(ext::UnEvalExtension{ZeroExtension},args...)
  space = ext.space(args...)
  GenericExtension(ZeroExtension(),space,ext.dof_to_bg_dofs)
end

function Arrays.evaluate(ext::UnEvalExtension{FunctionExtension},args...)
  f, = ext.metadata
  space = ext.space(args...)
  fμ = f(args...)
  GenericExtension(FunctionExtension(),space,ext.dof_to_bg_dofs,fμ)
end

function Arrays.evaluate(ext::UnEvalExtension{HarmonicExtension},args...)
  a,l, = ext.metadata
  spaceμ = parameterize(ext.space,args...)
  aμ = a
  lμ(v) = l(args...,v)

  laplacian = assemble_matrix(aμ,spaceμ,spaceμ)
  vector = assemble_vector(lμ,spaceμ)
  factor = lu(testitem(laplacian))
  values = similar(vector)
  for i in param_eachindex(laplacian)
    x = param_getindex(values,i)
    b = param_getindex(vector,i)
    ldiv!(x,factor,b)
    if i < param_length(laplacian)
      A = param_getindex(laplacian,i+1)
      lu!(factor,A)
    end
  end

  vh = FEFunction(spaceμ,values)
  GenericExtension(HarmonicExtension(),laplacian,vector,vh,ext.dof_to_bg_dofs)
end

(ext::UnEvalExtension)(::Nothing) = ext
(ext::UnEvalExtension)(μ) = evaluate(ext,μ)
(ext::UnEvalExtension)(μ,t) = evaluate(ext,μ,t)

get_out_dof_to_bg_dofs(ext::UnEvalExtension) = ext.dof_to_bg_dofs
FESpaces.get_fe_space(f::UnEvalExtension) = f.space
