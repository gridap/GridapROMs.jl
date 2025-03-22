struct ParamExtension{E<:ExtensionStyle} <: ExtensionStyle
  style::E
end

ParamExtension() = ParamExtension(HarmonicExtension())

function Extension(
  style::ParamExtension,
  space::SingleFieldFESpace,
  fdof_to_bg_fdofs::AbstractVector,
  ddof_to_bg_ddofs::AbstractVector,
  args...)

  metadata = space,args...
  matdata = nothing
  vecdata = nothing
  values = zero(space(nothing))
  Extension(style,matdata,vecdata,values,fdof_to_bg_fdofs,ddof_to_bg_ddofs,metadata)
end

function Arrays.evaluate!(cache,ext::Extension{ParamExtension{ZeroExtension}},args...)
  space, = ext.matdata
  spaceμ = space(args...)
  zh = zero(spaceμ)

  ext.matdata = _mass_data(spaceμ)
  ext.vecdata = _interp_data(spaceμ,zh)
  ext.values = zh

  ext
end

function Arrays.evaluate!(cache,ext::Extension{ParamExtension{FunctionExtension}},args...)
  space,f = ext.metadata
  spaceμ = space(args...)
  fμ = f(args...)

  ext.matdata = _mass_data(spaceμ)
  ext.vecdata = _interp_data(spaceμ,fh)
  ext.values = zh

  ext
end

function Arrays.evaluate!(cache,ext::Extension{ParamExtension{HarmonicExtension}},args...)
  space,a,l = ext.metadata
  spaceμ = parameterize(space,args...)
  aμ = a
  lμ(v) = l(args...,v)

  matdata = _get_matdata(spaceμ,aμ)
  vecdata = _get_vecdata(spaceμ,lμ)
  assem = SparseMatrixAssembler(spaceμ,spaceμ)
  A = assemble_matrix(assem,matdata)
  b = assemble_vector(assem,vecdata)
  u = zero_free_values(spaceμ)
  solve!(u,LUSolver(),A,b)

  ext.matdata = matdata
  ext.vecdata = vecdata
  ext.values = FEFunction(spaceμ,u)

  ext
end

(ext::Extension)(::Nothing) = ext
(ext::Extension)(μ) = evaluate(ext,μ)
(ext::Extension)(r::TransientRealization) = evaluate(ext,get_params(r),get_times(r))
(ext::Extension)(μ,t) = evaluate(ext,μ,t)
