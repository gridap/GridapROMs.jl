struct ExtensionFESpace{S<:SingleFieldFESpace,E<:Extension} <: SingleFieldFESpace
  space::S
  extension::E
  bg_space::SingleFieldFESpace
  fdof_to_bg_fdofs::AbstractVector
  ddof_to_bg_ddofs::AbstractVector
end

function ExtensionFESpace(
  space::SingleFieldFESpace,
  extension::Extension,
  bg_space::SingleFieldFESpace)

  fdof_to_bg_fdofs,ddof_to_bg_ddofs = get_dof_to_bg_dof(bg_space,space)
  ExtensionFESpace(space,extension,bg_space,fdof_to_bg_fdofs,ddof_to_bg_ddofs)
end

function ZeroExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace
  )

  ext = Extension(ZeroExtension(),bg_space,ext_space)
  ExtensionFESpace(int_space,ext,bg_space)
end

function FunctionExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  g::Function)

  ext = Extension(FunctionExtension(),bg_space,ext_space,g)
  ExtensionFESpace(int_space,ext,bg_space)
end

function HarmonicExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  a::Function,
  l::Function)

  ext = Extension(HarmonicExtension(),bg_space,ext_space,a,l)
  ExtensionFESpace(int_space,ext,bg_space)
end

FESpaces.ConstraintStyle(::Type{<:ExtensionFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::ExtensionFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::ExtensionFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::ExtensionFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::ExtensionFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::ExtensionFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::ExtensionFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::ExtensionFESpace) = get_fe_dof_basis(f.space)

FESpaces.get_cell_isconstrained(f::ExtensionFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::ExtensionFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::ExtensionFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::ExtensionFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::ExtensionFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::ExtensionFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::ExtensionFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_vector_type(f::ExtensionFESpace) = get_vector_type(f.space)

function FESpaces.scatter_free_and_dirichlet_values(f::ExtensionFESpace,fv,dv)
  scatter_free_and_dirichlet_values(f.space,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(f::ExtensionFESpace,cv)
  FESpaces.gather_free_and_dirichlet_values(f.space,cv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::ExtensionFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,f.space,cv)
end

# param interface

function ParamZeroExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace)

  ext = Extension(ParamExtension(ZeroExtension()),bg_space,ext_space,a,l)
  ExtensionFESpace(int_space,ext,bg_space)
end

function ParamFunctionExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  f::Function)

  ext = Extension(ParamExtension(FunctionExtension()),bg_space,ext_space,f)
  ExtensionFESpace(int_space,ext,bg_space)
end

function ParamHarmonicExtensionFESpace(
  bg_space::SingleFieldFESpace,
  int_space::SingleFieldFESpace,
  ext_space::SingleFieldFESpace,
  a::Function,
  l::Function)

  ext = Extension(ParamExtension(HarmonicExtension()),bg_space,ext_space,a,l)
  ExtensionFESpace(int_space,ext,bg_space)
end

function Arrays.evaluate(f::ExtensionFESpace,args...)
  extension = f.extension(args...)
  ExtensionFESpace(f.space,extension,f.bg_space,f.fdof_to_bg_fdofs,f.ddof_to_bg_ddofs)
end

(space::ExtensionFESpace)(t) = evaluate(space,t)
(space::ExtensionFESpace)(μ,t) = evaluate(space,t)

function ODEs.allocate_space(U::UnEvalTrialFESpace{<:ExtensionFESpace},μ::Realization)
  HomogeneousTrialParamFESpace(U.space(μ),length(μ))
end

function ODEs.allocate_space(U::UnEvalTrialFESpace{<:ExtensionFESpace},μ::Realization,t)
  HomogeneousTrialParamFESpace(U.space(μ,t),length(μ)*length(t))
end

# extended interface

get_ext_space(f::SingleFieldFESpace) = @abstractmethod
get_bg_fe_space(f::SingleFieldFESpace) = @abstractmethod
get_extension(f::SingleFieldFESpace) = @notimplemented

get_ext_space(f::ExtensionFESpace) = f
get_bg_fe_space(f::ExtensionFESpace) = f.bg_space
get_extension(f::ExtensionFESpace) = f.extension

for F in (:get_ext_space,:get_bg_fe_space,:get_extension)
  for T in (:SingleFieldParamFESpace,:UnEvalTrialFESpace,:TransientTrialFESpace,:TrialFESpace)
    if !(F==:get_bg_fe_space && T==:SingleFieldParamFESpace)
      @eval begin
        $F(f::$T{<:ExtensionFESpace}) = $F(f.space)
      end
    end
  end
end

function get_bg_fe_space(f::TrivialParamFESpace{<:ExtensionFESpace})
  TrivialParamFESpace(get_bg_fe_space(f.space),f.plength)
end

function get_bg_fe_space(f::TrialParamFESpace{<:ExtensionFESpace})
  TrialParamFESpace(get_bg_fe_space(f.space),f.dirichlet_values)
end

zero_bg_free_values(f::SingleFieldFESpace) = zero_free_values(get_bg_fe_space(f))
zero_bg_dirichlet_values(f::SingleFieldFESpace) = zero_dirichlet_values(get_bg_fe_space(f))

zero_bg_free_values(f::ExtensionFESpace) = zero_free_values(f.bg_space)
zero_bg_dirichlet_values(f::ExtensionFESpace) = zero_dirichlet_values(f.bg_space)

function zero_bg_free_values(f::SingleFieldParamFESpace{<:ExtensionFESpace})
  global_parameterize(zero_bg_free_values(get_ext_space(f)),param_length(f))
end
function zero_bg_dirichlet_values(f::SingleFieldParamFESpace{<:ExtensionFESpace})
  global_parameterize(zero_bg_dirichlet_values(get_ext_space(f)),param_length(f))
end

function ExtendedFEFunction(f::SingleFieldFESpace,fv::AbstractVector,dv::AbstractVector)
  cell_vals = scatter_extended_free_and_dirichlet_values(f,fv,dv)
  cell_field = ExtendedCellField(f,cell_vals)
  SingleFieldFEFunction(cell_field,cell_vals,fv,dv,f)
end

function ExtendedFEFunction(f::SingleFieldFESpace,fv::AbstractVector)
  dv = get_dirichlet_dof_values(f)
  ExtendedFEFunction(f,fv,dv)
end

function ExtendedFEFunction(
  f::ExtensionFESpace{<:ZeroMeanFESpace},fv::AbstractVector,dv::AbstractVector
  )

  c = FESpaces._compute_new_fixedval(fv,dv,f.vol_i,f.vol,f.space.dof_to_fix)
  zmfv = lazy_map(+,fv,Fill(c,length(fv)))
  zmdv = dv .+ c
  FEFunction(f.space,zmfv,zmdv)
end

function _extended_cell_values(f::SingleFieldFESpace,object)
  FESpaces._cell_vals(get_bg_fe_space(f),object)
end

function _extended_cell_values(f::SingleFieldFESpace,object::SingleFieldFEFunction)
  bg_f = get_bg_fe_space(f)
  cell_vals = get_cell_dof_values(object)
  is_bg = _is_background_cell_vals(f,cell_vals)

  if is_bg
    bg_object = object
  else
    bg_cell_vals = extend_incut_cell_vals(f,cell_vals)
    bg_cell_field = CellField(bg_f,bg_cell_vals)
    bg_fv,bg_dv = gather_extended_free_and_dirichlet_values(f,bg_cell_vals)
    bg_object = SingleFieldFEFunction(bg_cell_field,bg_cell_vals,bg_fv,bg_dv,bg_f)
  end

  FESpaces._cell_vals(bg_f,bg_object)
end

function _is_background_cell_vals(f::SingleFieldFESpace,cell_vals)
  bg_f = get_bg_fe_space(f)
  bg_trian = get_triangulation(bg_f)
  num_bg_cells = num_cells(bg_trian)
  if length(cell_vals) < num_bg_cells
    return false
  else
    @assert length(cell_vals) == num_bg_cells
    return true
  end
end

function extended_interpolate(object,f::SingleFieldFESpace)
  fv = zero_bg_free_values(f)
  extended_interpolate!(object,fv,f)
end

function extended_interpolate!(object,fv,f::SingleFieldFESpace)
  bg_cell_vals = _extended_cell_values(f,object)
  gather_extended_free_values!(fv,f,bg_cell_vals)
  ExtendedFEFunction(f,fv)
end

function extended_interpolate_everywhere(object,f::SingleFieldFESpace)
  fv = zero_bg_free_values(f)
  dv = zero_bg_dirichlet_values(f)
  extended_interpolate_everywhere!(object,fv,dv,f)
end

function extended_interpolate_everywhere!(object,fv,dv,f::SingleFieldFESpace)
  bg_cell_vals = _extended_cell_values(f,object)
  gather_extended_free_and_dirichlet_values!(fv,dv,f,bg_cell_vals)
  ExtendedFEFunction(f,fv,dv)
end

function extended_interpolate_dirichlet(object,f::SingleFieldFESpace)
  fv = zero_bg_free_values(f)
  dv = zero_bg_dirichlet_values(f)
  extended_interpolate_dirichlet!(object,fv,dv,f)
end

function extended_interpolate_dirichlet!(object,fv,dv,f::SingleFieldFESpace)
  bg_cell_vals = _extended_cell_values(f,object)
  gather_extended_dirichlet_values!(dv,f,bg_cell_vals)
  fill!(fv,zero(eltype(fv)))
  ExtendedFEFunction(f,fv,dv)
end

function ExtendedCellField(f::SingleFieldFESpace,bg_cellvals)
  CellField(get_bg_fe_space(f),bg_cellvals)
end

function scatter_extended_free_and_dirichlet_values(f::SingleFieldFESpace,fv,dv)
  bg_f = get_bg_fe_space(f)
  if length(fv) == num_free_dofs(bg_f) && length(dv) == num_dirichlet_dofs(bg_f)
    scatter_free_and_dirichlet_values(bg_f,fv,dv)
  else
    incut_bg_cells = get_incut_cells_to_bg_cells(f)
    out_bg_cells = get_out_cells_to_bg_cells(f)
    bg_cells = lazy_append(incut_bg_cells,out_bg_cells)

    incut_cell_vals = scatter_free_and_dirichlet_values(f,fv,dv)
    bg_cell_vals = extend_incut_cell_vals(f,incut_cell_vals)
    lazy_map(Reindex(bg_cell_vals),sortperm(bg_cells))
  end
end

function gather_extended_dirichlet_values(f::SingleFieldFESpace,bg_cell_vals)
  bg_dv = zero_bg_dirichlet_values(f)
  gather_extended_dirichlet_values!(bg_dv,f,bg_cell_vals)
  dv
end

function gather_extended_dirichlet_values!(bg_dv,f::SingleFieldFESpace,bg_cell_vals)
  bg_fv = zero_bg_free_values(f)
  gather_extended_free_and_dirichlet_values!(bg_fv,bg_dv,f,bg_cell_vals)
  bg_dv
end

function gather_extended_free_values(f::SingleFieldFESpace,bg_cell_vals)
  bg_fv = zero_bg_free_values(f)
  gather_extended_free_values!(bg_fv,f,bg_cell_vals)
  bg_fv
end

function gather_extended_free_values!(bg_fv,f::SingleFieldFESpace,bg_cell_vals)
  bg_dv = zero_bg_dirichlet_values(f)
  gather_extended_free_and_dirichlet_values!(bg_fv,bg_dv,f,bg_cell_vals)
  bg_fv
end

function gather_extended_free_and_dirichlet_values(f::SingleFieldFESpace,bg_cell_vals)
  bg_fv = zero_bg_free_values(f)
  bg_dv = zero_bg_dirichlet_values(f)
  gather_extended_free_and_dirichlet_values!(bg_fv,bg_dv,f,bg_cell_vals)
end

function gather_extended_free_and_dirichlet_values!(bg_fv,bg_dv,f::SingleFieldFESpace,bg_cell_vals)
  bg_f = get_bg_fe_space(f)
  bg_vals_from_bg_cell_vals!(bg_fv,bg_dv,bg_f,bg_cell_vals)
end

function extend_free_values(f::SingleFieldFESpace,fv)
  dv = zero_dirichlet_values(f)
  fv,dv = extend_free_and_dirichlet_values(f,fv,dv)
  return fv
end

function extend_dirichlet_values(f::SingleFieldFESpace,dv)
  fv = zero_free_values(f)
  fv,dv = extend_free_and_dirichlet_values(f,fv,dv)
  return dv
end

function extend_free_and_dirichlet_values(f::SingleFieldFESpace,fv,dv)
  bg_fv = zero_bg_free_values(f)
  bg_dv = zero_bg_dirichlet_values(f)
  _bg_vals_from_in_vals!(bg_fv,bg_dv,get_ext_space(f),fv,dv)
  return bg_fv,bg_dv
end

# utils

get_incut_fe_space(f::ExtensionFESpace) = f.space
get_outcut_fe_space(f::ExtensionFESpace) = get_fe_space(f.extension)

function get_incut_cells_to_bg_cells(f::ExtensionFESpace)
  get_cell_to_bg_cell(get_incut_fe_space(f))
end

function get_outcut_cells_to_bg_cells(f::ExtensionFESpace)
  get_cell_to_bg_cell(get_outcut_fe_space(f))
end

function get_cut_cells_to_bg_cells(f::ExtensionFESpace)
  incut_bg_cells = get_incut_cells_to_bg_cells(f)
  outcut_bg_cells = get_outcut_cells_to_bg_cells(f)
  intersect(incut_bg_cells,outcut_bg_cells)
end

function get_in_cells_to_bg_cells(f::ExtensionFESpace)
  incut_bg_cells = get_incut_cells_to_bg_cells(f)
  cut_bg_cells = get_cut_cells_to_bg_cells(f)
  setdiff(incut_bg_cells,cut_bg_cells)
end

function get_out_cells_to_bg_cells(f::ExtensionFESpace)
  outcut_bg_cells = get_outcut_cells_to_bg_cells(f)
  cut_bg_cells = get_cut_cells_to_bg_cells(f)
  setdiff(outcut_bg_cells,cut_bg_cells)
end

function get_bg_cells_to_incut_cells(f::ExtensionFESpace)
  get_bg_cell_to_cell(get_incut_fe_space(f))
end

function get_in_cells_to_incut_cells(f::ExtensionFESpace)
  in_bg_cells = get_in_cells_to_bg_cells(f)
  bg_incut_cells = get_bg_cells_to_incut_cells(f)
  collect(lazy_map(Reindex(bg_incut_cells),in_bg_cells))
end

function get_bg_cells_to_outcut_cells(f::ExtensionFESpace)
  get_bg_cell_to_cell(get_outcut_fe_space(f))
end

function get_out_cells_to_outcut_cells(f::ExtensionFESpace)
  out_bg_cells = get_out_cells_to_bg_cells(f)
  bg_outcut_cells = get_bg_cells_to_outcut_cells(f)
  collect(lazy_map(Reindex(bg_outcut_cells),out_bg_cells))
end

function extend_incut_cell_vals(f::ExtensionFESpace,incut_cell_vals)
  out_out_cells = get_out_cells_to_outcut_cells(f)
  outcut_cell_vals = get_cell_dof_values(f.extension)
  out_cell_vals = lazy_map(Reindex(outcut_cell_vals),out_out_cells)
  bg_cell_vals = lazy_append(incut_cell_vals,out_cell_vals)
  return bg_cell_vals
end

get_in_fdof_to_bg_fdofs(f::ExtensionFESpace) = f.fdof_to_bg_fdofs
get_in_ddof_to_bg_ddofs(f::ExtensionFESpace) = f.ddof_to_bg_ddofs
get_out_fdof_to_bg_fdofs(f::ExtensionFESpace) = get_out_fdof_to_bg_fdofs(f.extension)
get_out_ddof_to_bg_ddofs(f::ExtensionFESpace) = get_out_ddof_to_bg_ddofs(f.extension)

for f in (:get_incut_fe_space,:get_outcut_fe_space,:get_incut_cells_to_bg_cells,
    :get_outcut_cells_to_bg_cells,:get_cut_cells_to_bg_cells,:get_in_cells_to_bg_cells,
    :get_out_cells_to_bg_cells,:get_bg_cells_to_incut_cells,:get_in_cells_to_incut_cells,
    :get_bg_cells_to_outcut_cells,:get_out_cells_to_outcut_cells,:extend_incut_cell_vals,
    :get_in_fdof_to_bg_fdofs,:get_in_ddof_to_bg_ddofs,:get_out_fdof_to_bg_fdofs,
    :get_out_ddof_to_bg_ddofs)
  @eval begin
    $f(fs::SingleFieldFESpace,args...) = $f(get_ext_space(fs),args...)
  end
end

function _bg_vals_from_in_vals!(bg_fv,bg_dv,f::ExtensionFESpace,in_fv,in_dv)
  out_fv = get_free_dof_values(f.extension.values)
  out_dv = get_dirichlet_dof_values(f.extension.values)
  for (in_fdof,bg_fdof) in enumerate(f.fdof_to_bg_fdofs)
    bg_fv[bg_fdof] = in_fv[in_fdof]
  end
  for (out_fdof,bg_fdof) in enumerate(f.extension.fdof_to_bg_fdofs)
    bg_fv[bg_fdof] = out_fv[out_fdof]
  end
  for (in_ddof,bg_ddof) in enumerate(f.ddof_to_bg_ddofs)
    bg_dv[bg_ddof] = in_dv[in_ddof]
  end
  for (out_ddof,bg_ddof) in enumerate(f.extension.ddof_to_bg_ddofs)
    bg_dv[bg_ddof] = out_dv[out_ddof]
  end
end

function _bg_vals_from_in_vals!(
  bg_fv::ConsecutiveParamVector,
  bg_dv::ConsecutiveParamVector,
  f::ExtensionFESpace,
  in_fv::ConsecutiveParamVector,
  in_dv::ConsecutiveParamVector)

  out_fv = get_free_dof_values(f.extension.values)
  out_dv = get_dirichlet_dof_values(f.extension.values)
  bg_fdata = get_all_data(bg_fv)
  in_fdata = get_all_data(in_fv)
  out_fdata = get_all_data(out_fv)
  bg_ddata = get_all_data(bg_dv)
  in_ddata = get_all_data(in_dv)
  out_ddata = get_all_data(out_dv)
  for k in param_eachindex(bg_fv)
    for (in_fdof,bg_fdof) in enumerate(f.fdof_to_bg_fdofs)
      bg_fdata[bg_fdof,k] = in_fdata[in_fdof,k]
    end
    for (out_fdof,bg_fdof) in enumerate(f.extension.fdof_to_bg_fdofs)
      bg_fdata[bg_fdof,k] = out_fdata[out_fdof,k]
    end
    for (in_ddof,bg_ddof) in enumerate(f.ddof_to_bg_ddofs)
      bg_ddata[bg_ddof,k] = in_ddata[in_fdof,k]
    end
    for (out_ddof,bg_ddof) in enumerate(f.extension.ddof_to_bg_ddofs)
      bg_ddata[bg_ddof,k] = out_ddata[out_ddof,k]
    end
  end
end

function bg_vals_from_bg_cell_vals!(bg_fv,bg_dv,f::ExtensionFESpace,bg_cv)
  incut_cellids = get_cell_dof_ids(f)
  out_cellids = get_cell_dof_ids(get_outcut_fe_space(f))

  incut_cv,out_cv = _split_bg_cell_vals(bg_cv)
  cache_incut_cv = array_cache(incut_cv)
  cache_out_cv = array_cache(out_cv)

  cache_incut_dofs = array_cache(incut_cellids)
  cache_out_dofs = array_cache(out_cellids)

  incut_bg_cells = 1:length(incut_cv)
  out_bg_cells = 1:length(out_cv)

  FESpaces._free_and_dirichlet_values_fill!(
    bg_fv,bg_dv,cache_incut_cv,cache_incut_dofs,incut_cv,incut_cellids,incut_bg_cells)
  FESpaces._free_and_dirichlet_values_fill!(
    bg_fv,bg_dv,cache_out_cv,cache_out_dofs,out_cv,out_cellids,out_bg_cells)
end

function _split_bg_cell_vals(
  f::ExtensionFESpace,
  bg_cv::LazyArray{<:Fill{<:Broadcasting{<:PosNegReindex}}}
  )

  incut_cell = get_incut_cells_to_bg_cells(f)
  out_cell = get_out_cells_to_bg_cells(f)
  (lazy_map(Reindex(cv),incut_cell),lazy_map(Reindex(cv),out_cell))
end

function _split_bg_cell_vals(
  f::ExtensionFESpace,
  bg_cv::LazyArray{<:Fill{<:Reindex{<:AppendedArray}}}
  )

  app_vals = bg_cv.maps[1].values
  (app_vals.a,app_vals.b)
end

function DofMaps.get_dof_map(f::ExtensionFESpace,args...)
  get_dof_map(get_bg_fe_space(f),args...)
end
