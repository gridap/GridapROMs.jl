function ParamFESpaces.UnEvalTrialFESpace(
  f::DistributedSingleFieldFESpace,
  dirichlet::Union{Function,AbstractVector{<:Function}}
  )

  spaces = map(local_views(f)) do space
    UnEvalTrialFESpace(space,dirichlet)
  end
  gids = get_free_dof_ids(f)
  trian = get_triangulation(f)
  vector_type = get_vector_type(f)
  DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
end

function ParamFESpaces.TrialParamFESpace(f::DistributedSingleFieldFESpace)
  spaces = map(f.spaces) do s
    TrialParamFESpace(s)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function ParamFESpaces.TrialParamFESpace(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialParamFESpace(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function ParamFESpaces.TrialParamFESpace(fun,f::DistributedSingleFieldFESpace)
  spaces = map(f.spaces) do s
    TrialParamFESpace(fun,s)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function ParamFESpaces.TrialParamFESpace!(f::DistributedSingleFieldFESpace,fun)
  spaces = map(f.spaces) do s
    TrialParamFESpace!(s,fun)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function ParamFESpaces.HomogeneousTrialParamFESpace(
  f::DistributedSingleFieldFESpace,
  args...
  )

  spaces = map(f.spaces) do s
    HomogeneousTrialParamFESpace(s,args...)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function ParamFESpaces.TrivialParamFESpace(f::DistributedSingleFieldFESpace,args...)
  spaces = map(f.spaces) do s
    TrivialParamFESpace(s,args...)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

const DistributedUnEvalTrialFESpace = DistributedSingleFieldFESpace{<:AbstractArray{<:UnEvalTrialFESpace}}

for f in (:(Arrays.evaluate),:(ODEs.allocate_space))
  @eval begin
    function $f(space::DistributedUnEvalTrialFESpace,r::AbstractRealisation)
      spaces = map(local_views(space)) do space
        $f(space,r)
      end
      gids = get_free_dof_ids(space)
      trian = get_triangulation(space)
      vector_type = get_vector_type(space)
      DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
    end

    function $f(space::DistributedMultiFieldFESpace,r::AbstractRealisation)
      if !ParamFESpaces.has_param(space)
        return space
      end
      field_fe_space = map(s->$f(s,r),space.field_fe_space)
      style = MultiFieldStyle(space)
      spaces = to_parray_of_arrays(map(local_views,field_fe_space))
      part_fe_spaces = map(s->MultiFieldFESpace(s;style),spaces)
      gids = get_free_dof_ids(space)
      _DistributedMultiFieldFESpace(field_fe_space,part_fe_spaces,gids)
    end
  end
end

function Arrays.evaluate!(
  spacex::DistributedFESpace,
  space::DistributedFESpace,
  x::AbstractRealisation
  )

  map(local_views(spacex),local_views(space)) do spacex,space
    Arrays.evaluate!(spacex,space,x)
  end
  return spacex
end

for T in (:AbstractRealisation,:Nothing)
  S = T==:Nothing ? :Nothing : :Any
  for f in (:(Arrays.evaluate),:(ODEs.allocate_space))
    @eval begin
      function $f(space::DistributedUnEvalTrialFESpace,x::$T,y::$S)
        spaces = map(local_views(space)) do space
          $f(space,x,y)
        end
        gids = get_free_dof_ids(space)
        trian = get_triangulation(space)
        vector_type = get_vector_type(space)
        DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
      end

      function $f(space::DistributedMultiFieldFESpace,x::$T,y::$S)
        if !ParamODEs.has_param_transient(space)
          return space
        end
        field_fe_space = map(s->$f(s,x,y),space.field_fe_space)
        style = MultiFieldStyle(space)
        spaces = to_parray_of_arrays(map(local_views,field_fe_space))
        part_fe_spaces = map(s->MultiFieldFESpace(s;style),spaces)
        gids = get_free_dof_ids(space)
        _DistributedMultiFieldFESpace(field_fe_space,part_fe_spaces,gids)
      end
    end
  end

  @eval begin
    function Arrays.evaluate!(
      spacex::DistributedFESpace,
      space::DistributedFESpace,
      x::$T,
      y::$S
      )

      map(local_views(spacex),local_views(space)) do spacex,space
        Arrays.evaluate!(spacex,space,x,y)
      end
      return spacex
    end
  end
end

function _DistributedMultiFieldFESpace(field_fe_space,part_fe_spaces,gids)
  if isa(gids,GridapDistributed.BlockPRange)
    fv = mortar(map(zero_free_values,field_fe_space))
    V = typeof(fv)
  else
    fv = map(zero_free_values,field_fe_space)
    V = promote_type(typeof.(fv)...)
  end
  DistributedMultiFieldFESpace(field_fe_space,part_fe_spaces,gids,V)
end

function ParamFESpaces.has_param(f::DistributedMultiFieldFESpace)
  ParamFESpaces.has_param(getany(local_views(f)))
end

function ParamODEs.has_param_transient(f::DistributedMultiFieldFESpace)
  ParamODEs.has_param_transient(getany(local_views(f)))
end

const DistributedSingleFieldParamFESpace = DistributedSingleFieldFESpace{<:AbstractArray{<:SingleFieldParamFESpace}}
const DistributedMultiFieldParamFESpace{MS,A,B,C,D<:AbstractParamPVector} = DistributedMultiFieldFESpace{MS,A,B,C,D}
const DistributedParamFESpace = Union{DistributedSingleFieldParamFESpace,DistributedMultiFieldParamFESpace}

function ParamDataStructures.param_length(f::DistributedParamFESpace)
  param_length(getany(local_views(f)))
end

function ParamFESpaces.get_vector_type2(f::DistributedSingleFieldParamFESpace)
  V = ParamFESpaces.get_vector_type2(getany(local_views(f)))
  typeof(PVector{V}(undef,partition(get_free_dof_ids(f))))
end

function ParamFESpaces.get_vector_type2(f::DistributedMultiFieldParamFESpace)
  ParamFESpaces.get_vector_type2(first(f.field_fe_space))
end

function FESpaces.zero_free_values(f::DistributedParamFESpace)
  param_zero_free_values(f)
end

function FESpaces.zero_dirichlet_values(f::DistributedParamFESpace)
  param_zero_dirichlet_values(f)
end

function FESpaces.zero_dirichlet_values(f::DistributedSingleFieldParamFESpace)
  map(zero_dirichlet_values,local_views(f))
end

function FESpaces.zero_free_values(f::DistributedMultiFieldParamFESpace{<:BlockMultiFieldStyle})
  mortar(map(zero_free_values,f.field_fe_space))
end

function FESpaces.zero_dirichlet_values(f::DistributedMultiFieldParamFESpace)
  map(zero_dirichlet_values,f.field_fe_space)
end

function GridapDistributed.DistributedMultiFieldFEFunction(
  field_fe_fun::AbstractVector{<:GridapDistributed.DistributedSingleFieldFEFunction},
  part_fe_fun::AbstractArray{<:MultiFieldParamFEFunction},
  free_values::AbstractVector
  )

  metadata = GridapDistributed.DistributedFEFunctionData(free_values)
  GridapDistributed.DistributedMultiFieldCellField(field_fe_fun,part_fe_fun,metadata)
end

function FESpaces.SparseMatrixAssembler(
  trial::DistributedParamFESpace,
  test::DistributedFESpace,
  par_strategy=SubAssembledRows()
  )

  PT = get_vector_type(getany(local_views(trial)))
  T = eltype2(PT)
  Tm = SparseMatrixCSC{T,Int}
  Tv = Vector{T}
  SparseMatrixAssembler(Tm,Tv,trial,test,par_strategy)
end

function ParamDataStructures.parameterise(
  a::GridapDistributed.DistributedSparseMatrixAssembler,
  plength::Int
  )

  assems = map(local_views(a)) do assem
    parameterise(assem,plength)
  end
  matrix_builder = parameterise(a.matrix_builder,plength)
  vector_builder = parameterise(a.vector_builder,plength)

  GridapDistributed.DistributedSparseMatrixAssembler(
    a.strategy,
    assems,
    matrix_builder,
    vector_builder,
    a.test_dofs_gids_prange,
    a.trial_dofs_gids_prange
  )
end

DofMaps.get_dof_eltype(a::GridapDistributed.DistributedCellDof) = get_dof_eltype(getany(local_views(a)))

function ParamODEs.collect_param_solutions(sol::ODEParamSolution{<:PVector})
  u0 = first(sol.us0)
  ncols = num_params(sol.r)*num_times(sol.r)
  sols = ParamODEs._allocate_solutions(u0,ncols)
  for (k,(rk,uk)) in enumerate(sol)
    ParamODEs._collect_solutions!(sols,uk,k)
  end
  return sols
end

function ParamODEs.collect_param_solutions(sol::ODEParamSolution{<:BlockPArray})
  u0 = first(sol.us0)
  ncols = num_params(sol.r)*num_times(sol.r)
  sols = ParamODEs._allocate_solutions(u0,ncols)
  for (k,(rk,uk)) in enumerate(sol)
    for i in 1:blocklength(u0)
      ParamODEs._collect_solutions!(blocks(sols)[i],blocks(uk)[i],k)
    end
  end
  return sols
end

function ParamODEs._allocate_solutions(u0::PVector,ncols)
  partition = map(local_views(u0)) do u0i
    ParamODEs._allocate_solutions(u0i,ncols)
  end
  PVector(partition,u0.index_partition)
end

function ParamODEs._allocate_solutions(u0::BlockPArray,ncols)
  mortar(map(b -> ParamODEs._allocate_solutions(b,ncols),blocks(u0)))
end

function ParamODEs._collect_solutions!(sols::PVector,ui::PVector,it::Int)
  map(local_views(sols),local_views(ui)) do sols,ui
    ParamODEs._collect_solutions!(sols,ui,it)
  end
end

Utils.get_polynomial_order(f::DistributedFESpace) = get_polynomial_order(getany(local_views(f)))