function ParamFESpaces.UnEvalTrialFESpace(
  space::DistributedSingleFieldFESpace,
  dirichlet::Union{Function,AbstractVector{<:Function}}
  )

  spaces = map(local_views(space)) do space
    UnEvalTrialFESpace(space,dirichlet)
  end
  gids  = get_free_dof_ids(space)
  trian = get_triangulation(space)
  vector_type = get_vector_type(space)
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

function ParamFESpaces.HomogeneousTrialParamFESpace(f::DistributedSingleFieldFESpace,args...)
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
    function $f(space::DistributedUnEvalTrialFESpace,x::AbstractRealization)
      spaces = map(local_views(space)) do space
        $f(space,x)
      end
      gids  = get_free_dof_ids(space)
      trian = get_triangulation(space)
      vector_type = get_vector_type(space)
      DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
    end

    function $f(space::DistributedMultiFieldFESpace,x::AbstractRealization)
      if !ParamFESpaces.has_param(space)
        return space
      end
      field_fe_space = map(s->$f(s,x),space.field_fe_space)
      style = MultiFieldStyle(space)
      spaces = to_parray_of_arrays(map(local_views,field_fe_space))
      part_fe_spaces = map(s->MultiFieldFESpace(s;style),spaces)
      gids = get_free_dof_ids(space)
      vector_type = get_vector_type(space)
      DistributedMultiFieldFESpace(field_fe_space,part_fe_spaces,gids,vector_type)
    end
  end
end

function Arrays.evaluate!(spacex::DistributedFESpace,space::DistributedFESpace,x::AbstractRealization)
  map(local_views(spacex),local_views(space)) do spacex,space
    Arrays.evaluate!(spacex,space,x)
  end
  return spacex
end

for T in (:AbstractRealization,:Nothing)
  S = T==:Nothing ? :Nothing : :Any
  for f in (:(Arrays.evaluate),:(ODEs.allocate_space))
    @eval begin
      function $f(space::DistributedUnEvalTrialFESpace,x::$T,y::$S)
        spaces = map(local_views(space)) do space
          $f(space,x,y)
        end
        gids  = get_free_dof_ids(space)
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
        vector_type = get_vector_type(space)
        DistributedMultiFieldFESpace(field_fe_space,part_fe_spaces,gids,vector_type)
      end
    end
  end

  @eval begin
    function Arrays.evaluate!(spacex::DistributedFESpace,space::DistributedFESpace,x::$T,y::$S)
      map(local_views(spacex),local_views(space)) do spacex,space
        Arrays.evaluate!(spacex,space,x,y)
      end
      return spacex
    end
  end
end

function ParamFESpaces.has_param(space::DistributedMultiFieldFESpace)
  getany(map(ParamFESpaces.has_param,local_views(space)))
end

function ParamODEs.has_param_transient(space::DistributedMultiFieldFESpace)
  getany(map(ParamODEs.has_param_transient,local_views(space)))
end

const DistributedSingleFieldParamFESpace = DistributedSingleFieldFESpace{<:AbstractArray{<:SingleFieldParamFESpace}}
const DistributedMultiFieldParamFESpace{MS} = DistributedMultiFieldFESpace{MS,<:AbstractVector{<:DistributedSingleFieldFESpace}}
const DistributedParamFESpace = Union{DistributedSingleFieldParamFESpace,DistributedMultiFieldParamFESpace}

function ParamDataStructures.param_length(f::DistributedParamFESpace)
  getany(map(param_length,local_views(f)))
end

function FESpaces.zero_free_values(f::DistributedSingleFieldParamFESpace)
  param_zero_free_values(f)
end

function FESpaces.zero_free_values(f::DistributedMultiFieldParamFESpace{<:BlockMultiFieldStyle})
  mortar(map(zero_free_values,f.field_fe_space))
end

function GridapDistributed.DistributedMultiFieldFEFunction(
  field_fe_fun::AbstractVector{<:GridapDistributed.DistributedSingleFieldFEFunction},
  part_fe_fun::AbstractArray{<:MultiFieldParamFEFunction},
  free_values::AbstractVector
  )

  metadata = GridapDistributed.DistributedFEFunctionData(free_values)
  GridapDistributed.DistributedMultiFieldCellField(field_fe_fun,part_fe_fun,metadata)
end

function Utils.collect_cell_matrix_for_trian(
  trial::DistributedFESpace,
  test::DistributedFESpace,
  a::GridapDistributed.DistributedDomainContribution,
  trian::GridapDistributed.DistributedTriangulation)

  map(collect_cell_matrix_for_trian,
    local_views(trial),
    local_views(test),
    local_views(a),
    local_views(trian))
end

function FESpaces.SparseMatrixAssembler(
  trial::DistributedParamFESpace,
  test::DistributedFESpace,
  par_strategy=SubAssembledRows()
  )

  PT = getany(map(get_vector_type,local_views(trial)))
  T  = eltype2(PT)
  Tm = SparseMatrixCSC{T,Int}
  Tv = Vector{T}
  SparseMatrixAssembler(Tm,Tv,trial,test,par_strategy)
end

function ParamDataStructures.parameterize(a::GridapDistributed.DistributedSparseMatrixAssembler,plength::Int)
  assems = map(local_views(a)) do assem
    parameterize(assem,plength)
  end
  matrix_builder = parameterize(a.matrix_builder,plength)
  vector_builder = parameterize(a.vector_builder,plength)

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

function DofMaps.get_dof_map(f::DistributedFESpace)
  map(local_views(f)) do f
    get_dof_map(f)
  end
end

function DofMaps.get_sparse_dof_map(f::DistributedFESpace,g::DistributedFESpace)
  map(local_views(f),local_views(g)) do f,g
    get_sparse_dof_map(f,g)
  end
end

function ParamODEs.collect_param_solutions(sol::ODEParamSolution{<:PVector{T}}) where T
  ncols = num_params(sol.r)*num_times(sol.r)
  partition = map(local_values(sol.u0)) do u0item
    ParamODEs._allocate_solutions(u0item,ncols)
  end
  sols = PVector(partition,sol.u0.index_partition)
  for (k,(rk,uk)) in enumerate(sol)
    ParamODEs._collect_solutions!(sols,uk,k)
  end
  return sols
end

function ParamODEs.collect_param_solutions(sol::ODEParamSolution{<:BlockPArray})

end

function ParamODEs._collect_solutions!(
  sols::PVector{<:ConsecutiveParamArray},
  ui::PVector{<:ConsecutiveParamArray},
  it::Int)

  map(local_views(sols),local_views(ui)) do sols,ui
    ParamODEs._collect_solutions!(sols.data,ui,it)
  end
end
