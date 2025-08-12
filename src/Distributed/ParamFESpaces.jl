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

for T in (:AbstractRealization,:Nothing)
  S = T==:Nothing ? :Nothing : :Any
  @eval begin
    function Arrays.evaluate(space::DistributedUnEvalTrialFESpace,x::$T)
      spaces = map(local_views(space)) do space
        Arrays.evaluate(space,x)
      end
      gids  = get_free_dof_ids(space)
      trian = get_triangulation(space)
      vector_type = get_vector_type(space)
      DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
    end

    function Arrays.evaluate(space::DistributedUnEvalTrialFESpace,x::$T,y::$S)
      spaces = map(local_views(space)) do space
        Arrays.evaluate(space,x,y)
      end
      gids  = get_free_dof_ids(space)
      trian = get_triangulation(space)
      vector_type = get_vector_type(space)
      DistributedSingleFieldFESpace(spaces,gids,trian,vector_type)
    end

    function Arrays.evaluate!(spacex::DistributedFESpace,space::DistributedFESpace,x::$T)
      map(local_views(spacex),local_views(space)) do spacex,space
        Arrays.evaluate!(spacex,space,x)
      end
      return spacex
    end

    function Arrays.evaluate!(spacex::DistributedFESpace,space::DistributedFESpace,x::$T,y::$S)
      map(local_views(spacex),local_views(space)) do spacex,space
        Arrays.evaluate!(spacex,space,x,y)
      end
      return spacex
    end
  end
end

const DistributedSingleFieldParamFESpace = DistributedSingleFieldFESpace{<:AbstractArray{<:SingleFieldParamFESpace}}

function ParamDataStructures.param_length(f::DistributedSingleFieldParamFESpace)
  PartitionedArrays.getany(map(param_length,local_views(f)))
end

function FESpaces.zero_free_values(f::DistributedSingleFieldParamFESpace)
  param_zero_free_values(f)
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
