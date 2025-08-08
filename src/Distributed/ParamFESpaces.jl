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
  end
end

function Utils.collect_cell_matrix_for_trian(
  trial::GridapDistributed.DistributedFESpace,
  test::GridapDistributed.DistributedFESpace,
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
  GridapDistributed.DistributedSparseMatrixAssembler(a.par_strategy,assems,a.mat_builder,a.vec_builder,a.rows,a.cols)
end
