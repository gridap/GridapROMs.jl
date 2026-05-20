for P in (:Lines,:Scatter,:ScatterLines)
  @eval begin
    function convert_arguments(
      ::Type{$P},
      trian::Triangulation{Dc,1},
      f::AbstractParamFunction
      ) where Dc

      _nan_stack_1d(trian,f)
    end

    function convert_arguments(
      ::Type{$P},
      trian::Triangulation{Dc,1},
      uh::SingleFieldParamFEFunction
      ) where Dc

      _nan_stack_1d(trian,uh)
    end

    function convert_arguments(
      ::Type{$P},
      uh::SingleFieldParamFEFunction
      )

      msg = "Use plot(trian, uh) for non-1d fields; " * string($(QuoteNode(P))) * "(uh) only works for 1-D triangulations."
      trian = get_triangulation(uh)
      @assert num_point_dims(trian) == 1 msg
      _nan_stack_1d(trian,uh)
    end
  end
end

function convert_arguments(
  ::Type{<:MeshField},
  trian::Triangulation,
  uh::SingleFieldParamFEFunction
  )

  (trian,param_getindex(uh,1))
end

function convert_arguments(
  ::Type{<:MeshField},
  uh::SingleFieldParamFEFunction
  )
  
  trian = get_triangulation(uh)
  (trian,param_getindex(uh,1))
end

# utils 

function _nan_stack_1d(trian::Triangulation{Dc,1},f) where Dc
  xs = Float64[]
  ys = Float64[]
  for i in param_eachindex(f)
    x,y = _xy_1d(trian,param_getindex(f,i))
    append!(xs,x)
    append!(ys,Float64.(y))
    push!(xs,NaN)
    push!(ys,NaN)
  end
  return xs,ys
end

function _nan_stack_1d(
  trian::Triangulation{Dc,1},
  uh::SingleFieldParamFEFunction
  ) where Dc

  xs = Float64[]
  ys = Float64[]
  for i in param_eachindex(uh)
    x,y = _xy_1d(trian,param_getindex(uh,i))
    append!(xs,x)
    append!(ys,Float64.(y))
    push!(xs,NaN)
    push!(ys,NaN)
  end
  return xs,ys
end

function _xy_1d(trian::Triangulation{Dc,1},uh) where Dc
  vds = first(visualization_data(trian,"",cellfields = ["uh" => uh]))
  y = to_scalar.(vds.nodaldata["uh"])
  x = getindex.(get_node_coordinates(vds.grid),1)
  idx = sortperm(x)
  return x[idx],y[idx]
end