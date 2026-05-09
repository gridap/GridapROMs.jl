function ODEs.time_derivative(r::RBSpace)
  fet = time_derivative(get_fe_space(r))
  rb = get_reduced_subspace(r)
  reduced_subspace(fet,rb)
end

for (f,f!) in zip((:space_project,:space_inv_project),(:space_project!,:inv_space_project!))
  @eval begin
    function $f(r::RBSpace,x::AbstractVector)
      $f(get_reduced_subspace(r),x)
    end

    function $f!(y,r::RBSpace,x::AbstractVector)
      $f!(y,get_reduced_subspace(r),x)
    end
  end
end

function space_project(r::RBSpace,a::RBParamVector)
  space_project!(a.data,r,a.fe_data)
  return a.data
end

function space_inv_project(r::RBSpace,a::RBParamVector)
  inv_space_project!(a.fe_data,r,a.data)
  return a.fe_data
end
