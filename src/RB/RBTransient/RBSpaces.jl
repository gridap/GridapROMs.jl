function ODEs.time_derivative(r::RBSpace)
  fet = time_derivative(get_fe_space(r))
  rb = get_reduced_subspace(r)
  reduced_subspace(fet,rb)
end

function RBSteady.project(r1::RBSpace,x::Projection,r2::RBSpace,combine::Function)
  galerkin_projection(get_reduced_subspace(r1),x,get_reduced_subspace(r2),combine)
end
