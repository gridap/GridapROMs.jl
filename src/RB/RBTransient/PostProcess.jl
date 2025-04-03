function RBSteady.load_contribution(
  dir,
  trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}},
  args...;
  label="",
  kwargs...)

  c = ()
  for (i,trian) in enumerate(trians)
    ci = load_contribution(dir,trian,args...;label=_get_label(label,i),kwargs...)
    c = (c...,ci)
  end
  return c
end

function DrWatson.save(
  dir,
  contribs::Tuple{Vararg{Contribution}},
  ::ODEParamOperator;
  label="jac")

  for (i,contrib) in enumerate(contribs)
    save(dir,contrib;label=_get_label(label,i))
  end
end

function DrWatson.save(
  dir,
  contrib::Tuple{Vararg{Contribution}},
  feop::LinearNonlinearODEParamOperator;
  label="res")

  @check length(contrib) == 2
  save(dir,first(contrib),get_linear_operator(feop);label=_get_label(label,"lin"))
  save(dir,last(contrib),get_nonlinear_operator(feop);label=_get_label(label,"nlin"))
end

function DrWatson.save(
  dir,
  contribs::Tuple{Vararg{Tuple{Vararg{Contribution}}}},
  feop::LinearNonlinearODEParamOperator;
  label="jac")

  @check length(contribs) == 2
  save(dir,first(contribs),get_linear_operator(feop);label=_get_label(label,"lin"))
  save(dir,last(contribs),get_nonlinear_operator(feop);label=_get_label(label,"nlin"))
end

function RBSteady.load_residuals(dir,feop::LinearNonlinearODEParamOperator;label="res")
  res_lin = load_residuals(dir,get_linear_operator(feop);label=_get_label(label,"lin"))
  res_nlin = load_residuals(dir,get_nonlinear_operator(feop);label=_get_label(label,"nlin"))
  return (res_lin,res_nlin)
end

function RBSteady.load_jacobians(dir,feop::LinearNonlinearODEParamOperator;label="jac")
  jac_lin = load_jacobians(dir,get_linear_operator(feop);label=_get_label(label,"lin"))
  jac_nlin = load_jacobians(dir,get_nonlinear_operator(feop);label=_get_label(label,"nlin"))
  return (jac_lin,jac_nlin)
end

function RBSteady._save_trian_operator_parts(dir,op::TransientRBOperator;label="")
  save(dir,op.rhs;label=_get_label(label,"rhs"))
  for (i,lhsi) in enumerate(op.lhs)
    save(dir,lhsi;label=_get_label(label,"lhs_$i"))
  end
end

function DrWatson.save(dir,op::TransientRBOperator;kwargs...)
  RBSteady._save_fixed_operator_parts(dir,op;kwargs...)
  RBSteady._save_trian_operator_parts(dir,op;kwargs...)
end

function RBSteady._load_trian_operator_parts(dir,feop::ODEParamOperator,trial,test;label="")
  trian_res = get_domains_res(feop)
  trian_jacs = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res,test;label=_get_label(label,"rhs"))
  red_lhs = load_contribution(dir,trian_jacs,trial,test;label=_get_label(label,"lhs"))
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_odeop = change_domains(feop,trians_rhs,trians_lhs)
  return new_odeop,red_lhs,red_rhs
end

function RBSteady.load_operator(dir,feop::ODEParamOperator;kwargs...)
  trial,test = RBSteady._load_fixed_operator_parts(dir,feop;kwargs...)
  feop′,red_lhs,red_rhs = RBSteady._load_trian_operator_parts(dir,feop,trial,test;kwargs...)
  op = GenericRBOperator(feop′,trial,test,red_lhs,red_rhs)
  return op
end

function RBSteady.load_operator(dir,feop::LinearNonlinearODEParamOperator;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  trial,test = RBSteady._load_fixed_operator_parts(dir,feop_lin;label)
  odeop_lin,red_lhs_lin,red_rhs_lin = RBSteady._load_trian_operator_parts(
    dir,feop_lin,trial,test;label=_get_label("lin",label))
  odeop_nlin,red_lhs_nlin,red_rhs_nlin = RBSteady._load_trian_operator_parts(
    dir,feop_nlin,trial,test;label=_get_label("nlin",label))
  op_lin = GenericRBOperator(odeop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = GenericRBOperator(odeop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  return LinearNonlinearRBOperator(op_lin,op_nlin)
end

function Utils.compute_relative_error(
  sol::TransientSnapshots{T,N},
  sol_approx::TransientSnapshots{T,N},
  args...) where {T,N}

  @check size(sol) == size(sol_approx)
  err_norm = zeros(num_times(sol))
  sol_norm = zeros(num_times(sol))
  errors = zeros(num_params(sol))
  @inbounds for ip = 1:num_params(sol)
    for it in 1:num_times(sol)
      solitp = param_getindex(sol,it,ip)
      solitp_approx = param_getindex(sol_approx,it,ip)
      err_norm[it] = induced_norm(solitp-solitp_approx,args...)
      sol_norm[it] = induced_norm(solitp,args...)
    end
    errors[ip] = norm(err_norm) / norm(sol_norm)
  end
  return mean(errors)
end

function RBSteady.plot_a_solution(dir,Ω,uh,ûh,r::TransientRealization)
  np = num_params(r)
  for i in 1:num_times(r)
    uhi = param_getindex(uh,(i-1)*np+1)
    ûhi = param_getindex(ûh,(i-1)*np+1)
    ehi = uhi - ûhi
    writevtk(Ω,dir*"_$i.vtu",cellfields=["uh"=>uhi,"ûh"=>ûhi,"eh"=>ehi])
  end
end
