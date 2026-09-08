function save(
  dir,
  contribs::Tuple{Vararg{Contribution}};
  label=""
  )

  for (i,contrib) in enumerate(contribs)
    save(dir,contrib;label=_get_label(label,i))
  end
end

function RBSteady.load_contribution(
  dir,
  trians::Tuple{Vararg{Tuple}};
  label=""
  )

  c = ()
  for (i,trian) in enumerate(trians)
    c = (c...,load_contribution(dir,trian;label=_get_label(label,i)))
  end
  return c
end

function RBSteady._save_trian_operator_parts(dir,op::TransientReducedOperator;label="")
  save(dir,op.rhs;label=_get_label(label,RHS_LABEL))
  for (i,lhsi) in enumerate(op.lhs)
    save(dir,lhsi;label=_get_label(label,_get_label(LHS_LABEL,i)))
  end
end

function save(dir,op::TransientReducedOperator;kwargs...)
  RBSteady._save_fixed_operator_parts(dir,op;kwargs...)
  RBSteady._save_trian_operator_parts(dir,op;kwargs...)
end

function RBSteady._load_trian_operator_parts(dir,feop::ODEParamOperator;label="")
  trian_res = get_domains_res(feop)
  trian_jacs = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res;label=_get_label(label,RHS_LABEL))
  red_lhs = load_contribution(dir,trian_jacs;label=_get_label(label,LHS_LABEL))
  return red_lhs,red_rhs
end

function RBSteady.load_operator(dir,feop::ODEParamOperator;kwargs...)
  trial,test = RBSteady._load_fixed_operator_parts(dir,feop;kwargs...)
  red_lhs,red_rhs = RBSteady._load_trian_operator_parts(dir,feop;kwargs...)
  op = ReducedOperator(feop,trial,test,red_lhs,red_rhs)
  return op
end

function RBSteady.load_operator(dir,feop::LinearNonlinearODEParamOperator;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  trial,test = RBSteady._load_fixed_operator_parts(dir,feop_lin;label)
  red_lhs_lin,red_rhs_lin = RBSteady._load_trian_operator_parts(
    dir,feop_lin;label=_get_label(LINEAR_LABEL,label))
  red_lhs_nlin,red_rhs_nlin = RBSteady._load_trian_operator_parts(
    dir,feop_nlin;label=_get_label(NONLINEAR_LABEL,label))
  op_lin = ReducedOperator(feop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = ReducedOperator(feop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  return LinearNonlinearReducedOperator(op_lin,op_nlin)
end

function Utils.compute_relative_error(
  sol::TransientSnapshots{T,N},
  sol_approx::TransientSnapshots{T,N},
  args...
  ) where {T,N}

  @check size(sol) == size(sol_approx)
  err_norm = zeros(num_times(sol))
  sol_norm = zeros(num_times(sol))
  errors = zeros(num_params(sol))
  @inbounds for ip = 1:num_params(sol)
    for it in 1:num_times(sol)
      solitp = param_getindex(sol,ip,it)
      solitp_approx = param_getindex(sol_approx,ip,it)
      err_norm[it] = induced_norm(solitp-solitp_approx,args...)
      sol_norm[it] = induced_norm(solitp,args...)
    end
    errors[ip] = norm(err_norm) / norm(sol_norm)
  end
  return mean(errors)
end

include("Diagnostics.jl")