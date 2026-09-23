function save(
  dir,
  contribs::ContributionTuple;
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
  return ContributionTuple(c)
end

function RBSteady.load_operator(dir,feop::LinearNonlinearODEParamOperator;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  # test and trial are the same for both the linear and nonlinear operators
  test = load_subspace(dir,get_test(feop_lin);label=_get_label(label,TEST_LABEL))
  trial = load_subspace(dir,get_trial(feop_lin);label=_get_label(label,TRIAL_LABEL))
  # weakform-related quantities are different for the linear and nonlinear operators
  red_rhs_lin = load_contribution(dir,get_domains_res(feop_lin);label=_get_label(label,LINEAR_LABEL,RHS_LABEL))
  red_lhs_lin = load_contribution(dir,get_domains_jac(feop_lin);label=_get_label(label,LINEAR_LABEL,LHS_LABEL))
  red_rhs_nlin = load_contribution(dir,get_domains_res(feop_nlin);label=_get_label(label,NONLINEAR_LABEL,RHS_LABEL))
  red_lhs_nlin = load_contribution(dir,get_domains_jac(feop_nlin);label=_get_label(label,NONLINEAR_LABEL,LHS_LABEL))
  op_lin = ReducedOperator(feop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = ReducedOperator(feop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  return LinearNonlinearReducedOperator(op_lin,op_nlin)
end

include("Diagnostics.jl")