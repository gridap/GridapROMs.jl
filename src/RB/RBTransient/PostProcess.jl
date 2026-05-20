function DrWatson.save(
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
  trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}};
  label=""
  )

  c = ()
  for (i,trian) in enumerate(trians)
    c = (c...,load_contribution(dir,trian;label=_get_label(label,i)))
  end
  return c
end

function RBSteady._save_trian_operator_parts(dir,op::TransientRBOperator;label="")
  save(dir,op.rhs;label=_get_label(label,rhs_label))
  for (i,lhsi) in enumerate(op.lhs)
    save(dir,lhsi;label=_get_label(label,_get_label(lhs_label,i)))
  end
end

function DrWatson.save(dir,op::TransientRBOperator;kwargs...)
  RBSteady._save_fixed_operator_parts(dir,op;kwargs...)
  RBSteady._save_trian_operator_parts(dir,op;kwargs...)
end

function RBSteady._load_trian_operator_parts(dir,feop::ODEParamOperator;label="")
  trian_res = get_domains_res(feop)
  trian_jacs = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res;label=_get_label(label,rhs_label))
  red_lhs = load_contribution(dir,trian_jacs;label=_get_label(label,lhs_label))
  return red_lhs,red_rhs
end

function RBSteady.load_operator(dir,feop::ODEParamOperator;kwargs...)
  trial,test = RBSteady._load_fixed_operator_parts(dir,feop;kwargs...)
  red_lhs,red_rhs = RBSteady._load_trian_operator_parts(dir,feop;kwargs...)
  op = RBOperator(feop,trial,test,red_lhs,red_rhs)
  return op
end

function RBSteady.load_operator(dir,feop::LinearNonlinearODEParamOperator;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  trial,test = RBSteady._load_fixed_operator_parts(dir,feop_lin;label)
  red_lhs_lin,red_rhs_lin = RBSteady._load_trian_operator_parts(
    dir,feop_lin;label=_get_label(linear_label,label))
  red_lhs_nlin,red_rhs_nlin = RBSteady._load_trian_operator_parts(
    dir,feop_nlin;label=_get_label(nonlinear_label,label))
  op_lin = RBOperator(feop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = RBOperator(feop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  return LinearNonlinearRBOperator(op_lin,op_nlin)
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

function RBSteady._plot_solutions(dir,trian,uh,ûh,r::TransientRealisation;field=1)
  T = eltype2(get_free_dof_values(uh))
  np = num_params(r)
  nt = num_times(r)
  ptrian = num_point_dims(trian) < 3 ? trian : BoundaryTriangulation(get_background_model(trian))
  for ip in 1:np
    dir_param = joinpath(dir,"param$ip")
    RBSteady.create_dir(dir_param)
    ufields = [param_getindex(uh,(it-1)*np+ip) for it in 1:nt]
    ûfields = [param_getindex(ûh,(it-1)*np+ip) for it in 1:nt]
    efields = [ufields[it]-ûfields[it] for it in 1:nt]
    if T <: Complex
      ufields = abs2.(ufields)
      ûfields = abs2.(ûfields)
      efields = abs2.(efields)
    end
    it_obs = Makie.Observable(1)
    uplot = Makie.lift(i->ufields[i],it_obs)
    ûplot = Makie.lift(i->ûfields[i],it_obs)
    eplot = Makie.lift(i->efields[i],it_obs)
    fig = Makie.Figure()
    Makie.plot(fig[1,1],ptrian,uplot)
    Makie.plot(fig[1,2],ptrian,ûplot)
    Makie.plot(fig[1,3],ptrian,eplot)
    Makie.record(fig,joinpath(dir_param,"field_$(field).gif"),1:nt) do it
      it_obs[] = it
    end
  end
end

include("Diagnostics.jl")