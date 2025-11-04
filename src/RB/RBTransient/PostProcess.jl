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
  save(dir,op.rhs;label=_get_label(label,"rhs"))
  for (i,lhsi) in enumerate(op.lhs)
    save(dir,lhsi;label=_get_label(label,"lhs_$i"))
  end
end

function DrWatson.save(dir,op::TransientRBOperator;kwargs...)
  RBSteady._save_fixed_operator_parts(dir,op;kwargs...)
  RBSteady._save_trian_operator_parts(dir,op;kwargs...)
end

function RBSteady._load_trian_operator_parts(dir,feop::ODEParamOperator;label="")
  trian_res = get_domains_res(feop)
  trian_jacs = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res;label=_get_label(label,"rhs"))
  red_lhs = load_contribution(dir,trian_jacs;label=_get_label(label,"lhs"))
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
    dir,feop_lin;label=_get_label("lin",label))
  red_lhs_nlin,red_rhs_nlin = RBSteady._load_trian_operator_parts(
    dir,feop_nlin;label=_get_label("nlin",label))
  op_lin = RBOperator(feop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = RBOperator(feop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
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
      solitp = param_getindex(sol,ip,it)
      solitp_approx = param_getindex(sol_approx,ip,it)
      err_norm[it] = induced_norm(solitp-solitp_approx,args...)
      sol_norm[it] = induced_norm(solitp,args...)
    end
    errors[ip] = norm(err_norm) / norm(sol_norm)
  end
  return mean(errors)
end

function RBSteady.plot_a_solution(dir,Ω,uh,ûh,r::TransientRealization)
  T = eltype2(get_free_dof_values(uh))
  np = num_params(r)
  for i in 1:num_times(r)
    uhi = param_getindex(uh,(i-1)*np+1)
    ûhi = param_getindex(ûh,(i-1)*np+1)
    ehi = uhi - ûhi
    RBSteady._writevtk(T,Ω,dir*"_$i.vtu",uhi,ûhi,ehi)
  end
end

function RBSteady.to_snapshots(rbop::AbstractLocalRBOperator,x̂::AbstractParamVector,r::TransientRealization)
  xvec = map(enumerate(get_params(r))) do (i,μ)
    x̂μ = param_getindex(x̂,i)
    opμ = get_local(rbop,μ)
    trialμ = get_trial(opμ)
    inv_project(trialμ,x̂μ)
  end
  x = ParamArray(xvec)
  i = get_dof_map(rbop)
  s = Snapshots(x,i,r)
  _permutelastdims(s)
end

function _permutelastdims(s::Snapshots{T,N}) where {T,N}
  ids = (ntuple(i->i,Val(N-2))...,N,N-1)
  data = permutedims(get_all_data(s),ids)
  pids = (:,size(data,N-1),size(data,N))
  pdata = ConsecutiveParamArray(reshape(data,pids))
  Snapshots(pdata,get_dof_map(s),get_realization(s))
end

function _permutelastdims(s::TransientSnapshotsWithIC)
  TransientSnapshotsWithIC(s.initial_data,_permutelastdims(s.snaps))
end

function _permutelastdims(s::BlockSnapshots{T,N}) where {T,N}
  array = Array{Snapshots,N}(undef,size(s))
  touched = s.touched
  for i in eachindex(touched)
    if touched[i]
      array[i] = _permutelastdims(s[i])
    end
  end
  return BlockSnapshots(array,touched)
end
