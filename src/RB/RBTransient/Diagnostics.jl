function RBSteady.hr_diagnostics(c::TupOfAffineContribution)
  Tuple(RBSteady.hr_diagnostics(v) for v in c)
end

function RBSteady.hr_error(
  op::RBOperator,
  res::AbstractSnapshots,
  jac::Tuple{Vararg{AbstractSnapshots}},
  μ::AbstractRealisation
  )

  err_res = RBSteady.hr_error_res(op,res,μ)
  err_jac = RBSteady.hr_error_jac(op,jac,μ)
  return err_res,err_jac
end

function RBSteady.hr_error_jac(
  op::RBOperator,
  jac::Tuple{Vararg{AbstractSnapshots}},
  μ::AbstractRealisation
  )

  lhs = get_lhs(op)    
  test = get_test(op)
  trial = get_trial(op)
  Ψ_test = get_basis(get_reduced_subspace(test))
  Ψ_trial = get_basis(get_reduced_subspace(trial))
  np = num_params(μ)

  map(lhs,jac) do lhs_k,jac_k
    Â_fom_k = galerkin_projection(Ψ_test,get_param_data(jac_k),Ψ_trial)
    Â_fom_p_k = permutedims(Â_fom_k,(1,3,2))

    fecache_k = contribution(get_domains(lhs_k)) do trian
      interp = get_interpolation(lhs_k[trian])
      rows = get_interpolation_rows(interp)
      isnothing(rows) ? get_param_data(jac_k) :
        _get_at_domain(jac_k,(rows,get_interpolation_cols(interp)))
    end

    cache_k = allocate_hrtrian_cache(lhs_k,trial,test,μ)
    A_k = DiagnosticsContribution(fecache_k,cache_k.coeff,cache_k.hypred)
    interpolate!(A_k,lhs_k,trial,test)

    map(get_contributions(A_k.hypred)) do h_t
      hr_data = get_all_data(h_t)
      mean(1:np) do i
        compute_relative_error(Â_fom_p_k,hr_data)
      end
    end |> Tuple
  end |> Tuple
end

# IO: save N derivative snapshots under separate labels (jac_1, jac_2, …)
function RBSteady.save_jacobians(
  dir,
  jac::Tuple{Vararg{AbstractSnapshots}};
  label=""
  )

  for (k,jac_k) in enumerate(jac)
    save(dir,jac_k;label=_get_label(label,"jac_$k"))
  end
end

# IO: load N derivative snapshots; falls back to computing fresh if missing
function RBSteady.load_jacobians(
  dir,
  rbsolver,
  feop::ODEParamOperator,
  fesnaps
  )

  n = get_time_order(get_fe_solver(rbsolver)) + 1
  try
    Tuple(load_snapshots(dir;label="jac_$k") for k in 1:n)
  catch
    jac = jacobian_snapshots(rbsolver,feop,fesnaps)
    save_jacobians(dir,jac)
    jac
  end
end
