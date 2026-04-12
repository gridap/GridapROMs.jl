function Algebra.solve(
  solver::ODESolver,
  op::TransientRBOperator,
  r::TransientRealisation,
  uhs0::Tuple{Vararg{Function}}
  )

  trial = get_trial(op)
  params = get_params(r)
  us0 = ()
  for uh0 in uhs0
    u0 = get_free_dof_values(uh0(params))
    û0 = project(trial,u0)
    us0 = (us0...,reduced_vector(û0,u0))
  end
  solve(solver,op,r,us0)
end

function ODEs.ode_finish!(
  uf::RBParamVector,
  solver::ODESolver,
  op::TransientRBOperator,
  r::TransientRealisation,
  statef::Tuple{Vararg{RBParamVector}},
  odecache
  )

  copyto!(uf,first(statef))
  inv_project(get_trial(op),uf)
  uf
end

for T in (:JointTransientRBOperator,:SplitTransientRBOperator)
  @eval begin
    function Algebra.residual!(
      b::HRParamArray,
      op::$T,
      r::TransientRealisationAt,
      us::Tuple{Vararg{AbstractVector}},
      paramcache
      )

      fill!(b,zero(eltype(b)))

      uh = ODEs._make_uh_from_us(op,us,paramcache.trial)

      test = get_test(op)
      v = get_fe_basis(test)

      trian_res = get_domains_res(op)
      μ,t = get_params(r),get_times(r)
      rhs = get_rhs(op)
      res = get_res(op)
      dc = res(μ,t,uh,v)

      for strian in trian_res
        b_strian = b.fecache[strian]
        rhs_strian = get_interpolation(rhs[strian])
        vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian)
        assemble_hr_array_add!(b_strian,vecdata...)
      end

      interpolate!(b,rhs)
    end

    function Algebra.jacobian!(
      A::HRParamArray,
      op::$T,
      r::TransientRealisationAt,
      us::Tuple{Vararg{AbstractVector}},
      ws::Tuple{Vararg{Real}},
      paramcache
      )

      fill!(A,zero(eltype(A)))

      uh = ODEs._make_uh_from_us(op,us,paramcache.trial)

      trial = get_trial(op)
      du = get_trial_fe_basis(trial)
      test = get_test(op)
      v = get_fe_basis(test)

      trian_jacs = get_domains_jac(op)
      μ,t = get_params(r),get_times(r)
      lhss = get_lhs(op)
      jacs = get_jacs(op)

      for k in 1:get_order(op)+1
        Ak = A.fecache[k]
        lhs = lhss[k]
        jac = jacs[k]
        w = ws[k]
        iszero(w) && continue
        dc = w * jac(μ,t,uh,du,v)
        trian_jac = trian_jacs[k]
        for strian in trian_jac
          A_strian = Ak[strian]
          lhs_strian = get_interpolation(lhs[strian])
          matdata = collect_cell_hr_matrix(trial,test,dc,strian,lhs_strian)
          assemble_hr_array_add!(A_strian,matdata...)
        end
      end

      interpolate!(A,lhss)
    end
  end 
end

function ParamODEs.collect_param_solutions(sol::ODEParamSolution{<:RBParamVector{T,<:ConsecutiveParamVector{T}}}) where T
  u0 = first(sol.us0)
  ncols = num_params(sol.r)*num_times(sol.r)
  sols = ParamODEs._allocate_solutions(u0,ncols)
  for (k,(rk,uk)) in enumerate(sol)
    ParamODEs._collect_solutions!(sols.data,uk.data,k)
  end
  trial = get_trial(sol.odeop)(sol.r)
  inv_project!(sols.fe_data,trial,sols.data)
  return sols
end

function ParamODEs.collect_param_solutions(sol::ODEParamSolution{<:RBParamVector{T,<:BlockParamVector{T}}}) where T
  u0 = first(sol.us0)
  u0item = testitem(u0)
  ncols = num_params(sol.r)*num_times(sol.r)
  sols = ParamODEs._allocate_solutions(u0,ncols)
  for (k,(rk,uk)) in enumerate(sol)
    for i in 1:blocklength(u0item)
      ParamODEs._collect_solutions!(blocks(sols.data)[i],blocks(uk.data)[i],k)
    end
  end
  trial = get_trial(sol.odeop)(sol.r)
  inv_project!(sols.fe_data,trial,sols.data)
  return sols
end

function ParamODEs._allocate_solutions(u0::RBParamVector,ncols) 
  data = ParamODEs._allocate_solutions(u0.data,ncols)
  fe_data = ParamODEs._allocate_solutions(u0.fe_data,ncols)
  RBParamVector(data,fe_data)
end