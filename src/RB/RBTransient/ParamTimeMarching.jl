function ODEs.ode_finish!(
  uf::RBParamVector,
  solver::ODESolver,
  op::RBOperator,
  r::TransientRealisation,
  statef::Tuple{Vararg{AbstractVector}},
  odecache
  )

  copyto!(uf.fe_data,first(statef))
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

