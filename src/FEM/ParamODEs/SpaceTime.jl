include("TimeCombinations.jl")

function spacetime_residual(
  c::TimeCombination,
  odeop::ODEParamOperator,
  s::AbstractSnapshots
  )

  r = get_realisation(s)
  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  spacetime_residual(c,odeop,r,u,us0)
end

function spacetime_residual(
  c::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  us0::Tuple{Vararg{AbstractParamVector}}
  )

  b,usx,paramcache = allocate_spacetime_residual(c,odeop,r,u,us0)
  spacetime_residual!(b,c,odeop,r,u,usx,us0,paramcache)
  return b
end

function allocate_spacetime_residual(
  c::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  us0::Tuple{Vararg{AbstractParamVector}}
  )

  to_stencil!(r,c)
  paramcache = allocate_paramcache(odeop,r;evaluated=true)
  from_stencil!(r,c)
  usx = allocate_time_combination(u,us0)
  b = allocate_residual(odeop,r,usx,paramcache)
  return b,usx,paramcache
end

function spacetime_residual!(
  b,
  c::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  usx::Tuple{Vararg{AbstractParamVector}},
  us0::Tuple{Vararg{AbstractParamVector}},
  paramcache
  )

  time_combination!(usx,c,u,us0)
  to_stencil!(r,c)
  residual!(b,odeop,r,usx,paramcache)
  from_stencil!(r,c)
  return b
end

function spacetime_residual!(
  b,
  c::TimeCombination,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  u::AbstractParamVector,
  usx::Tuple{Vararg{AbstractParamVector}},
  us0::Tuple{Vararg{AbstractParamVector}},
  paramcache
  )

  zero_time_combination!(usx,c,us0)
  to_stencil!(r,c)
  residual!(b,odeop,r,usx,paramcache)
  from_stencil!(r,c)
  return b
end

function spacetime_jacobian(
  c::TimeCombination,
  odeop::ODEParamOperator,
  s::AbstractSnapshots
  )

  r = get_realisation(s)
  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  spacetime_jacobian(c,odeop,r,u,us0)
end

function spacetime_jacobian(
  c::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  us0::Tuple{Vararg{AbstractParamVector}}
  )

  A,usx,paramcache = allocate_spacetime_jacobian(c,odeop,r,u,us0)
  spacetime_jacobian!(A,c,odeop,r,u,usx,us0,paramcache)
  return A
end

function allocate_spacetime_jacobian(
  c::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  us0::Tuple{Vararg{AbstractParamVector}}
  )

  to_stencil!(r,c)
  paramcache = allocate_paramcache(odeop,r;evaluated=true)
  from_stencil!(r,c)
  usx = allocate_time_combination(u,us0)
  A = allocate_jacobian(odeop,r,usx,paramcache)
  return A,usx,paramcache
end

function spacetime_jacobian!(
  A,
  c::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  usx::Tuple{Vararg{AbstractParamVector}},
  us0::Tuple{Vararg{AbstractParamVector}},
  paramcache
  )

  ws = ntuple(_ -> 1,Val(length(us0)))
  time_combination!(usx,c,u,us0)
  to_stencil!(r,c)
  jacobian!(A,odeop,r,usx,ws,paramcache)
  from_stencil!(r,c)
  return A
end

function spacetime_jacobian!(
  A,
  c::TimeCombination,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  u::AbstractParamVector,
  usx::Tuple{Vararg{AbstractParamVector}},
  us0::Tuple{Vararg{AbstractParamVector}},
  paramcache
  )

  ws = ntuple(_ -> 1,Val(length(us0)))
  zero_time_combination!(usx,c,us0)
  to_stencil!(r,c)
  jacobian!(A,odeop,r,usx,ws,paramcache)
  from_stencil!(r,c)
  return A
end

get_stencil_shift(s::GridapType) = get_stencil_shift(TimeCombination(s))
get_stencil_shift(c::TimeCombination) = @abstractmethod
get_stencil_shift(c::CombinationOrder) = get_stencil_shift(c.combination)
get_stencil_shift(c::ThetaMethodCombination) = c.dt*(1-c.θ)
get_stencil_shift(c::GenAlpha1Combination) = c.dt*(1-c.αf)
get_stencil_shift(c::GenAlpha2Combination) = c.dt*c.αf

function to_stencil!(r::TransientRealisation,s_or_c)
  δ = get_stencil_shift(s_or_c)
  shift!(r,-δ)
end

function from_stencil!(r::TransientRealisation,s_or_c)
  δ = get_stencil_shift(s_or_c)
  shift!(r,δ)
end
