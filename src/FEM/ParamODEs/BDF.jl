get_bdf_coeffs(::Val{k}) where k = @abstractmethod
get_bdf_coeffs(::Val{1}) = (-1,)
get_bdf_coeffs(::Val{2}) = (-4/3,1/3)
get_bdf_coeffs(::Val{3}) = (-18/11,9/11,-2/11)
get_bdf_coeffs(::Val{4}) = (-48/25,36/25,-16/25,3/25)
get_bdf_coeffs(::Val{5}) = (-300/137,300/137,-200/137,75/137,-12/137)
get_bdf_coeffs(::Val{6}) = (-360/147,450/147,-400/147,225/147,-72/147,10/147)

get_rhs_coeff(::Val{k}) where k = -sum(get_bdf_coeffs(Val(k)))

"""
"""
struct BDF{N} <: ODESolver
  sysslvr::NonlinearSolver
  dt::Real
  coeffs::NTuple{Number,N}
  function BDF{N}(sysslvr::NonlinearSolver,dt::Real)
    coeffs = get_bdf_coeffs(Val(N))
    new{N}(sysslvr,dt,coeffs)
  end
end

get_rhs_coeff(odeslvr::BDF{N}) where N = get_rhs_coeff(Val(N))

function ODEs.allocate_odecache(
  odeslvr::BDF{N},
  odeop::ODEOperator,
  t0::Real,
  us0::NTuple{1,AbstractVector}
  ) where N

  u0 = us0[1]
  v0 = us0[1]
  allocate_odecache(odeslvr,odeop,t0,u0)
end

##################
# Nonlinear case #
##################
function allocate_odecache(
  odeslvr::BDF{N},
  odeop::ODEOperator,
  t0::Real,
  us0::NTuple{N,AbstractVector}
  ) where N

  u0 = us0[1]
  v0 = sum(us0[2:N])
  us0N = (u0,v0)
  odeopcache = allocate_odeopcache(odeop,t0,us0N)

  uF = copy(u0)

  sysslvrcache = nothing
  odeslvrcache = (uF,sysslvrcache)

  (odeslvrcache,odeopcache)
end


function ode_march!(
  stateF::NTuple{N,AbstractVector},
  odeslvr::BDF,
  odeop::ODEOperator,
  t0::Real,
  state0::NTuple{N,AbstractVector},
  odecache)

  # Unpack inputs
  u0 = state0[1]
  odeslvrcache,odeopcache = odecache
  uF,sysslvrcache = odeslvrcache

  # Unpack solver
  sysslvr = odeslvr.sysslvr
  dt,coeffs = odeslvr.dt,odeslvr.coeffs

  # Define scheme
  x = stateF[1]
  tx = t0 + dt
  function usx(x)
    copy!(uθ,u0)
    axpy!(dtθ,x,uθ)
    (uθ,x)
  end
  ws = (1,1)

  update_odeopcache!(odeopcache,odeop,tx)

  stageop = NonlinearStageOperator(
    odeop,odeopcache,
    tx,usx,ws
  )

  sysslvrcache = solve!(x,sysslvr,stageop,sysslvrcache)

  tF = tx
  stateF = _update_bdf!(stateF,state0,dt,x)

  odeslvrcache = (uθ,sysslvrcache)
  odecache = (odeslvrcache,odeopcache)
  (tF,stateF,odecache)
end
