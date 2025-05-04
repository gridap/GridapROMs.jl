function ODEs.time_derivative(f::TransientParamFunction)
  function dfdt(μ,t)
    fμt = f.fun(μ,t)
    function dfdt_t(x)
      T = return_type(fμt,x)
      ODEs._time_derivative(T,f.fun,μ,t,x)
    end
    dfdt_t
  end
  TransientParamFunction(dfdt,f.params,f.times)
end

function ODEs._time_derivative(T::Type{<:Real},f,μ,t,x)
  partial(t) = f(μ,t)(x)
  ForwardDiff.derivative(partial,t)
end

function ODEs._time_derivative(T::Type{<:VectorValue},f,μ,t,x)
  partial(t) = get_array(f(μ,t)(x))
  VectorValue(ForwardDiff.derivative(partial,t))
end

function ODEs._time_derivative(T::Type{<:TensorValue},f,μ,t,x)
  partial(t) = get_array(f(μ,t)(x))
  TensorValue(ForwardDiff.derivative(partial,t))
end

# #TODO do not actually need the following, but the current implementation of
# Gridap's time_derivative won't allow multiple dispatching

function ∂ₚt(f)
  time_derivative_param_fun(f)
end

function ∂ₚt(f,::Val{k}) where k
  time_derivative_param_fun(f,Val(k))
end

function ∂ₚtt(f)
  time_derivative_param_fun(f,Val(2))
end

function time_derivative_param_fun(args...)
  time_derivative(args...)
end

function time_derivative_param_fun(f,::Val{0})
  f
end

function time_derivative_param_fun(f,::Val{1})
  time_derivative_param_fun(f)
end

function time_derivative_param_fun(f,::Val{k}) where k
  time_derivative_param_fun(time_derivative_param_fun(f),Val(k-1))
end

function time_derivative_param_fun(f::Function)
  function dfdt(μ,t)
    fμt = f(μ,t)
    function dfdt_t(x)
      T = return_type(fμt,x)
      ODEs._time_derivative(T,f,μ,t,x)
    end
    dfdt_t
  end
  dfdt
end
