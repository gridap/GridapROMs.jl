# check HighDimHyperReduction for more details
abstract type TimeCombination end

TimeCombination(odesolver::ODESolver) = @abstractmethod

get_coefficients(c::TimeCombination,args...) = @abstractmethod

function get_combination(
  c::TimeCombination,
  u::AbstractParamVector,
  us0::NTuple{N,AbstractParamVector}
  ) where N

  usx = ntuple(_ -> similar(u),Val{N}())
  for i in eachindex(us0)
    tcomb_i = CombinationOrder{i}(c)
    _combination!(usx[i],tcomb_i,u,us0)
  end
  return usx
end

struct ThetaMethodCombination <: TimeCombination
  dt::Real
  θ::Real
end

function TimeCombination(odesolver::ThetaMethod)
  ThetaMethodCombination(odesolver.dt,odesolver.θ)
end

struct GenAlpha1Combination <: TimeCombination
  dt::Real
  αf::Real
  αm::Real
  γ::Real
end

function TimeCombination(odesolver::GeneralizedAlpha1)
  GenAlpha1Combination(
    odesolver.dt,
    odesolver.αf,
    odesolver.αm,
    odesolver.γ
  )
end

struct GenAlpha2Combination <: TimeCombination
  dt::Real
  αf::Real
  αm::Real
  γ::Real
  β::Real
end

function TimeCombination(odesolver::GeneralizedAlpha2)
  GenAlpha2Combination(
    odesolver.dt,
    odesolver.αf,
    odesolver.αm,
    odesolver.γ,
    odesolver.β
  )
end

struct CombinationOrder{A,N} <: TimeCombination
  combination::A 
  CombinationOrder{N}(c::A) where {A,N} = new{A,N}(c)
end

const ThetaMethodStrategy{N} = CombinationOrder{ThetaMethodCombination,N}

function get_coefficients(c::ThetaMethodStrategy{1},args...)
  (c.combination.θ,1-c.combination.θ)
end

function get_coefficients(c::ThetaMethodStrategy{2},args...)
  (1/c.combination.dt,-1/c.combination.dt)
end

function _combination!(
  uθ::AbstractParamVector,
  c::ThetaMethodStrategy,
  u::AbstractParamVector,
  us0::NTuple{1,AbstractParamVector}
  )
  
  u0, = us0 

  np = param_length(u0)
  nt = round(Int,param_length(u) / np)
  η = get_coefficients(c,nt)

  dataθ = get_all_data(uθ)
  data = get_all_data(u)
  data0 = get_all_data(u0)
  
  for ipt = param_eachindex(u)
    it = slow_index(ipt,np)
    if it == 1
      for is in axes(data,1)
        dataθ[is,ipt] = η[1]*data[is,ipt] + η[2]*data0[is,ipt]
      end
    else
      for is in axes(data,1)
        dataθ[is,ipt] = η[1]*data[is,ipt] + η[2]*data[is,ipt-np]
      end
    end
  end

  return dataθ
end

const GenAlpha1Strategy{N} = CombinationOrder{GenAlpha1Combination,N}

function get_coefficients(c::GenAlpha1Strategy{1},args...)
  (c.combination.αf,1-c.combination.αf)
end

function get_coefficients(c::GenAlpha1Strategy{2},N::Int)
  @unpack dt,αf,αm,γ = c.combination 
  a = 1 / (γ*dt)
  b = 1 - 1/γ
  c = a * (1 - αm + b*αm)
  η = (a*αm,c - a*αm)
  for j in 3:N
    η = (η...,c*(b^(j-2) - b^(j-3)))
  end
  η
end

function _combination!(
  uα::AbstractParamVector,
  c::GenAlpha1Strategy{1},
  u::AbstractParamVector,
  us0::NTuple{2,AbstractParamVector}
  )
  
  u0, = us0 

  np = param_length(u0)
  nt = round(Int,param_length(u) / np)
  η = get_coefficients(c,nt)

  dataα = get_all_data(uα)
  data = get_all_data(u)
  data0 = get_all_data(u0)
  
  for ipt = param_eachindex(u)
    it = slow_index(ipt,np)
    if it == 1
      for is in axes(data,1)
        dataα[is,ipt] = η[1]*data[is,ipt] + η[2]*data0[is,ipt]
      end
    else
      for is in axes(data,1)
        dataα[is,ipt] = η[1]*data[is,ipt] + η[2]*data[is,ipt-np]
      end
    end
  end

  return dataα
end

function _combination!(
  vα::AbstractParamVector,
  c::GenAlpha1Strategy{2},
  u::AbstractParamVector,
  us0::NTuple{2,AbstractParamVector}
  )
  
  u0,v0 = us0 
  @unpack dt,αf,αm,γ = c.combination 
  a = 1 / (γ*dt)
  b = 1 - 1/γ
  c = a * (1 - αm + b*αm)

  np = param_length(u0)

  ddataα = get_all_data(vα)
  data = get_all_data(u)
  data0 = get_all_data(u0)
  ddata0 = get_all_data(v0)
  for ipt = param_eachindex(u)
    it = slow_index(ipt,np)
    if it == 1
      for is in axes(data,1)
        ddataα[is,ipt] = a*αm*data[is,ipt] - a*αm*data0[is,ipt] + c / a * ddata0[is,ipt]
      end
    else
      for is in axes(data,1)
        ddataα[is,ipt] = a*αm*data[is,ipt] + (c - a*αm)*data[is,ipt-np]
        for (j,ipt_back) in enumerate(ipt-2*np : -np : 1)
          ddataα[is,ipt] += c*(b^(j-1) - b^(j-2))*data[is,ipt_back]
        end
      end
    end
  end

  return ddataα
end

const GenAlpha2Strategy{N} = CombinationOrder{GenAlpha2Combination,N}

