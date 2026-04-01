# check HighDimHyperReduction for more details
abstract type TimeCombination end

TimeCombination(odesolver::ODESolver) = @abstractmethod

get_coefficients(c::TimeCombination,args...) = @abstractmethod

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

const GenAlpha1Strategy{N} = CombinationOrder{GenAlpha1Combination,N}

function get_coefficients(c::GenAlpha1Strategy{1},args...)
  (c.combination.αf,1-c.combination.αf)
end

function get_coefficients(c::GenAlpha1Strategy{2},N::Int)
  @unpack dt,αf,αm,γ = c.combination 
  a = 1 / (γ*dt)
  b = 1 - 1/γ
  c = a * (1 - αm + b*αm)
  θ = (a * αm,c - a*αm)
  for j in 3:N
    θ = (θ...,c*(b^(j-2) - b^(j-3)))
  end
  θ
end

const GenAlpha2Strategy{N} = CombinationOrder{GenAlpha2Combination,N}

