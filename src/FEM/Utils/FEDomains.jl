filter_domains(a) = a
filter_domains(a::Tuple) = map(filter_domains,a)

"""
    struct FEDomains{A,B}
      domains_res::A
      domains_jac::B
    end

Fields:
- `domains_res`: triangulations relative to the residual (nothing by default)
- `domains_jac`: triangulations relative to the Jacobian (nothing by default)
"""
struct FEDomains{A,B}
  domains_res::A
  domains_jac::B

  function FEDomains(domains_res,domains_jac)
    domains_res′ = filter_domains(domains_res)
    domains_jac′ = filter_domains(domains_jac)
    A = typeof(domains_res′)
    B = typeof(domains_jac′)
    new{A,B}(domains_res′,domains_jac′)
  end
end

FEDomains(args...) = FEDomains(nothing,nothing)

get_domains_res(d::FEDomains) = d.domains_res
get_domains_jac(d::FEDomains) = d.domains_jac

"""
"""
abstract type OperatorType <: GridapType end

"""
"""
struct LinearEq <: OperatorType end

"""
"""
struct NonlinearEq <: OperatorType end

"""
    abstract type TriangulationStyle <: GridapType end

Subtypes:

- [`JointDomains`](@ref)
- [`SplitDomains`](@ref)
"""
abstract type TriangulationStyle <: GridapType end

"""
    struct JointDomains <: TriangulationStyle end

Trait for a FE operator indicating that residuals/Jacobians in this operator
should be computed summing the contributions relative to each triangulation as
occurs in [`Gridap`](@ref)
"""
struct JointDomains <: TriangulationStyle end

"""
    struct SplitDomains <: TriangulationStyle end

Trait for a FE operator indicating that residuals/Jacobians in this operator
should be computed keeping the contributions relative to each triangulation separate
"""
struct SplitDomains <: TriangulationStyle end
