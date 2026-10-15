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
    struct Bundle{A,B}
      linear::A
      nonlinear::B
    end
"""
struct Bundle{A,B}
  linear::A
  nonlinear::B
end

"""
    (b::Bundle)(args...)

A `Bundle` of two callables (e.g. two weak-form functions `res(μ,u,v)`) is itself
callable, and evaluates to the sum of its two halves. This is what lets a
Bundle-valued `res`/`jac` field be passed, unchanged, to any call site that expects
a plain weak-form function: the linear and nonlinear contributions are aggregated
by summation at evaluation time, instead of being cached and combined separately
"""
(b::Bundle)(args...) = b.linear(args...) + b.nonlinear(args...)

"""
    get_linear_part(b::Bundle) -> A
    get_nonlinear_part(b::Bundle) -> B

Return the linear/nonlinear half of a [`Bundle`](@ref). Extended, for the
relevant operator types, to split a `Bundle`-based linear+nonlinear operator
back into its two plainly-tagged halves (see e.g. `ParamFEOpFromWeakForm`,
`TransientParamFEOpFromWeakForm`, `RBOperator`)
"""
get_linear_part(b::Bundle) = b.linear
get_nonlinear_part(b::Bundle) = b.nonlinear

get_domains_res(b::Bundle) = Bundle(get_domains_res(b.linear),get_domains_res(b.nonlinear))
get_domains_jac(b::Bundle) = Bundle(get_domains_jac(b.linear),get_domains_jac(b.nonlinear))

change_domains(a::Bundle,trians::Bundle) = Bundle(change_domains(a.linear,trians.linear),change_domains(a.nonlinear,trians.nonlinear))

"""
    abstract type OperatorType <: GridapType end

Trait encoding the linearity structure of a parametric FE operator.

Subtypes:

- [`LinearEq`](@ref)
- [`NonlinearEq`](@ref)
"""
abstract type OperatorType <: GridapType end

"""
    struct LinearEq <: OperatorType end

Trait indicating that a parametric FE operator is fully linear.
"""
struct LinearEq <: OperatorType end

"""
    struct NonlinearEq <: OperatorType end

Trait indicating that a parametric FE operator is nonlinear.
"""
struct NonlinearEq <: OperatorType end

"""
    abstract type WeakFormStyle <: GridapType end

Subtypes:

- [`PlainForm`](@ref)
- [`BundleForm`](@ref)
"""
abstract type WeakFormStyle <: GridapType end

"""
    struct PlainForm <: WeakFormStyle end

Trait for a FE operator whose `res`/`jac`/`domains` fields are plain (not a
[`Bundle`](@ref) of a linear and a nonlinear half)
"""
struct PlainForm <: WeakFormStyle end

"""
    struct BundleForm <: WeakFormStyle end

Trait for a FE operator whose `res`/`jac`/`domains` fields are each a [`Bundle`](@ref)
of a linear and a nonlinear half
"""
struct BundleForm <: WeakFormStyle end

# utils 

"""
    get_polynomial_order(f::FESpace) -> Integer

Retrieves the polynomial order of `f`
"""
get_polynomial_order(f::SingleFieldFESpace) = get_polynomial_order(get_fe_basis(f))
get_polynomial_order(f::MultiFieldFESpace) = maximum(map(get_polynomial_order,f.spaces))

function get_polynomial_order(basis)
  cell_basis = get_data(basis)
  shapefun = testitem(cell_basis)
  _get_order(shapefun)
end

_get_order(ϕ) = @abstractmethod
_get_order(ϕ::Fields.LinearCombinationFieldVector) = get_order(ϕ.fields)

function _get_order(ϕ::Fields.BroadcastOpFieldArray)
  found = ()
  for ai in ϕ.args
    try
      oi = _get_order(ai)
      found = (found...,oi)
    catch
    end
  end

  if isempty(found)
    msg = "Unable to infer polynomial order from BroadcastOpFieldArray: _get_order failed for all arguments"
    @assert false msg
  elseif length(found) == 1
    return first(found)
  elseif all(==(first(found)),found)
    return first(found)
  else
    msg = "Inconsistent polynomial orders found in BroadcastOpFieldArray arguments: $(found)"
    @notimplemented msg
  end
end

"""
    get_polynomial_orders(fs::FESpace) -> NTuple{D,Int}

Retrieves the polynomial order of `fs` for every dimension `d = 1,...,D` of the underlying reference FE
"""
get_polynomial_orders(fs::SingleFieldFESpace) = get_polynomial_orders(get_fe_basis(fs))
get_polynomial_orders(fs::MultiFieldFESpace) = maximum.(map(get_polynomial_orders,fs.spaces))

function get_polynomial_orders(basis)
  cell_basis = get_data(basis)
  shapefun = testitem(cell_basis)
  get_orders(shapefun.fields)
end

"""
    function collect_cell_matrix_for_trian(
      trial::FESpace,
      test::FESpace,
      a::DomainContribution,
      strian::Triangulation
      ) -> Tuple{Vector{<:Any},Vector{<:Any},Vector{<:Any}}

Computes the cell-wise data needed to assemble a global sparse matrix for a given
input triangulation `strian`
"""
function collect_cell_matrix_for_trian(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation
  )

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  rows = get_cell_dof_ids(test,trian)
  cols = get_cell_dof_ids(trial,trian)
  Any[cell_mat_rc],Any[rows],Any[cols]
end

"""
    function collect_cell_vector_for_trian(
      test::FESpace,
      a::DomainContribution,
      strian::Triangulation
      ) -> Tuple{Vector{<:Any},Vector{<:Any}}

Computes the cell-wise data needed to assemble a global vector for a given
input triangulation `strian`
"""
function collect_cell_vector_for_trian(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation
  )

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)
  Any[cell_vec_r],Any[rows]
end