"""
    abstract type ExtensionStyle end

Abstraction for different extension methods. Subtypes:
- [`ZeroExtension`](@ref)
- [`MassExtension`](@ref)
- [`HarmonicExtension`](@ref)
"""
abstract type ExtensionStyle end

"""
    struct ZeroExtension <: ExtensionStyle end

Extension by zero
"""
struct ZeroExtension <: ExtensionStyle end

"""
    struct MassExtension <: ExtensionStyle end

Extension by means of a mass operator
"""
struct MassExtension <: ExtensionStyle end

"""
    struct HarmonicExtension <: ExtensionStyle end

Extension by means of a discrete Laplace operator
"""
struct HarmonicExtension <: ExtensionStyle end

"""
    struct BlockExtension <: ExtensionStyle
      extension::Vector{<:ExtensionStyle}
    end

Extension style for multi-field variables
"""
struct BlockExtension <: ExtensionStyle
  extension::Vector{<:ExtensionStyle}
end

"""
    extend_solution(ext::ExtensionStyle,f::FESpace,u::AbstractVector) -> AbstractVector

Extension of the nodal values `u` associated with a FE function defined on `f`
according to the extension style `ext`
"""
function extend_solution(ext::ExtensionStyle,f::FESpace,u::AbstractVector)
  @abstractmethod
end

function extend_solution(ext::ZeroExtension,f::FESpace,u::AbstractVector)
  extend_free_values(f,u)
end

function extend_solution(ext::MassExtension,f::SingleFieldFESpace,u::AbstractVector)
  fin = get_space(f)
  uh_in = FEFunction(fin,u)
  uh_in_bg = ExtendedFEFunction(f,u)

  fout = get_out_space(f)
  uh_out = mass_extension(fout,uh_in_bg)

  uh_bg = uh_in ⊕ uh_out
  get_free_dof_values(uh_bg)
end

function extend_solution(ext::HarmonicExtension,f::SingleFieldFESpace,u::AbstractVector)
  fin = get_space(f)
  uh_in = FEFunction(fin,u)
  uh_in_bg = ExtendedFEFunction(f,u)

  fout = get_out_space(f)
  uh_out = harmonic_extension(fout,uh_in_bg)

  uh_bg = uh_in ⊕ uh_out
  get_free_dof_values(uh_bg)
end

function extend_solution(ext::BlockExtension,f::MultiFieldFESpace,u::Union{BlockVector,BlockParamVector})
  uh_bg = map(extend_solution,ext.extension,f.spaces,blocks(u))
  mortar(uh_bg)
end

function extend_solution(f::FESpace,u::AbstractVector)
  extend_solution(HarmonicExtension(),f,u)
end

"""
    extend_solution!(u_bg::AbstractVector,ext::ExtensionStyle,f::FESpace,u::AbstractVector)

In-place version of [`extend_solution`](@ref)
"""
function extend_solution!(u_bg::AbstractVector,ext::ZeroExtension,f::FESpace,u::AbstractVector)
  u_bg
end

function extend_solution!(u_bg::AbstractVector,ext::MassExtension,f::SingleFieldFESpace,u::AbstractVector)
  fin = get_space(f)
  uh_in_bg = ExtendedFEFunction(f,u)

  fout = get_out_space(f)
  uh_out = mass_extension(fout,uh_in_bg)

  gather_extended_free_values!(u_bg,f,get_cell_dof_values(uh_out))
end

function extend_solution!(u_bg::AbstractVector,ext::HarmonicExtension,f::SingleFieldFESpace,u::AbstractVector)
  fin = get_space(f)
  uh_in_bg = ExtendedFEFunction(f,u)

  fout = get_out_space(f)
  uh_out = harmonic_extension(fout,uh_in_bg)

  gather_extended_free_values!(u_bg,f,get_cell_dof_values(uh_out))
end

function extend_solution!(u_bg::AbstractVector,f::FESpace,u::AbstractVector)
  extend_solution!(u_bg,HarmonicExtension(),f,u)
end

struct ExtensionSolver <: NonlinearSolver
  solver::NonlinearSolver
  extension::ExtensionStyle
end

function ExtensionSolver(solver::NonlinearSolver)
  extension = HarmonicExtension()
  ExtensionSolver(solver,extension)
end

function Algebra.solve!(
  u_bg::AbstractVector,
  solver::ExtensionSolver,
  op::NonlinearOperator,
  cache
  )

  solve!(u_bg,solver.solver,op,cache)
end

function Algebra.solve(solver::ExtensionSolver,op::DomainOperator)
  u = solve(solver.solver,op)
  u_bg = extend_solution(solver.extension,get_trial(op),u)
  return u_bg
end

function Algebra.solve(solver::ExtensionSolver,op::ParamOperator,r::Realization)
  u,stats = solve(solver.solver,op,r)
  u_bg = extend_solution(solver.extension,get_trial(op)(r),u)
  return u_bg,stats
end

# transient

struct ExtensionODESolver <: ODESolver
  solver::ODESolver
  extension::ExtensionStyle
end

struct ExtensionODEParamSolution{E<:ExtensionStyle}
  extension::E
  odesol::ODEParamSolution
end

function ExtensionODEParamSolution(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealization,
  u0)

  odesol = ODEParamSolution(solver.solver,odeop,r,u0)
  ExtensionODEParamSolution(solver.extension,odesol)
end

function ParamODEs.ODEParamSolution(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealization,
  u0::V) where V

  ExtensionODEParamSolution(solver,odeop,r,u0)
end

function Base.collect(sol::ExtensionODEParamSolution)
  u,stats = collect(sol.odesol)
  u_bg = extend_solution(sol.extension,get_trial(sol.odesol.odeop)(sol.odesol.r),u)
  return u_bg,stats
end

function ParamODEs.initial_condition(sol::ExtensionODEParamSolution)
  u0 = initial_condition(sol.odesol)
  r0 = get_at_time(sol.odesol.r,:initial)
  extend_solution(sol.extension,get_trial(sol.odesol.odeop)(r0),u0)
end

# utils

"""
    mass_extension(fout::SingleFieldFESpace,uh_in_bg::FEFunction) -> FEFunction

Given a FEFunction `uh_in_bg` defined on an [`EmbeddedFESpace`](@ref) -- so that
nodal and cell values are available on a background FE space, in addition to the
FE space itself -- it returns the mass extension to the complementary FE space `fout`
"""
function mass_extension(fout::SingleFieldFESpace,uh_in_bg::FEFunction)
  degree = 2*get_polynomial_order(fout)
  Ωout = get_triangulation(fout)
  dΩout = Measure(Ωout,degree)

  m(u,v) = ∫(u⋅v)dΩout
  l(v) = (-1)*∫(uh_in_bg⋅v)dΩout
  assem = SparseMatrixAssembler(fout,fout)

  M = assemble_matrix(m,assem,fout,fout)
  b = assemble_vector(l,assem,fout)
  uout = solve(LUSolver(),M,b)

  FEFunction(fout,uout)
end

"""
    harmonic_extension(fout::SingleFieldFESpace,uh_in_bg::FEFunction) -> FEFunction

Given a FEFunction `uh_in_bg` defined on an [`EmbeddedFESpace`](@ref) -- so that
nodal and cell values are available on a background FE space, in addition to the
FE space itself -- it returns the harmonic extension to the complementary FE space `fout`
"""
function harmonic_extension(fout::SingleFieldFESpace,uh_in_bg::FEFunction)
  degree = 2*get_polynomial_order(fout)
  Ωout = get_triangulation(fout)
  dΩout = Measure(Ωout,degree)

  a(u,v) = ∫(∇(u)⊙∇(v))dΩout
  l(v) = (-1)*∫(∇(uh_in_bg)⊙∇(v))dΩout
  assem = SparseMatrixAssembler(fout,fout)

  A = assemble_matrix(a,assem,fout,fout)
  b = assemble_vector(l,assem,fout)
  uout = solve(LUSolver(),A,b)

  FEFunction(fout,uout)
end
