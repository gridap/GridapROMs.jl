abstract type ExtensionStyle end

struct ZeroExtension <: ExtensionStyle end
struct FunctionExtension <: ExtensionStyle end
struct HarmonicExtension <: ExtensionStyle end

struct BlockExtension <: ExtensionStyle
  extension::Vector{<:ExtensionStyle}
end

function extend_solution(ext::ExtensionStyle,f::FESpace,u::AbstractVector)
  @abstractmethod
end

function extend_solution(ext::ZeroExtension,f::FESpace,u::AbstractVector)
  extend_free_values(f,u)
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

function extend_solution!(u_bg::AbstractVector,ext::ZeroExtension,f::FESpace,u::AbstractVector)
  u_bg
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

function Algebra.solve(solver::ExtensionSolver,op::NonlinearOperator)
  u = solve(solver.solver,op)
  u_bg = extend_solution(solver.extension,get_trial(op),u)
  return u_bg
end

function Algebra.solve(solver::ExtensionSolver,op::ExtensionParamOperator,r::Realization)
  u,stats = solve(solver.solver,op.op,r)
  u_bg = extend_solution(solver.extension,get_trial(op)(r),u)
  return u_bg,stats
end

# transient

remove_extension(s::ODESolver) = @abstractmethod
remove_extension(s::ThetaMethod) = ThetaMethod(s.sysslvr.solver,s.dt,s.θ)
get_extension(s::ODESolver) = @abstractmethod
get_extension(s::ThetaMethod) = s.sysslvr.extension

struct ExtensionODEParamSolution{E<:ExtensionStyle}
  extension::E
  odesol::ODEParamSolution
end

function ExtensionODEParamSolution(
  solver::ODESolver,
  odeop::ODEExtensionParamOperator,
  r::TransientRealization,
  u0)

  extension = get_extension(solver)
  odesol = ODEParamSolution(remove_extension(solver),odeop.op,r,u0)
  ExtensionODEParamSolution(extension,odesol)
end

function Base.collect(sol::ExtensionODEParamSolution)
  u,stats = collect(sol.odesol)
  u_bg = extend_solution(sol.extension,get_trial(sol.odesol.odeop)(sol.odesol.r),u)
  return u_bg,stats
end

function ParamODEs.ODEParamSolution(
  solver::ODESolver,
  odeop::ODEExtensionParamOperator,
  r::TransientRealization,
  u0::V) where V

  ExtensionODEParamSolution(solver,odeop,r,u0)
end

function ParamODEs.initial_condition(sol::ExtensionODEParamSolution)
  u0 = initial_condition(sol.odesol)
  r0 = get_at_time(sol.odesol.r,:initial)
  extend_solution(sol.extension,get_trial(sol.odesol.odeop)(r0),u0)
end

# utils

function harmonic_extension(fout::SingleFieldFESpace,uh_in_bg::FEFunction)
  degree = 2*get_polynomial_order(fout)
  Ωout = get_triangulation(fout)
  dΩout = Measure(Ωout,degree)

  a(u,v) = ∫(∇(u)⊙∇(v))dΩout
  l(v) = ∫(∇(uh_in_bg)⊙∇(v))dΩout
  assem = SparseMatrixAssembler(fout,fout)

  A = assemble_matrix(a,assem,fout,fout)
  b = assemble_vector(l,assem,fout)
  uout = solve(LUSolver(),A,b)

  FEFunction(fout,uout)
end
