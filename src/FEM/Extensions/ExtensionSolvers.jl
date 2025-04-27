abstract type ExtensionStyle end

struct ZeroExtension <: ExtensionStyle end
struct FunctionExtension <: ExtensionStyle end
struct HarmonicExtension <: ExtensionStyle end

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
