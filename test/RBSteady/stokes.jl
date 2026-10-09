module StokesEquation

using Gridap
using Gridap.MultiField
using Test
using DrWatson

using GridapROMs

function main(
  method=:pod,compression=:global,hypred_strategy=:deim;
  tol=1e-4,nparams=15,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),ncentroids=2
  )

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:deim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :deim

  println("Running test with $compression ($method, $hypred_strategy) strategy")

  pdomain = (1,10,-1,5,1,2)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (8,8)
  model = method==:ttsvd ? TProductDiscreteModel(domain,partition) : CartesianDiscreteModel(domain,partition)

  order = 2
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(μ) = x -> μ[1]*exp(-x[1])
  aμ(μ) = parameterise(a,μ)

  g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
  gμ(μ) = parameterise(g,μ)

  stiffness(μ,(u,p),(v,q)) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  res(μ,(u,p),(v,q)) = stiffness(μ,(u,p),(v,q))

  trian_res = (Ω,)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  trial_u = ParamTrialFESpace(test_u,gμ)
  trial_p = ParamTrialFESpace(test_p)
  test = MultiFieldFESpace([test_u,test_p])
  trial = MultiFieldFESpace([trial_u,trial_p])

  energy = BlockNorm((H1(),L2()))
  coupling = DivCoupling()

  if method == :pod
    state_reduction = SupremizerReduction(coupling,tol,energy;nparams,compression,ncentroids)
  elseif method == :ttsvd
    state_reduction = SupremizerReduction(coupling,fill(tol,3),energy;nparams,compression,ncentroids)
  end

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=10,sampling=:uniform)
  x̂ = solve(rbsolver,rbop,μon)
  x, = solution_snapshots(rbsolver,feop,μon)
  println(rom_performance(rbsolver,rbop,x,x̂))
end

for method in (:pod,:ttsvd), compression in (:local,:global), hypred_strategy in (:deim,:sopt,:rbf,:none,:affine)
  main(method,compression,hypred_strategy)
end

end

using Gridap, Gridap.Algebra
using GridapSolvers
using GridapSolvers.LinearSolvers
using LinearAlgebra

function LinearSolvers.get_solver_caches(
  solver::GMRESSolver,
  A::AbstractMatrix{<:Number}
  )
  m, Pl, Pr = solver.m, solver.Pl, solver.Pr

  V  = [allocate_in_domain(A) for i in 1:m+1]
  zr = !isnothing(Pr) ? allocate_in_domain(A) : nothing
  zl = allocate_in_domain(A)

  T = eltype(A)

  H = zeros(T,m+1,m) # Hessenberg matrix
  g = zeros(T,m+1)   # Residual vector
  c = zeros(T,m)     # Gibens rotation cosines
  s = zeros(T,m)     # Gibens rotation sines
  return (V,zr,zl,H,g,c,s)
end

function Gridap.Algebra.solve!(
  x::AbstractVector{<:Number},
  ns::LinearSolvers.GMRESNumericalSetup,
  b::AbstractVector{<:Number}
  )
  solver, A, Pl, Pr, caches = ns.solver, ns.mat, ns.Pl_ns, ns.Pr_ns, ns.caches
  V, zr, zl, H, g, c, s = caches
  m   = LinearSolvers.krylov_cache_length(ns)
  log = solver.log

  println("1")

  fill!(V[1],zero(eltype(V[1])))
  !isnothing(zr) && fill!(zr,zero(eltype(zr)))
  fill!(zl,zero(eltype(zl)))

  # Initial residual
  LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
  β    = norm(V[1])
  done = GridapSolvers.init!(log,β)

  println("2")

  while !done
    # Arnoldi process
    j = 1
    V[1] ./= β
    fill!(H,zero(eltype(H)))
    fill!(g,zero(eltype(g))); g[1] = β
    while !done && LinearSolvers.restart(solver,j)
      # Expand Krylov basis if needed
      if j > m  
        H, g, c, s = LinearSolvers.expand_krylov_caches!(ns)
        m = LinearSolvers.krylov_cache_length(ns)
      end

      println("3")

      # Arnoldi orthogonalization by Modified Gram-Schmidt
      fill!(V[j+1],zero(eltype(V[j+1])))
      LinearSolvers.krylov_mul!(V[j+1],A,V[j],Pr,Pl,zr,zl)
      for i in 1:j
        H[i,j] = dot(V[i],V[j+1])
        V[j+1] .-= H[i,j] .* V[i]
      end
      H[j+1,j] = norm(V[j+1])
      V[j+1] ./= H[j+1,j]

      println("4")

      # Update QR
      for i in 1:j-1
        γ = c[i]*H[i,j] + s[i]*H[i+1,j]
        H[i+1,j] = -conj(s[i])*H[i,j] + c[i]*H[i+1,j]
        H[i,j] = γ
      end

      # New Givens rotation, update QR and residual
      c[j], s[j], _ = LinearAlgebra.givensAlgorithm(H[j,j],H[j+1,j])
      H[j,j] = c[j]*H[j,j] + s[j]*H[j+1,j]; H[j+1,j] = zero(eltype(H))
      g[j+1] = -s[j]*g[j]; g[j] = c[j]*g[j]

      println("5")

      β  = abs(g[j+1])
      j += 1
      done = GridapSolvers.update!(log,β)
    end
    j = j-1

    # Solve least squares problem Hy = g by backward substitution
    for i in j:-1:1
      g[i] = (g[i] - dot(H[i,i+1:j],g[i+1:j])) / H[i,i]
    end

    # Update solution & residual
    if isnothing(Pr)
      for i in 1:j
        x .+= g[i] .* V[i]
      end
    else
      fill!(zl,zero(eltype(zl)))
      for i in 1:j
        zl .+= g[i] .* V[i]
      end
      solve!(zr,Pr,zl)
      x .+= zr
      println("6")
    end
    LinearSolvers.krylov_residual!(V[1],x,A,b,Pl,zl)
    β_actual = norm(V[1])
    println("actual residual = ", β_actual)
    println("estimated residual = ", β)
  end

  GridapSolvers.finalize!(log,β)
  return x
end

sol(x) = im*x[1] + x[2]
f(x)   = zero(ComplexF64)

domain = (0,1,0,1)
nc = (8,8)

model = CartesianDiscreteModel(domain,nc)
order  = 1
qorder = order*2 + 1
reffe  = ReferenceFE(lagrangian,Float64,order)
Vh     = TestFESpace(
  model,reffe;conformity=:H1,dirichlet_tags="boundary",vector_type=Vector{ComplexF64}
)
Uh     = TrialFESpace(Vh,sol)
u      = interpolate(sol,Uh)

Ω      = Triangulation(model)
dΩ     = Measure(Ω,qorder)
a(u,v) = ∫(∇(v)⋅∇(u))*dΩ
l(v)   = ∫(v⋅f)*dΩ
op = AffineFEOperator(a,l,Uh,Vh)

P = JacobiLinearSolver()

# GMRES with left and right preconditioner
solver = LinearSolvers.GMRESSolver(40;Pr=P,Pl=P,rtol=1.e-8,verbose=true)

A, b = get_matrix(op), get_vector(op);
ns = numerical_setup(symbolic_setup(solver,A),A)

x = allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b)

u  = interpolate(sol,Uh)
uh = FEFunction(Uh,x)
eh = uh - u
E  = sum(∫(eh*eh)*dΩ)
@test E < 1.e-6