# module TimeCombinationsTest

using Test
using LinearAlgebra
using SparseArrays

using Gridap
using Gridap.FESpaces
using Gridap.Algebra
using Gridap.ODEs

using GridapROMs
using GridapROMs.Utils
using GridapROMs.ParamDataStructures
using GridapROMs.ParamODEs
using GridapROMs.RBSteady
using GridapROMs.RBTransient

const get_combination = ParamODEs.get_combination

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# ConsecutiveParamArray layout: columns ordered as
#   (p1,t1),(p2,t1),...,(pNp,t1),(p1,t2),...,(pNp,tNt)
# slow_index(ipt,np) = time index,fast_index(ipt,np) = param index
cpa(data::Matrix) = ConsecutiveParamArray(data)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  get_combination — ThetaMethod
# ─────────────────────────────────────────────────────────────────────────────

@testset "get_combination ThetaMethod θ=0.5 displacement" begin
  dt = 0.1; θ = 0.5
  # 1 dof,2 params,3 time steps
  # column layout: (p1t1,p2t1,p1t2,p2t2,p1t3,p2t3)
  u_data  = Float64[1 2 3 4 5 6]   # (1×6)
  u0_data = Float64[0 0.5]          # (1×2)  initial condition at t=0

  u  = cpa(u_data)
  u0 = cpa(u0_data)

  v0  = cpa(similar(u0_data))         # ThetaMethod ignores v0; provide anyway
  c   = ThetaMethodCombination(dt,θ)
  usx = get_combination(c,u,(u0,v0))  # NTuple{2,...} → returns 2-tuple

  @test length(usx) == 2
  uθ = get_all_data(usx[1])

  # t=1: uθ = θ·u[n] + (1-θ)·u0
  @test uθ[1,1] ≈ θ*1 + (1-θ)*0     # p1
  @test uθ[1,2] ≈ θ*2 + (1-θ)*0.5   # p2

  # t=2: uθ = θ·u[t2] + (1-θ)·u[t1]
  @test uθ[1,3] ≈ θ*3 + (1-θ)*1     # p1
  @test uθ[1,4] ≈ θ*4 + (1-θ)*2     # p2

  # t=3
  @test uθ[1,5] ≈ θ*5 + (1-θ)*3     # p1
  @test uθ[1,6] ≈ θ*6 + (1-θ)*4     # p2
end

@testset "get_combination ThetaMethod θ=1 implicit Euler" begin
  dt = 0.1; θ = 1.0
  u_data  = Float64[1 2 3 4 5 6]
  u0_data = Float64[0 0.5]

  u  = cpa(u_data);  u0 = cpa(u0_data)
  c  = ThetaMethodCombination(dt,θ)
  uθ = get_all_data(get_combination(c,u,(u0,similar(u0)))[1])

  # θ=1: uθ = u (current time step only)
  @test uθ[1,1] ≈ 1.0   # t1,p1
  @test uθ[1,2] ≈ 2.0   # t1,p2
  @test uθ[1,3] ≈ 3.0   # t2,p1
  @test uθ[1,5] ≈ 5.0   # t3,p1
end

@testset "get_combination ThetaMethod θ=0 explicit Euler" begin
  dt = 0.1; θ = 0.0
  u_data  = Float64[1 2 3 4 5 6]
  u0_data = Float64[0 0.5]

  u  = cpa(u_data);  u0 = cpa(u0_data)
  c  = ThetaMethodCombination(dt,θ)
  uθ = get_all_data(get_combination(c,u,(u0,similar(u0)))[1])

  # θ=0: uθ = previous time step
  @test uθ[1,1] ≈ 0.0   # t1: u0[p1]
  @test uθ[1,2] ≈ 0.5   # t1: u0[p2]
  @test uθ[1,3] ≈ 1.0   # t2: u[t1,p1]
  @test uθ[1,4] ≈ 2.0   # t2: u[t1,p2]
  @test uθ[1,5] ≈ 3.0   # t3: u[t2,p1]
end

@testset "get_combination ThetaMethod velocity (CombinationOrder{2})" begin
  # CombinationOrder{2} computes (u_n - u_{n-1})/dt.
  dt = 0.1; θ = 0.5
  np = 1; nt = 3; ndof = 1
  u_data  = reshape(Float64[1.0,3.0,6.0],1,3)  # single param
  u0_data = reshape(Float64[0.0],   1,1)

  u  = cpa(u_data);  u0 = cpa(u0_data)

  # Velocity-order combination: coefficients (1/dt,-1/dt)
  # ThetaMethod _combination! requires NTuple{2}; v0 (2nd entry) is unused here.
  c_vel = CombinationOrder{2}(ThetaMethodCombination(dt,θ))
  v = similar(u)
  ParamODEs._combination!(v,c_vel,u,(u0,similar(u0)))
  vdata = get_all_data(v)

  @test vdata[1,1] ≈ (1.0 - 0.0) / dt   # (u[t1] - u0) / dt
  @test vdata[1,2] ≈ (3.0 - 1.0) / dt   # (u[t2] - u[t1]) / dt
  @test vdata[1,3] ≈ (6.0 - 3.0) / dt   # (u[t3] - u[t2]) / dt
end

# ─────────────────────────────────────────────────────────────────────────────
# 2.  get_combination — GeneralizedAlpha1
# ─────────────────────────────────────────────────────────────────────────────
# Use ρ∞=1 → αf=αm=γ=0.5,giving c=0 and clean finite-difference formulas.

@testset "get_combination GenAlpha1 (ρ∞=1) displacement" begin
  dt = 0.1
  αf = 0.5; αm = 0.5; γ = 0.5   # ρ∞ = 1
  # 1 dof,2 params,3 time steps
  u_data  = Float64[1 2 3 4 5 6]
  u0_data = Float64[0 0.5]
  v0_data = Float64[0 0]          # initial velocity (used by GenAlpha1 order-2)

  u  = cpa(u_data)
  u0 = cpa(u0_data)
  v0 = cpa(v0_data)

  c   = GenAlpha1Combination(dt,αf,αm,γ)
  usx = get_combination(c,u,(u0,v0))   # NTuple{2,...} → returns 2-tuple

  @test length(usx) == 2

  # Displacement: uαf = αf·u[n] + (1-αf)·u[n-1]
  uαf = get_all_data(usx[1])
  @test uαf[1,1] ≈ 0.5*1 + 0.5*0     # t1,p1
  @test uαf[1,2] ≈ 0.5*2 + 0.5*0.5   # t1,p2
  @test uαf[1,3] ≈ 0.5*3 + 0.5*1     # t2,p1
  @test uαf[1,4] ≈ 0.5*4 + 0.5*2     # t2,p2
  @test uαf[1,5] ≈ 0.5*5 + 0.5*3     # t3,p1
end

@testset "get_combination GenAlpha1 (ρ∞=1) velocity (αm=γ=0.5 → FD)" begin
  # With αm=0.5,γ=0.5 and c = a*(1-αm+b*αm) = 0:
  #   vαm[t=1] = (αm/γ)*(u[1]-u0)/dt = (u[1]-u0)/dt
  #   vαm[t>1] = (u[t]-u[t-1])/dt  (pure backward finite difference)
  dt = 0.1
  αf = 0.5; αm = 0.5; γ = 0.5
  np = 1; nt = 3; ndof = 1
  u_data  = reshape(Float64[1.0,3.0,6.0],1,3)
  u0_data = reshape(Float64[0.0],    1,1)
  v0_data = reshape(Float64[0.0],    1,1)

  u  = cpa(u_data);  u0 = cpa(u0_data);  v0 = cpa(v0_data)
  c  = GenAlpha1Combination(dt,αf,αm,γ)
  vαm = get_all_data(get_combination(c,u,(u0,v0))[2])

  # a = 1/(γ*dt) = 20,b = -1,c_coeff = 0 → vαm = a*αm*(u[t]-u[t-1]) = 10*(u[t]-u[t-1])
  @test vαm[1,1] ≈ (1.0 - 0.0) / dt   # (u[t1]-u0)/dt
  @test vαm[1,2] ≈ (3.0 - 1.0) / dt   # (u[t2]-u[t1])/dt
  @test vαm[1,3] ≈ (6.0 - 3.0) / dt   # (u[t3]-u[t2])/dt
end

@testset "get_combination GenAlpha1 (ρ∞=0.5) velocity — multi-step sum" begin
  # ρ∞=0.5 → αf≈2/3,αm≈5/6,γ≈2/3.  c is non-zero,so history matters.
  fesolver_ref = GeneralizedAlpha1(LUSolver(),0.1,0.5)
  dt = fesolver_ref.dt
  αf = fesolver_ref.αf; αm = fesolver_ref.αm; γ = fesolver_ref.γ

  a = 1/(γ*dt)
  b = 1 - 1/γ
  c_coeff = a * (1 - αm + b*αm)

  np = 1; nt = 3; ndof = 1
  u_raw   = [2.0,5.0,9.0]
  u0_raw  = 0.0
  v0_raw  = 1.0   # non-zero initial velocity to test c/a*v0 term

  u_data  = reshape(u_raw,1,nt)
  u0_data = reshape([u0_raw],1,1)
  v0_data = reshape([v0_raw],1,1)
  u  = cpa(u_data);  u0 = cpa(u0_data);  v0 = cpa(v0_data)

  c  = GenAlpha1Combination(dt,αf,αm,γ)
  vαm = get_all_data(get_combination(c,u,(u0,v0))[2])

  # Step 1 (it=1): vαm = a*αm*u[1] - a*αm*u0 + (c_coeff/a)*v0
  expected_t1 = a*αm*u_raw[1] - a*αm*u0_raw + (c_coeff/a)*v0_raw
  @test vαm[1,1] ≈ expected_t1 atol=1e-10

  # Step 2 (it=2): vαm = a*αm*u[2] + (c_coeff-a*αm)*u[1] - c_coeff*u0 + (c_coeff/a)*b*v0
  # (no backward sum since it-2*np = 2-2*1 ≤ 0)
  expected_t2 = a*αm*u_raw[2] + (c_coeff - a*αm)*u_raw[1] - c_coeff*u0_raw + (c_coeff/a)*b*v0_raw
  @test vαm[1,2] ≈ expected_t2 atol=1e-10

  # Step 3 (it=3): vαm = a*αm*u[3] + (c_coeff-a*αm)*u[2] + c_coeff*(b^0-b^{-1})*u[1] - c_coeff*b*u0 + (c_coeff/a)*b^2*v0
  # backward loop: j=1,ipt_back = 3-2*1 = 1 → c_coeff*(b^0 - b^{-1}) impossible since b^{-1} undefined for b=-0.5
  # Actually get_coefficients generates: η[3] = c_coeff*(b^{3-2} - b^{3-3}) = c*(b^1-b^0)
  η3 = c_coeff*(b^1 - b^0)
  expected_t3 = a*αm*u_raw[3] + (c_coeff - a*αm)*u_raw[2] + η3*u_raw[1] - c_coeff*b*u0_raw + (c_coeff/a)*b^2*v0_raw
  @test vαm[1,3] ≈ expected_t3 atol=1e-10
end

# ─────────────────────────────────────────────────────────────────────────────
# 2b. get_combination — GeneralizedAlpha2
# ─────────────────────────────────────────────────────────────────────────────
# Use ρ∞=1 → αf=αm=γ=0.5,β=0.25 (average-acceleration Newmark limit).

@testset "get_combination GenAlpha2 (ρ∞=1) displacement" begin
  dt = 0.1
  αf = 0.5; αm = 0.5; γ = 0.5; β = 0.25
  u_data  = Float64[1 2 3 4 5 6]
  u0_data = Float64[0 0.5]
  v0_data = Float64[0 0]
  a0_data = Float64[0 0]

  u  = cpa(u_data)
  u0 = cpa(u0_data)
  v0 = cpa(v0_data)
  a0 = cpa(a0_data)

  c = GenAlpha2Combination(dt,αf,αm,γ,β)
  usx = get_combination(c,u,(u0,v0,a0))

  @test length(usx) == 3

  # Displacement: uαf = (1-αf)·u[n] + αf·u[n-1]
  uαf = get_all_data(usx[1])
  @test uαf[1,1] ≈ 0.5*1 + 0.5*0
  @test uαf[1,2] ≈ 0.5*2 + 0.5*0.5
  @test uαf[1,3] ≈ 0.5*3 + 0.5*1
  @test uαf[1,4] ≈ 0.5*4 + 0.5*2
  @test uαf[1,5] ≈ 0.5*5 + 0.5*3
end

@testset "get_combination GenAlpha2 (ρ∞=1) velocity (zero ICs → FD)" begin
  # With αf=αm=γ=0.5,β=0.25 and zero initial velocity/acceleration:
  #   vαf[t=1] = (u[1]-u0)/dt
  #   vαf[t>1] = (u[t]-u[t-1])/dt
  dt = 0.1
  αf = 0.5; αm = 0.5; γ = 0.5; β = 0.25
  u_data  = reshape(Float64[1.0,3.0,6.0],1,3)
  u0_data = reshape(Float64[0.0],1,1)
  v0_data = reshape(Float64[0.0],1,1)
  a0_data = reshape(Float64[0.0],1,1)

  u  = cpa(u_data)
  u0 = cpa(u0_data)
  v0 = cpa(v0_data)
  a0 = cpa(a0_data)
  c  = GenAlpha2Combination(dt,αf,αm,γ,β)
  vαf = get_all_data(get_combination(c,u,(u0,v0,a0))[2])

  @test vαf[1,1] ≈ (1.0 - 0.0) / dt
  @test vαf[1,2] ≈ (3.0 - 1.0) / dt
  @test vαf[1,3] ≈ (6.0 - 3.0) / dt
end

@testset "get_combination GenAlpha2 (ρ∞=0.5) velocity and acceleration — multi-step sums" begin
  fesolver_ref = GeneralizedAlpha2(LUSolver(),0.1,0.5)
  dt = fesolver_ref.dt
  αf = fesolver_ref.αf
  αm = fesolver_ref.αm
  γ = fesolver_ref.γ
  β = fesolver_ref.β

  u_raw  = [2.0,5.0,9.0]
  u0_raw = 0.5
  v0_raw = -1.0
  a0_raw = 0.25

  u_data  = reshape(u_raw,1,length(u_raw))
  u0_data = reshape([u0_raw],1,1)
  v0_data = reshape([v0_raw],1,1)
  a0_data = reshape([a0_raw],1,1)

  u  = cpa(u_data)
  u0 = cpa(u0_data)
  v0 = cpa(v0_data)
  a0 = cpa(a0_data)
  c  = GenAlpha2Combination(dt,αf,αm,γ,β)

  ηv = get_coefficients(CombinationOrder{2}(c),length(u_raw))
  ηa = get_coefficients(CombinationOrder{3}(c),length(u_raw))
  vαf = get_all_data(get_combination(c,u,(u0,v0,a0))[2])
  aαm = get_all_data(get_combination(c,u,(u0,v0,a0))[3])

  a = γ / (dt * β)
  b = -a
  cP = 1 - γ / β
  d = dt * (1 - γ / (2 * β))
  e = 1 / (dt^2 * β)
  f = -e
  g = -1 / (dt * β)
  h = 1 - 1 / (2 * β)
  P = [cP d; g h]

  an(n) = ([1.0 0.0] * P^n * [1.0,0.0])[1]
  bn(n) = ([1.0 0.0] * P^n * [0.0,1.0])[1]
  cn(n) = ([0.0 1.0] * P^n * [1.0,0.0])[1]
  dn(n) = ([0.0 1.0] * P^n * [0.0,1.0])[1]

  βnj(n,j) = ([1.0 0.0] * P^(n-j) * [b,f])[1]
  δnj(n,j) = ([0.0 1.0] * P^(n-j) * [b,f])[1]

  cαn(n) = (1 - αf) * βnj(n,0) + αf * βnj(n-1,0)
  dαn(n) = (1 - αf) * an(n+1) + αf * an(n)
  eαn(n) = (1 - αf) * bn(n+1) + αf * bn(n)

  iαn(n) = (1 - αm) * δnj(n,0) + αm * δnj(n-1,0)
  jαn(n) = (1 - αm) * cn(n+1) + αm * cn(n)
  kαn(n) = (1 - αm) * dn(n+1) + αm * dn(n)

  expected_v1 = ηv[1]*u_raw[1] + (1-αf)*βnj(0,0)*u0_raw + ((1-αf)*an(1) + αf)*v0_raw + (1-αf)*bn(1)*a0_raw
  expected_v2 = ηv[1]*u_raw[2] + ηv[2]*u_raw[1] + cαn(1)*u0_raw + dαn(1)*v0_raw + eαn(1)*a0_raw
  expected_v3 = ηv[1]*u_raw[3] + ηv[2]*u_raw[2] + ηv[3]*u_raw[1] + cαn(2)*u0_raw + dαn(2)*v0_raw + eαn(2)*a0_raw

  expected_a1 = ηa[1]*u_raw[1] + (1-αf)*δnj(0,0)*u0_raw + (1-αm)*cn(1)*v0_raw + ((1-αm)*dn(1) + αm)*a0_raw
  expected_a2 = ηa[1]*u_raw[2] + ηa[2]*u_raw[1] + iαn(1)*u0_raw + jαn(1)*v0_raw + kαn(1)*a0_raw
  expected_a3 = ηa[1]*u_raw[3] + ηa[2]*u_raw[2] + ηa[3]*u_raw[1] + iαn(2)*u0_raw + jαn(2)*v0_raw + kαn(2)*a0_raw

  @test vαf[1,1] ≈ expected_v1 atol=1e-10
  @test vαf[1,2] ≈ expected_v2 atol=1e-10
  @test vαf[1,3] ≈ expected_v3 atol=1e-10

  @test aαm[1,1] ≈ expected_a1 atol=1e-10
  @test aαm[1,2] ≈ expected_a2 atol=1e-10
  @test aαm[1,3] ≈ expected_a3 atol=1e-10
end

@testset "get_combination consistent with CombinationOrder accessors" begin
  # Calling get_combination with the outer TimeCombination returns the same
  # as calling _combination! on each CombinationOrder{i} individually.
  dt = 0.1; θ = 0.5
  u_data  = rand(3,6)   # 3 dofs,2 params,3 time steps
  u0_data = rand(3,2)

  u  = cpa(u_data);  u0 = cpa(u0_data)
  c  = ThetaMethodCombination(dt,θ)
  usx = get_combination(c,u,(u0,similar(u0)))

  # Also test via CombinationOrder{1} directly
  c1 = CombinationOrder{1}(c)
  uθ_direct = similar(u)
  ParamODEs._combination!(uθ_direct,c1,u,(u0,similar(u0)))

  @test get_all_data(usx[1]) ≈ get_all_data(uθ_direct)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Space-time Galerkin projection (GalerkinProjections.jl)
# ─────────────────────────────────────────────────────────────────────────────
# galerkin_projection(Φl::Nt×nl,Φ::Nt×n,Φr::Nt×nr,combine::CombinationOrder)
# computes the (nl,n,nr) ROM tensor with time-combination weights.

@testset "galerkin_projection ThetaMethod θ=1 (implicit Euler)" begin
  # With θ=1,coefficients = (1,0).  Only γ=1 term contributes.
  # Result = Σ_{α=1}^{Nt-1}  Φl[α,:]' * Φ[α,:] * Φr[α,:]'  (outer sum)
  Nt = 8; nl = 3; n = 4; nr = 5
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  dt = 0.1; θ = 1.0
  c1 = CombinationOrder{1}(ThetaMethodCombination(dt,θ))
  proj = galerkin_projection(Φl,Φ,Φr,c1)

  @test size(proj) == (nl,n,nr)

  # Manually compute: γ=1 term,α=1..Nt-1 (α+1 ≤ Nt → α ≤ Nt-1)
  proj_ref = zeros(nl,n,nr)
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt
      proj_ref[i,k,j] += 1.0 * Φl[α,i] * Φ[α,k] * Φr[α,j]
    end
  end
  @test proj ≈ proj_ref atol=1e-12
end

@testset "galerkin_projection ThetaMethod θ=0 (explicit Euler)" begin
  # With θ=0,coefficients = (0,1).  Only γ=2 term contributes.
  # Result = Σ_{α=1}^{Nt-2}  Φl[α+1,:]' * Φ[α+1,:] * Φr[α,:]'
  Nt = 8; nl = 3; n = 4; nr = 5
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  dt = 0.1; θ = 0.0
  c1 = CombinationOrder{1}(ThetaMethodCombination(dt,θ))
  proj = galerkin_projection(Φl,Φ,Φr,c1)

  proj_ref = zeros(nl,n,nr)
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt-1          # α+2 ≤ Nt
      proj_ref[i,k,j] += 1.0 * Φl[α+1,i] * Φ[α+1,k] * Φr[α,j]
    end
  end
  @test proj ≈ proj_ref atol=1e-12
end

@testset "galerkin_projection ThetaMethod θ=0.5 (Crank-Nicolson)" begin
  Nt = 6; nl = 2; n = 3; nr = 4
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  dt = 0.1; θ = 0.5
  c1 = CombinationOrder{1}(ThetaMethodCombination(dt,θ))
  proj = galerkin_projection(Φl,Φ,Φr,c1)

  proj_ref = zeros(nl,n,nr)
  # γ=1: coefficient θ=0.5
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt
      proj_ref[i,k,j] += 0.5 * Φl[α,i] * Φ[α,k] * Φr[α,j]
    end
  end
  # γ=2: coefficient 1-θ=0.5
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt-1
      proj_ref[i,k,j] += 0.5 * Φl[α+1,i] * Φ[α+1,k] * Φr[α,j]
    end
  end
  @test proj ≈ proj_ref atol=1e-12
end

@testset "galerkin_projection ThetaMethod θ=0.5 (Crank-Nicolson)" begin
  Nt = 6; nl = 2; n = 3; nr = 4
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  dt = 0.1; θ = 0.5
  dt_inv = 10
  c1 = CombinationOrder{2}(ThetaMethodCombination(dt,θ))
  proj = galerkin_projection(Φl,Φ,Φr,c1)

  proj_ref = zeros(nl,n,nr)
  # γ=1: coefficient θ=0.5
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt
      proj_ref[i,k,j] += dt_inv * Φl[α,i] * Φ[α,k] * Φr[α,j]
    end
  end
  # γ=2: coefficient 1-θ=0.5
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt-1
      proj_ref[i,k,j] -= dt_inv * Φl[α+1,i] * Φ[α+1,k] * Φr[α,j]
    end
  end
  @test proj ≈ proj_ref atol=1e-12
end

@testset "galerkin_projection GenAlpha1 (ρ∞=1) displacement" begin
  # αf=0.5,γ=0.5 → get_coefficients(CombinationOrder{1},...) = (0.5,0.5)
  # Same as θ=0.5 Crank-Nicolson displacement combination
  Nt = 6; nl = 2; n = 3; nr = 4
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  αf = 0.5; αm = 0.5; γ = 0.5; dt = 0.1
  c1_alpha = CombinationOrder{1}(GenAlpha1Combination(dt,αf,αm,γ))
  c1_theta = CombinationOrder{1}(ThetaMethodCombination(dt,αf))  # αf plays role of θ here

  proj_alpha = galerkin_projection(Φl,Φ,Φr,c1_alpha)
  proj_theta = galerkin_projection(Φl,Φ,Φr,c1_theta)

  @test proj_alpha ≈ proj_theta atol=1e-12
end

@testset "galerkin_projection GenAlpha1 (ρ∞=0.5) velocity — multi-step sum" begin
  Nt = 4; nl = 2; n = 4; nr = 4
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  fesolver_ref = GeneralizedAlpha1(LUSolver(),0.1,0.5)
  dt = fesolver_ref.dt
  αf = fesolver_ref.αf; αm = fesolver_ref.αm; γ = fesolver_ref.γ

  a = 1/(γ*dt)
  b = 1 - 1/γ
  c_coeff = a * (1 - αm + b*αm)

  c1_alpha = CombinationOrder{1}(GenAlpha1Combination(dt,αf,αm,γ))
  c1_theta = CombinationOrder{1}(ThetaMethodCombination(dt,αf))  # αf plays role of θ here
  c2_alpha = CombinationOrder{2}(GenAlpha1Combination(dt,αf,αm,γ))
  θ = get_coefficients(c2_alpha,Nt)

  proj1_alpha = galerkin_projection(Φl,Φ,Φr,c1_alpha)
  proj_theta = galerkin_projection(Φl,Φ,Φr,c1_theta)
  proj2_alpha = galerkin_projection(Φl,Φ,Φr,c2_alpha)

  @test proj1_alpha ≈ proj_theta atol=1e-12

  _proj2_alpha = zeros(nl,n,nr)
  @inbounds for i = 1:nl,k = 1:n,j = 1:nr
    for α = 1:Nt
      idx = α + 1 - 1
      _proj2_alpha[i,k,j] += θ[1] * Φl[idx,i] * Φ[idx,k] * Φr[α,j]
    end
    for α = 1:Nt-1
      idx = α + 2 - 1
      _proj2_alpha[i,k,j] += θ[2] * Φl[idx,i] * Φ[idx,k] * Φr[α,j]
    end
    for α = 1:Nt-2
      idx = α + 3 - 1
      _proj2_alpha[i,k,j] += θ[3] * Φl[idx,i] * Φ[idx,k] * Φr[α,j]
    end
    for α = 1:Nt-3
      idx = α + 4 - 1
      _proj2_alpha[i,k,j] += θ[4] * Φl[idx,i] * Φ[idx,k] * Φr[α,j]
    end
  end

  @test proj2_alpha ≈ _proj2_alpha atol=1e-12
end

@testset "galerkin_projection GenAlpha2 (ρ∞=1) displacement" begin
  Nt = 6; nl = 2; n = 3; nr = 4
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  αf = 0.5; αm = 0.5; γ = 0.5; β = 0.25; dt = 0.1
  c1_alpha2 = CombinationOrder{1}(GenAlpha2Combination(dt,αf,αm,γ,β))
  c1_theta  = CombinationOrder{1}(ThetaMethodCombination(dt,1-αf))

  proj_alpha2 = galerkin_projection(Φl,Φ,Φr,c1_alpha2)
  proj_theta  = galerkin_projection(Φl,Φ,Φr,c1_theta)

  @test proj_alpha2 ≈ proj_theta atol=1e-12
end

@testset "galerkin_projection GenAlpha2 (ρ∞=0.5) velocity and acceleration — multi-step sums" begin
  Nt = 6; nl = 2; n = 4; nr = 4
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  fesolver_ref = GeneralizedAlpha2(LUSolver(),0.1,0.5)
  dt = fesolver_ref.dt
  αf = fesolver_ref.αf
  αm = fesolver_ref.αm
  γ = fesolver_ref.γ
  β = fesolver_ref.β

  c2_alpha2 = CombinationOrder{2}(GenAlpha2Combination(dt,αf,αm,γ,β))
  c3_alpha2 = CombinationOrder{3}(GenAlpha2Combination(dt,αf,αm,γ,β))
  θv = get_coefficients(c2_alpha2,Nt)
  θa = get_coefficients(c3_alpha2,Nt)

  proj2_alpha2 = galerkin_projection(Φl,Φ,Φr,c2_alpha2)
  proj3_alpha2 = galerkin_projection(Φl,Φ,Φr,c3_alpha2)

  proj2_ref = zeros(nl,n,nr)
  proj3_ref = zeros(nl,n,nr)
  @inbounds for i = 1:nl,k = 1:n,j = 1:nr
    for shift = eachindex(θv)
      for α = 1:(Nt - shift + 1)
        idx = α + shift - 1
        proj2_ref[i,k,j] += θv[shift] * Φl[idx,i] * Φ[idx,k] * Φr[α,j]
        proj3_ref[i,k,j] += θa[shift] * Φl[idx,i] * Φ[idx,k] * Φr[α,j]
      end
    end
  end

  @test proj2_alpha2 ≈ proj2_ref atol=1e-12
  @test proj3_alpha2 ≈ proj3_ref atol=1e-12
end

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Heat equation — SpaceTime.jl residual and jacobian
# ─────────────────────────────────────────────────────────────────────────────
# We build a minimal heat equation with a known exact solution,run
# solution_snapshots,then verify that:
#   (a) residual(solver,odeop,r,s) returns the correct number of
#       residual vectors (one per time step × param sample).
#   (b) jacobian(solver,odeop,r,s) returns sparse-matrix snapshots of
#       the correct size and that are symmetric positive definite.
#   (c) For the linear problem the residual at the discrete solution is
#       well below a loose tolerance (consistent residual check).

function _heat_eq_setup(;nparams=3)
  domain = (0,1,0,1)
  partition = (6,6)
  model = CartesianDiscreteModel(domain,partition)

  order = 1
  degree = 2*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  # Constant-coefficient heat equation: ∂_t u - μ Δu = 1 on (0,1)²,u=0 on ∂Ω
  a(μ,t) = x -> μ[1]
  aμt(μ,t) = parameterise(a,μ,t)
  f(μ,t) = x -> 1.0
  fμt(μ,t) = parameterise(f,μ,t)
  g(μ,t) = x -> x[1] * (1 + t)
  gμt(μ,t) = parameterise(g,μ,t)
  u0(μ)   = x -> g(μ,0.0)(x)
  u0μ(μ)  = parameterise(u0,μ)

  stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t) * ∇(v) ⋅ ∇(u))dΩ
  mass(_,_,uₜ,v,dΩ)     = ∫(v * uₜ)dΩ
  rhs(μ,t,v,dΩ)          = ∫(fμt(μ,t) * v)dΩ
  res(μ,t,u,v,dΩ)       = mass(μ,t,∂ₚt(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ)

  trian_res       = (Ω,)
  trian_stiffness = (Ω,)
  trian_mass      = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  reffe = ReferenceFE(lagrangian,Float64,order)
  test  = TestFESpace(Ω,reffe; conformity=:H1,dirichlet_tags="boundary")
  trial = TransientTrialParamFESpace(test,gμt)

  dt = 0.05; t0 = 0.0; tf = 4*dt
  tdomain = t0:dt:tf
  pdomain = (1,10)         # μ ∈ [1,10]  (1-parameter family)
  ptspace = TransientParamSpace(pdomain,tdomain)

  feop  = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  r = realisation(feop; nparams)
  return feop,r,uh0μ
end

function _wave_eq_setup(;nparams=3)
  domain = (0,1,0,1)
  partition = (6,6)
  model = CartesianDiscreteModel(domain,partition)

  order = 1
  degree = 2*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  # A weird wave equation: ∂^2_t u + ∂_t u + μ Δu = 0 on (0,1)²,u=0,∂_t u = 0 on ∂Ω
  a(μ,t) = x -> μ[1]
  aμt(μ,t) = parameterise(a,μ,t)
  g(μ,t) = x -> x[1] * (1 + t)^2
  gμt(μ,t) = parameterise(g,μ,t)
  # u(x,t) = x[1]*(1+t)²  ⟹  u(x,0) = x[1],∂_t u(x,0) = 2 x[1]
  u0(μ)   = x -> x[1]
  u0μ(μ)  = parameterise(u0,μ)
  v0(μ)   = x -> 2 * x[1]
  v0μ(μ)  = parameterise(v0,μ)

  stiffness(μ,t,u,v,dΩ)  = ∫(aμt(μ,t) * ∇(v) ⋅ ∇(u))dΩ
  damping(_,_,uₜ,v,dΩ)   = ∫(v * uₜ)dΩ
  mass(_,_,uₜₜ,v,dΩ)     = ∫(v * uₜₜ)dΩ
  res(μ,t,u,v,dΩ)        = mass(μ,t,∂ₚtt(u),v,dΩ) + damping(μ,t,∂ₚt(u),v,dΩ) + stiffness(μ,t,u,v,dΩ)

  trian_res       = (Ω,)
  trian_stiffness = (Ω,)
  trian_damping   = (Ω,)
  trian_mass      = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_damping,trian_mass))

  reffe = ReferenceFE(lagrangian,Float64,order)
  test  = TestFESpace(Ω,reffe; conformity=:H1,dirichlet_tags="boundary")
  trial = TransientTrialParamFESpace(test,gμt)

  dt = 0.05; t0 = 0.0; tf = 4*dt
  tdomain = t0:dt:tf
  pdomain = (1,10)         # μ ∈ [1,10]  (1-parameter family)
  ptspace = TransientParamSpace(pdomain,tdomain)

  feop  = TransientLinearParamOperator(res,(stiffness,damping,mass),ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
  vh0μ(μ) = interpolate_everywhere(v0μ(μ),trial(μ,t0))

  r = realisation(feop; nparams)
  return feop,r,uh0μ,vh0μ
end

function _make_transient_cellfield(odeslvr,feop,us,us0,odecache)
  odeop = get_algebraic_operator(feop)
  _,odeopcache = odecache
  ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
end

function _make_transient_cellfield(odeslvr::ThetaMethod,feop,us,us0,odecache)
  u, = us 
  u0, = us0 
  v = (u - u0) / odeslvr.dt 
  us = (u,v)
  odeop = get_algebraic_operator(feop)
  _,odeopcache = odecache
  ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
end

function _compare_with_gridap_heateq(odeslvr,snaps,A,b)
  domain = (0,1,0,1)
  partition = (6,6)
  model = CartesianDiscreteModel(domain,partition)

  order = 1
  degree = 2*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test  = TestFESpace(Ω,reffe; conformity=:H1,dirichlet_tags="boundary")

  dt = 0.05; t0 = 0.0; tf = 4*dt

  data = get_param_data(snaps)
  r = get_realisation(snaps)
  np = num_params(r)

  for (i,(μ,_)) in enumerate(r) 
    at(t) = x -> μ[1]
    ft(t) = x -> 1.0
    gt(t) = x -> x[1] * (1 + t)
    u0(x) = gt(0.0)(x)

    a = TimeSpaceFunction(at)
    f = TimeSpaceFunction(ft)
    g = TimeSpaceFunction(gt)

    stiffness(t,u,v) = ∫(a(t) * ∇(v) ⋅ ∇(u))dΩ
    mass(_,uₜ,v)     = ∫(v * uₜ)dΩ
    rhs(t,v)         = ∫(f(t) * v)dΩ
    res(t,u,v)       = mass(t,∂t(u),v) + stiffness(t,u,v) - rhs(t,v)

    trial = TransientTrialFESpace(test,g)
    feop  = TransientLinearFEOperator((stiffness,mass),rhs,trial,test)
    odeop = get_algebraic_operator(feop)
    uh0 = interpolate_everywhere(u0,trial(t0))

    fesltn = solve(odeslvr,feop,t0,tf,uh0)

    for (n,((t_n,u_n),(_,state_n,state_0,_,odecache))) in enumerate(fesltn.odesltn)
      ipt = i + (n-1)*np
      An = assemble_matrix((u,v)->stiffness(t_n,u,v),trial(t_n),test)
      Mn = assemble_matrix((u,v)->mass(t_n,u,v),trial(t_n),test)
      uh_nt = _make_transient_cellfield(odeslvr,feop,state_n,state_0,odecache)
      rn = assemble_vector(v->res(t_n,uh_nt,v),test)
      @test u_n ≈ param_getindex(data,ipt)
      @test An ≈ param_getindex(A[1][1],ipt)
      @test Mn ≈ param_getindex(A[2][1],ipt)
      @test rn ≈ param_getindex(b[1],ipt)
    end
  end
end

@testset "SpaceTime solution+residual+jacobian — heat equation" begin
  dt = 0.05
  fesolvers = (
    ThetaMethod(LUSolver(),dt,5*dt),
    GeneralizedAlpha1(LUSolver(),5*dt,0.5)
  )

  for fesolver in fesolvers 
    feop,r,uh0μ = _heat_eq_setup(;nparams=3)

    # Collect solution snapshots via the standard offline pipeline
    sol    = solve(fesolver,feop,r,uh0μ)
    vals,_  = collect(sol)
    initial_vals = initial_conditions(sol)
    i      = get_dof_map(feop)
    snaps  = Snapshots(vals,initial_vals,i,r)

    b = Algebra.residual(fesolver,feop,r,snaps)
    A = Algebra.jacobian(fesolver,feop,r,snaps)

    @test isa(b,ArrayContribution)
    @test isa(A,TupOfArrayContribution)
    _compare_with_gridap_heateq(fesolver,snaps,A,b)
  end
end

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Wave equation — SpaceTime.jl residual and jacobian (GeneralizedAlpha2)
# ─────────────────────────────────────────────────────────────────────────────

@testset "SpaceTime solution+residual+jacobian — wave equation" begin
  dt = 0.05
  fesolver = GeneralizedAlpha2(LUSolver(),dt,0.5)

  feop,r,uh0μ,vh0μ = _wave_eq_setup(;nparams=2)

  sol          = solve(fesolver,feop,r,(uh0μ,vh0μ))
  vals,_      = collect(sol)
  initial_vals = initial_conditions(sol)
  i            = get_dof_map(feop)
  snaps        = Snapshots(vals,initial_vals,i,r)

  b = Algebra.residual(fesolver,feop,r,snaps)
  A = Algebra.jacobian(fesolver,feop,r,snaps)

  @test isa(b,ArrayContribution)
  @test isa(A,TupOfArrayContribution)
  b = sum(b.values)
  A = sum(A[1].values) + sum(A[2].values) + sum(A[3].values)

  @test _compare_with_gridap_waveeq(fesolver,feop,r,snaps,A,b)
end

function _compare_with_gridap_waveeq(_,_,r,_,A,b)
  domain    = (0,1,0,1); partition = (6,6)
  model     = CartesianDiscreteModel(domain,partition)
  reffe     = ReferenceFE(lagrangian,Float64,1)
  V         = TestFESpace(model,reffe; conformity=:H1,dirichlet_tags="boundary")
  U0        = TrialFESpace(V,0.0)
  Ω         = Triangulation(model)
  dΩ        = Measure(Ω,2)

  np = num_params(r)
  nt = num_times(r)

  # 1 — residual ≈ 0 at the discrete solution
  b_data = get_all_data(b)
  ok_res = true
  for ip in 1:np,it in 1:nt
    ipt    = (it-1)*np + ip
    ok_res = ok_res && (norm(b_data[:,ipt]) < 1e-8)
  end

  # 2 — jacobian ≈ K(μ) + 2*M from Gridap  (damping C = M bilinear form)
  M_ref = assemble_matrix((u,v) -> ∫(v * u)dΩ,U0,V)

  ok_jac = true
  for (ip,μ_vec) in enumerate(get_params(r))
    μ     = μ_vec[1]
    K_ref = assemble_matrix((u,v) -> ∫(μ * ∇(v) ⋅ ∇(u))dΩ,U0,V)
    A_ref = K_ref + 2 * M_ref

    for it in 1:nt
      ipt      = (it-1)*np + ip
      nz       = collect(param_getindex(A,ipt))
      A_groms  = SparseMatrixCSC(A_ref.m,A_ref.n,A_ref.colptr,A_ref.rowval,nz)
      ok_jac   = ok_jac && isapprox(Matrix(A_groms),Matrix(A_ref),atol=1e-8)
    end
  end

  return ok_res && ok_jac
end


#

dt = 0.05
fesolvers = (
ThetaMethod(LUSolver(),dt,5*dt),
GeneralizedAlpha1(LUSolver(),5*dt,0.5)
)

fesolver = fesolvers[1]
feop,r,uh0μ = _heat_eq_setup(;nparams=3)

# Collect solution snapshots via the standard offline pipeline
sol    = solve(fesolver,feop,r,uh0μ)
vals,_  = collect(sol)
initial_vals = initial_conditions(sol)
i      = get_dof_map(feop)
snaps  = Snapshots(vals,initial_vals,i,r)

b = Algebra.residual(fesolver,feop,r,snaps)
A = Algebra.jacobian(fesolver,feop,r,snaps)

@test isa(b,ArrayContribution)
@test isa(A,TupOfArrayContribution)

# gridap 

domain = (0,1,0,1)
partition = (6,6)
model = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

reffe = ReferenceFE(lagrangian,Float64,order)
test  = TestFESpace(Ω,reffe; conformity=:H1,dirichlet_tags="boundary")

dt = 0.05; t0 = 0.0; tf = 4*dt

data = get_param_data(snaps)
r = get_realisation(snaps)
np = num_params(r)

i = 1
μ = r.params.params[i]

at(t) = x -> μ[1]
ft(t) = x -> 1.0
gt(t) = x -> x[1] * (1 + t)
u0(x) = gt(0.0)(x)

a = TimeSpaceFunction(at)
f = TimeSpaceFunction(ft)
g = TimeSpaceFunction(gt)

stiffness(t,u,v) = ∫(a(t) * ∇(v) ⋅ ∇(u))dΩ
mass(_,uₜ,v)     = ∫(v * uₜ)dΩ
rhs(t,v)         = ∫(f(t) * v)dΩ
res(t,u,v)       = mass(t,∂t(u),v) + stiffness(t,u,v) - rhs(t,v)

trial = TransientTrialFESpace(test,g)
feop  = TransientLinearFEOperator((stiffness,mass),rhs,trial,test)
odeop = get_algebraic_operator(feop)
uh0 = interpolate_everywhere(u0,trial(t0))

fesltn = solve(odeslvr,feop,t0,tf,uh0)

n = 1
(t_n,u_n),(_,state_n,state_0,_,odecache) = iterate(fesltn.odesltn)
tθ = t0 + odeslvr.θ*odeslvr.dt
ipt = i + (n-1)*np
An = assemble_matrix((u,v)->stiffness(tθ,u,v),trial(tθ),test)
Mn = assemble_matrix((u,v)->mass(tθ,u,v),trial(tθ),test)
uh_nt = _make_transient_cellfield(odeslvr,feop,state_n,state_0,odecache)
rn = assemble_vector(v->res(tθ,uh_nt,v),test)
@test u_n ≈ param_getindex(data,ipt)
@test An ≈ param_getindex(A[1][1],ipt)
@test Mn ≈ param_getindex(A[2][1],ipt)
@test rn ≈ param_getindex(b[1],ipt)
