module TimeCombinationsTest

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

const time_combination = ParamODEs.time_combination

cpa(data::Matrix) = ConsecutiveParamArray(data)

@testset "time_combination ThetaMethod θ=0.5 displacement" begin
  dt = 0.1; θ = 0.5
  u_data  = Float64[1 2 3 4 5 6]
  u0_data = Float64[0 0.5]

  u  = cpa(u_data)
  u0 = cpa(u0_data)

  v0  = cpa(similar(u0_data))
  c   = ThetaMethodCombination(dt,θ)
  usx = time_combination(c,u,(u0,v0))

  @test length(usx) == 2
  uθ = get_all_data(usx[1])

  @test uθ[1,1] ≈ θ*1 + (1-θ)*0
  @test uθ[1,2] ≈ θ*2 + (1-θ)*0.5

  @test uθ[1,3] ≈ θ*3 + (1-θ)*1
  @test uθ[1,4] ≈ θ*4 + (1-θ)*2

  @test uθ[1,5] ≈ θ*5 + (1-θ)*3
  @test uθ[1,6] ≈ θ*6 + (1-θ)*4
end

@testset "time_combination ThetaMethod θ=1 implicit Euler" begin
  dt = 0.1; θ = 1.0
  u_data  = Float64[1 2 3 4 5 6]
  u0_data = Float64[0 0.5]

  u  = cpa(u_data);  u0 = cpa(u0_data)
  c  = ThetaMethodCombination(dt,θ)
  uθ = get_all_data(time_combination(c,u,(u0,similar(u0)))[1])

  @test uθ[1,1] ≈ 1.0
  @test uθ[1,2] ≈ 2.0
  @test uθ[1,3] ≈ 3.0
  @test uθ[1,5] ≈ 5.0
end

@testset "time_combination ThetaMethod θ=0 explicit Euler" begin
  dt = 0.1; θ = 0.0
  u_data  = Float64[1 2 3 4 5 6]
  u0_data = Float64[0 0.5]

  u  = cpa(u_data);  u0 = cpa(u0_data)
  c  = ThetaMethodCombination(dt,θ)
  uθ = get_all_data(time_combination(c,u,(u0,similar(u0)))[1])

  @test uθ[1,1] ≈ 0.0
  @test uθ[1,2] ≈ 0.5
  @test uθ[1,3] ≈ 1.0
  @test uθ[1,4] ≈ 2.0
  @test uθ[1,5] ≈ 3.0
end

@testset "time_combination ThetaMethod velocity (CombinationOrder{2})" begin
  dt = 0.1; θ = 0.5
  np = 1; nt = 3; ndof = 1
  u_data  = reshape(Float64[1.0,3.0,6.0],1,3)
  u0_data = reshape(Float64[0.0],  1,1)

  u  = cpa(u_data);  u0 = cpa(u0_data)

  c_vel = CombinationOrder{2}(ThetaMethodCombination(dt,θ))
  v = similar(u)
  ParamODEs._combination!(v,c_vel,u,(u0,similar(u0)))
  vdata = get_all_data(v)

  @test vdata[1,1] ≈ (1.0 - 0.0) / dt
  @test vdata[1,2] ≈ (3.0 - 1.0) / dt
  @test vdata[1,3] ≈ (6.0 - 3.0) / dt
end


@testset "time_combination GenAlpha1 (ρ∞=1) displacement" begin
  dt = 0.1
  αf = 0.5; αm = 0.5; γ = 0.5
  u_data  = Float64[1 2 3 4 5 6]
  u0_data = Float64[0 0.5]
  v0_data = Float64[0 0]

  u  = cpa(u_data)
  u0 = cpa(u0_data)
  v0 = cpa(v0_data)

  c   = GenAlpha1Combination(dt,αf,αm,γ)
  usx = time_combination(c,u,(u0,v0))

  @test length(usx) == 2

  uαf = get_all_data(usx[1])
  @test uαf[1,1] ≈ 0.5*1 + 0.5*0
  @test uαf[1,2] ≈ 0.5*2 + 0.5*0.5
  @test uαf[1,3] ≈ 0.5*3 + 0.5*1
  @test uαf[1,4] ≈ 0.5*4 + 0.5*2
  @test uαf[1,5] ≈ 0.5*5 + 0.5*3
end

@testset "time_combination GenAlpha1 (ρ∞=1) velocity (αm=γ=0.5 → FD)" begin
  dt = 0.1
  αf = 0.5; αm = 0.5; γ = 0.5
  np = 1; nt = 3; ndof = 1
  u_data  = reshape(Float64[1.0,3.0,6.0],1,3)
  u0_data = reshape(Float64[0.0],   1,1)
  v0_data = reshape(Float64[0.0],   1,1)

  u  = cpa(u_data);  u0 = cpa(u0_data);  v0 = cpa(v0_data)
  c  = GenAlpha1Combination(dt,αf,αm,γ)
  vαm = get_all_data(time_combination(c,u,(u0,v0))[2])

  @test vαm[1,1] ≈ (1.0 - 0.0) / dt
  @test vαm[1,2] ≈ (3.0 - 1.0) / dt
  @test vαm[1,3] ≈ (6.0 - 3.0) / dt
end

@testset "time_combination GenAlpha1 (ρ∞=0.5) velocity — multi-step sum" begin
  fesolver_ref = GeneralizedAlpha1(LUSolver(),0.1,0.5)
  dt = fesolver_ref.dt
  αf = fesolver_ref.αf; αm = fesolver_ref.αm; γ = fesolver_ref.γ

  a = 1/(γ*dt)
  b = 1 - 1/γ
  c_coeff = a * (1 - αm + b*αm)

  np = 1; nt = 3; ndof = 1
  u_raw   = [2.0,5.0,9.0]
  u0_raw  = 0.0
  v0_raw  = 1.0

  u_data  = reshape(u_raw,1,nt)
  u0_data = reshape([u0_raw],1,1)
  v0_data = reshape([v0_raw],1,1)
  u  = cpa(u_data);  u0 = cpa(u0_data);  v0 = cpa(v0_data)

  c  = GenAlpha1Combination(dt,αf,αm,γ)
  vαm = get_all_data(time_combination(c,u,(u0,v0))[2])

  expected_t1 = a*αm*u_raw[1] - a*αm*u0_raw + (c_coeff/a)*v0_raw
  @test vαm[1,1] ≈ expected_t1 atol=1e-10

  expected_t2 = a*αm*u_raw[2] + (c_coeff - a*αm)*u_raw[1] - c_coeff*u0_raw + (c_coeff/a)*b*v0_raw
  @test vαm[1,2] ≈ expected_t2 atol=1e-10

  η3 = c_coeff*(b^1 - b^0)
  expected_t3 = a*αm*u_raw[3] + (c_coeff - a*αm)*u_raw[2] + η3*u_raw[1] - c_coeff*b*u0_raw + (c_coeff/a)*b^2*v0_raw
  @test vαm[1,3] ≈ expected_t3 atol=1e-10
end

@testset "time_combination GenAlpha2 (ρ∞=1) displacement" begin
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
  usx = time_combination(c,u,(u0,v0,a0))

  @test length(usx) == 3

  uαf = get_all_data(usx[1])
  @test uαf[1,1] ≈ 0.5*1 + 0.5*0
  @test uαf[1,2] ≈ 0.5*2 + 0.5*0.5
  @test uαf[1,3] ≈ 0.5*3 + 0.5*1
  @test uαf[1,4] ≈ 0.5*4 + 0.5*2
  @test uαf[1,5] ≈ 0.5*5 + 0.5*3
end

@testset "time_combination GenAlpha2 (ρ∞=1) velocity (zero ICs → FD)" begin
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
  vαf = get_all_data(time_combination(c,u,(u0,v0,a0))[2])

  @test vαf[1,1] ≈ (1.0 - 0.0) / dt
  @test vαf[1,2] ≈ (3.0 - 1.0) / dt
  @test vαf[1,3] ≈ (6.0 - 3.0) / dt
end

@testset "time_combination GenAlpha2 (ρ∞=0.5) velocity and acceleration — multi-step sums" begin
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

  vαf = get_all_data(time_combination(c,u,(u0,v0,a0))[2])
  aαm = get_all_data(time_combination(c,u,(u0,v0,a0))[3])

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

  αnj(n,j) = ([1.0 0.0] * P^(n-j) * [a,e])[1]
  βnj(n,j) = ([1.0 0.0] * P^(n-j) * [b,f])[1]
  κnj(n,j) = αnj(n,j-1) + βnj(n,j)

  γnj(n,j) = ([0.0 1.0] * P^(n-j) * [a,e])[1]
  δnj(n,j) = ([0.0 1.0] * P^(n-j) * [b,f])[1]
  ηnj(n,j) = γnj(n,j-1) + δnj(n,j)

  av(n) = (1 - αf) * αnj(n,n)
  bv(n) = (1 - αf) * κnj(n,n) + αf * αnj(n-1,n-1)
  cv(n) = (1 - αf) * βnj(n,0) + αf * βnj(n-1,0)
  dv(n) = (1 - αf) * an(n+1) + αf * an(n)
  ev(n) = (1 - αf) * bn(n+1) + αf * bn(n)

  aa(n) = (1 - αm) * γnj(n,n)
  ba(n) = (1 - αm) * ηnj(n,n) + αm * γnj(n-1,n-1)
  ca(n) = (1 - αm) * δnj(n,0) + αm * δnj(n-1,0)
  da(n) = (1 - αm) * cn(n+1) + αm * cn(n)
  ea(n) = (1 - αm) * dn(n+1) + αm * dn(n)

  expected_v1 = (1 - αf) * αnj(0,0) * u_raw[1] +
                (1 - αf) * βnj(0,0) * u0_raw +
                ((1 - αf) * an(1) + αf) * v0_raw +
                (1 - αf) * bn(1) * a0_raw
  expected_v2 = av(1) * u_raw[2] +
                bv(1) * u_raw[1] +
                cv(1) * u0_raw +
                dv(1) * v0_raw +
                ev(1) * a0_raw
  expected_v3 = av(2) * u_raw[3] +
                bv(2) * u_raw[2] +
                ((1 - αf) * κnj(2,1) + αf * κnj(1,1)) * u_raw[1] +
                cv(2) * u0_raw +
                dv(2) * v0_raw +
                ev(2) * a0_raw

  expected_a1 = (1 - αm) * γnj(0,0) * u_raw[1] +
                (1 - αm) * δnj(0,0) * u0_raw +
                (1 - αm) * cn(1) * v0_raw +
                ((1 - αm) * dn(1) + αm) * a0_raw
  expected_a2 = aa(1) * u_raw[2] +
                ba(1) * u_raw[1] +
                ca(1) * u0_raw +
                da(1) * v0_raw +
                ea(1) * a0_raw
  expected_a3 = aa(2) * u_raw[3] +
                ba(2) * u_raw[2] +
                ((1 - αm) * ηnj(2,1) + αm * ηnj(1,1)) * u_raw[1] +
                ca(2) * u0_raw +
                da(2) * v0_raw +
                ea(2) * a0_raw

  @test vαf[1,1] ≈ expected_v1 atol=1e-10
  @test vαf[1,2] ≈ expected_v2 atol=1e-10
  @test vαf[1,3] ≈ expected_v3 atol=1e-10

  @test aαm[1,1] ≈ expected_a1 atol=1e-10
  @test aαm[1,2] ≈ expected_a2 atol=1e-10
  @test aαm[1,3] ≈ expected_a3 atol=1e-10
end

@testset "time_combination consistent with CombinationOrder accessors" begin
  dt = 0.1; θ = 0.5
  u_data  = rand(3,6)
  u0_data = rand(3,2)

  u  = cpa(u_data);  u0 = cpa(u0_data)
  c  = ThetaMethodCombination(dt,θ)
  usx = time_combination(c,u,(u0,similar(u0)))

  c1 = CombinationOrder{1}(c)
  uθ_direct = similar(u)
  ParamODEs._combination!(uθ_direct,c1,u,(u0,similar(u0)))

  @test get_all_data(usx[1]) ≈ get_all_data(uθ_direct)
end

@testset "galerkin_projection ThetaMethod θ=1 (implicit Euler)" begin
  Nt = 8; nl = 3; n = 4; nr = 5
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  dt = 0.1; θ = 1.0
  c1 = CombinationOrder{1}(ThetaMethodCombination(dt,θ))
  proj = galerkin_projection(Φl,Φ,Φr,c1)

  @test size(proj) == (nl,n,nr)

  proj_ref = zeros(nl,n,nr)
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt
      proj_ref[i,k,j] += 1.0 * Φl[α,i] * Φ[α,k] * Φr[α,j]
    end
  end
  @test proj ≈ proj_ref atol=1e-12
end

@testset "galerkin_projection ThetaMethod θ=0 (explicit Euler)" begin
  Nt = 8; nl = 3; n = 4; nr = 5
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  dt = 0.1; θ = 0.0
  c1 = CombinationOrder{1}(ThetaMethodCombination(dt,θ))
  proj = galerkin_projection(Φl,Φ,Φr,c1)

  proj_ref = zeros(nl,n,nr)
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt-1
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
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt
      proj_ref[i,k,j] += 0.5 * Φl[α,i] * Φ[α,k] * Φr[α,j]
    end
  end
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
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt
      proj_ref[i,k,j] += dt_inv * Φl[α,i] * Φ[α,k] * Φr[α,j]
    end
  end
  for i=1:nl,k=1:n,j=1:nr
    for α=1:Nt-1
      proj_ref[i,k,j] -= dt_inv * Φl[α+1,i] * Φ[α+1,k] * Φr[α,j]
    end
  end
  @test proj ≈ proj_ref atol=1e-12
end

@testset "galerkin_projection GenAlpha1 (ρ∞=1) displacement" begin
  Nt = 6; nl = 2; n = 3; nr = 4
  Φl = Matrix(qr(rand(Nt,nl)).Q)
  Φ  = rand(Nt,n)
  Φr = Matrix(qr(rand(Nt,nr)).Q)

  αf = 0.5; αm = 0.5; γ = 0.5; dt = 0.1
  c1_alpha = CombinationOrder{1}(GenAlpha1Combination(dt,αf,αm,γ))
  c1_theta = CombinationOrder{1}(ThetaMethodCombination(dt,αf))

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
  c1_theta = CombinationOrder{1}(ThetaMethodCombination(dt,αf))
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

function _heat_eq_setup(;nparams=3)
  domain = (0,1,0,1)
  partition = (6,6)
  model = CartesianDiscreteModel(domain,partition)

  order = 1
  degree = 2*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(μ,t) = x -> μ[1]
  aμt(μ,t) = parameterise(a,μ,t)
  f(μ,t) = x -> 1.0
  fμt(μ,t) = parameterise(f,μ,t)
  g(μ,t) = x -> x[1] * (1 + t)
  gμt(μ,t) = parameterise(g,μ,t)
  u0(μ)   = x -> g(μ,0.0)(x)
  u0μ(μ)  = parameterise(u0,μ)
  v0(μ)   = x -> x[1]
  v0μ(μ)  = parameterise(v0,μ)

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
  pdomain = (1,10)
  ptspace = TransientParamSpace(pdomain,tdomain)

  feop  = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
  vh0μ(μ) = interpolate_everywhere(v0μ(μ),trial(μ,t0))

  r = realisation(feop; nparams)
  return feop,r,uh0μ,vh0μ
end

function _wave_eq_setup(;nparams=3)
  domain = (0,1,0,1)
  partition = (6,6)
  model = CartesianDiscreteModel(domain,partition)

  order = 1
  degree = 2*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(μ,t) = x -> μ[1]
  aμt(μ,t) = parameterise(a,μ,t)
  g(μ,t) = x -> x[1] * (1 + t)^2
  gμt(μ,t) = parameterise(g,μ,t)
  u0(μ)   = x -> x[1]
  u0μ(μ)  = parameterise(u0,μ)
  v0(μ)   = x -> 2 * x[1] * (1 + 0.0)
  v0μ(μ)  = parameterise(v0,μ)
  a0(μ)   = x -> 2 * x[1]
  a0μ(μ)  = parameterise(a0,μ)

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
  pdomain = (1,10)
  ptspace = TransientParamSpace(pdomain,tdomain)

  feop  = TransientLinearParamOperator(res,(stiffness,damping,mass),ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))
  vh0μ(μ) = interpolate_everywhere(v0μ(μ),trial(μ,t0))
  ah0μ(μ) = interpolate_everywhere(a0μ(μ),trial(μ,t0))

  r = realisation(feop; nparams)
  return feop,r,uh0μ,vh0μ,ah0μ
end

function _compare_with_gridap_heateq(odeslvr,snaps,b)
  domain    = (0,1,0,1)
  partition = (6,6)
  model     = CartesianDiscreteModel(domain,partition)

  order = 1
  degree = 2*order
  Ω  = Triangulation(model)
  dΩ = Measure(Ω,degree)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test  = TestFESpace(Ω,reffe; conformity=:H1,dirichlet_tags="boundary")

  data = get_param_data(snaps)
  r    = get_realisation(snaps)
  np   = num_params(r)

  t0 = get_initial_time(r)
  tf = get_final_time(r)

  for (ip,μ) in enumerate(get_params(r))
    at(_) = x -> μ[1]
    ft(_) = x -> 1.0
    gt(t) = x -> x[1] * (1 + t)
    vt(t) = x -> x[1]

    a = TimeSpaceFunction(at)
    f = TimeSpaceFunction(ft)
    g = TimeSpaceFunction(gt)

    stiffness(t,u,v) = ∫(a(t) * ∇(v) ⋅ ∇(u))dΩ
    mass(_,uₜ,v)     = ∫(v * uₜ)dΩ
    rhs(t,v)         = ∫(f(t) * v)dΩ

    trial  = TransientTrialFESpace(test,g)
    gfeop  = TransientLinearFEOperator((stiffness,mass),rhs,trial,test)
    uh0    = interpolate_everywhere(gt(t0),trial(t0))
    vh0    = interpolate_everywhere(vt(t0),trial(t0))

    if isa(odeslvr,ThetaMethod)
      fesltn = solve(odeslvr,gfeop,t0,tf,uh0)
    else
      fesltn = solve(odeslvr,gfeop,t0,tf,(uh0,vh0))
    end

    for (n,(_,uh_n)) in enumerate(fesltn)
      ipt = ip + (n-1)*np
      u_n = get_free_dof_values(uh_n)
      @test u_n ≈ param_getindex(data,ipt)
    end
  end

  b_sum  = sum(b.values)
  b_data = get_all_data(b_sum)
  nt = num_times(r)
  for ip in 1:np,it in 1:nt
    ipt = (it-1)*np + ip
    @test norm(b_data[:,ipt]) < 1e-8
  end
end

@testset "SpaceTime solution+residual+jacobian — heat equation" begin
  feop,r,uh0μ,vh0μ = _heat_eq_setup(;nparams=3)
  dt = 0.05
  fesolvers = (
    ThetaMethod(LUSolver(),dt,dt),
    GeneralizedAlpha1(LUSolver(),dt,0.5),
  )
  ics = (uh0μ,(uh0μ,vh0μ))

  fesolvers = (GeneralizedAlpha1(LUSolver(),dt,0.5),)
  ics = ((uh0μ,vh0μ),)

  for (fesolver,ic) in zip(fesolvers,ics)
    sol    = solve(fesolver,feop,r,ic)
    vals,_  = collect(sol)
    initial_vals = initial_conditions(sol)
    i      = get_dof_map(feop)
    snaps  = Snapshots(vals,initial_vals,i,r)

    b = spacetime_residual(fesolver,feop,r,snaps)
    A = spacetime_jacobian(fesolver,feop,r,snaps)

    @test isa(b,ArrayContribution)
    @test isa(A,TupOfArrayContribution)
    _compare_with_gridap_heateq(fesolver,snaps,b)
  end
end

function _compare_with_gridap_waveeq(odeslvr,snaps,b)
  domain    = (0,1,0,1)
  partition = (6,6)
  model     = CartesianDiscreteModel(domain,partition)

  order = 1
  degree = 2*order
  Ω  = Triangulation(model)
  dΩ = Measure(Ω,degree)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test  = TestFESpace(Ω,reffe; conformity=:H1,dirichlet_tags="boundary")

  data = get_param_data(snaps)
  r    = get_realisation(snaps)
  np   = num_params(r)

  t0 = get_initial_time(r)
  tf = get_final_time(r)

  for (ip,μ) in enumerate(get_params(r))
    at(_)  = x -> μ[1]
    gt(t)  = x -> x[1] * (1 + t)^2
    vt(t)  = x -> 2 * x[1] * (1 + t)
    aat(t)  = x -> 2 * x[1]

    a = TimeSpaceFunction(at)
    g = TimeSpaceFunction(gt)

    stiffness(t,u,v)   = ∫(a(t) * ∇(v) ⋅ ∇(u))dΩ
    damping(_,uₜ,v)    = ∫(v * uₜ)dΩ
    mass(_,uₜₜ,v)      = ∫(v * uₜₜ)dΩ
    rhs(_,v)           = ∫(0.0 * v)dΩ

    trial  = TransientTrialFESpace(test,g)
    gfeop  = TransientLinearFEOperator((stiffness,damping,mass),rhs,trial,test)
    uh0    = interpolate_everywhere(gt(0.0),trial(t0))
    vh0    = interpolate_everywhere(vt(0.0),trial(t0))
    ah0    = interpolate_everywhere(aat(0.0),trial(t0))

    fesltn = solve(odeslvr,gfeop,t0,tf,(uh0,vh0,ah0))

    for (n,(_,uh_n)) in enumerate(fesltn)
      ipt = ip + (n-1)*np
      u_n = get_free_dof_values(uh_n)
      @test u_n ≈ param_getindex(data,ipt)
    end
  end

  b_sum  = sum(b.values)
  b_data = get_all_data(b_sum)
  nt = num_times(r)
  for ip in 1:np,it in 1:nt
    ipt = (it-1)*np + ip
    @test norm(b_data[:,ipt]) < 1e-8
  end
end

@testset "SpaceTime solution+residual+jacobian — wave equation" begin
  dt = 0.05
  fesolver = GeneralizedAlpha2(LUSolver(),dt,0.5)

  feop,r,uh0μ,vh0μ,ah0μ = _wave_eq_setup(;nparams=2)

  sol          = solve(fesolver,feop,r,(uh0μ,vh0μ,ah0μ))
  vals,_       = collect(sol)
  initial_vals = initial_conditions(sol)
  i            = get_dof_map(feop)
  snaps        = Snapshots(vals,initial_vals,i,r)

  b = spacetime_residual(fesolver,feop,r,snaps)
  A = spacetime_jacobian(fesolver,feop,r,snaps)

  @test isa(b,ArrayContribution)
  @test isa(A,TupOfArrayContribution)
  _compare_with_gridap_waveeq(fesolver,snaps,b)
end

end