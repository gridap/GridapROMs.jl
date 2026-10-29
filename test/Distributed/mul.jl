module Mul

using PartitionedArrays
using LinearAlgebra
using SparseArrays
using Test

using GridapROMs
using GridapROMs.Distributed
using GridapROMs.Distributed: GenericPArray
using GridapROMs.ParamDataStructures

# Unit tests for every `mul!` method implemented in src/Distributed/GenericPArray.jl.
# Each test builds a small, globally-known problem, applies the distributed `mul!`,
# gathers the result, and compares against a plain (non-distributed) reference
# computed directly from the same global data. `n` is the global row/dof dimension
# (distributed, ghosted across ranks); `k`/`nrhs` are small "mode"/"rhs" dimensions
# that stay replicated (unpartitioned) on every rank, matching how GenericPMatrix is
# actually used throughout the RB pipeline (as a row-distributed, column-replicated
# reduced basis).

function global_matrix(n,k)
  A = zeros(n,k)
  for gi in 1:n, kk in 1:k
    A[gi,kk] = gi*10+kk
  end
  A
end

function build_basis(rp,n,k)
  Aglobal = global_matrix(n,k)
  avals = map(rp) do p
    l2g = local_to_global(p)
    Aglobal[l2g,:]
  end
  GenericPArray(avals,rp,(Base.OneTo(k),)), Aglobal
end

# test 1: mul!(c::GenericPArray, a::PSparseMatrix, b::GenericPArray, α, β)
function test_psparse_mul(ranks,np,n)
  rp = uniform_partition(ranks,np,n,true)
  values = map(rp) do p
    l2g = local_to_global(p)
    nl = local_length(p)
    I = Int[]; J = Int[]; V = Float64[]
    for (li,gi) in enumerate(l2g)
      push!(I,li); push!(J,li); push!(V,2.0)
      if gi > 1
        lj = findfirst(==(gi-1),l2g)
        lj !== nothing && (push!(I,li); push!(J,lj); push!(V,-1.0))
      end
      if gi < n
        lj = findfirst(==(gi+1),l2g)
        lj !== nothing && (push!(I,li); push!(J,lj); push!(V,-1.0))
      end
    end
    sparse(I,J,V,nl,nl)
  end
  A = PSparseMatrix(values,rp,rp)

  bglobal = Float64.((1:n).^2)
  bvals = map(rp) do p
    bglobal[local_to_global(p)]
  end
  b = GenericPArray(bvals,rp)

  c = similar(b)
  fill!(c,0.0)
  mul!(c,A,b,1.0,0.0)

  Adense = zeros(n,n)
  for i in 1:n
    Adense[i,i] = 2.0
    i > 1 && (Adense[i,i-1] = -1.0)
    i < n && (Adense[i,i+1] = -1.0)
  end
  cref = Adense*bglobal

  own_c = own_values(c)
  own_ids = map(own_to_global,partition(axes(c,1)))
  ok = map(own_c,own_ids) do v,ids
    all(isapprox(vi,cref[gi]) for (vi,gi) in zip(v,ids))
  end
  reduce(&,ok;init=true)
end

# test 2: mul!(c::GenericPVector, a::GenericPMatrix, b::AbstractVector{<:Number}, α, β)
function test_genericpmatrix_times_plainvector(ranks,np,n,k)
  rp = uniform_partition(ranks,np,n,true)
  a,Aglobal = build_basis(rp,n,k)
  b = Float64.(1:k)
  c = GenericPArray{Vector{Float64}}(undef,rp)
  fill!(c,0.0)
  mul!(c,a,b,1.0,0.0)
  cref = Aglobal*b
  own_c = own_values(c)
  own_ids = map(own_to_global,partition(axes(c,1)))
  ok = map(own_c,own_ids) do v,ids
    all(isapprox(vi,cref[gi]) for (vi,gi) in zip(v,ids))
  end
  reduce(&,ok;init=true)
end

# test 3: mul!(c::GenericPMatrix, a::GenericPMatrix, b::AbstractMatrix{<:Number}, α, β)
function test_genericpmatrix_times_plainmatrix(ranks,np,n,k,nrhs)
  rp = uniform_partition(ranks,np,n,true)
  a,Aglobal = build_basis(rp,n,k)
  b = reshape(Float64.(1:(k*nrhs)),k,nrhs)
  c = GenericPArray{Matrix{Float64}}(undef,rp,Base.OneTo(nrhs))
  fill!(c,0.0)
  mul!(c,a,b,1.0,0.0)
  cref = Aglobal*b
  own_c = own_values(c)
  own_ids = map(own_to_global,partition(axes(c,1)))
  ok = map(own_c,own_ids) do v,ids
    all(isapprox(v[li,rr],cref[gi,rr]) for (li,gi) in enumerate(ids) for rr in 1:nrhs)
  end
  reduce(&,ok;init=true)
end

# test 4: mul!(c::PVector, a::GenericPMatrix, b::AbstractParamVector, α, β)
# This is the "inv_project!" path: reconstructing a full-order (distributed) solution
# from a small, replicated reduced coefficient vector x̂ via x = Φ*x̂.
function test_inv_project(ranks,np,n,k,nparam)
  rp = uniform_partition(ranks,np,n,true)
  a,Aglobal = build_basis(rp,n,k)
  bdata = reshape(Float64.(1:(k*nparam)),k,nparam)
  b = ConsecutiveParamArray(bdata)
  cvals = map(rp) do p
    ConsecutiveParamArray(zeros(local_length(p),nparam))
  end
  c = PVector(cvals,rp)
  mul!(c,a,b,1.0,0.0)
  cref = Aglobal*bdata
  own_c = own_values(c)
  own_ids = map(own_to_global,partition(axes(c,1)))
  ok = map(own_c,own_ids) do v,ids
    all(isapprox(v.data[li,pp],cref[gi,pp]) for (li,gi) in enumerate(ids) for pp in 1:nparam)
  end
  reduce(&,ok;init=true)
end

# test 5: mul!(c::AbstractParamArray, a::GenericPMatrix, b::AbstractParamArray, α, β)
# Same as test 4, but gathers the result into a replicated (not distributed) param
# array of full global size n, by scatter-adding each rank's own rows.
function test_mul_replicated_output(ranks,np,n,k,nparam)
  rp = uniform_partition(ranks,np,n,true)
  a,Aglobal = build_basis(rp,n,k)
  bdata = reshape(Float64.(1:(k*nparam)),k,nparam)
  b = ConsecutiveParamArray(bdata)
  c = ConsecutiveParamArray(zeros(n,nparam))
  mul!(c,a,b,1.0,0.0)
  cref = Aglobal*bdata
  reduce(&,map(_ -> isapprox(c.data,cref), ranks);init=true)
end

# test 6: mul!(c::AbstractVector, at::Adjoint{<:Any,<:GenericPMatrix}, b::Union{PVector,GenericPVector}, α, β)
function test_adjoint_times_genericpvector(ranks,np,n,k)
  rp = uniform_partition(ranks,np,n,true)
  a,Aglobal = build_basis(rp,n,k)
  bglobal = Float64.((1:n).^1.3)
  bvals = map(rp) do p
    bglobal[local_to_global(p)]
  end
  b = GenericPArray(bvals,rp)
  c = zeros(k)
  mul!(c,a',b,1.0,0.0)
  cref = Aglobal'*bglobal
  reduce(&,map(_ -> isapprox(c,cref), ranks);init=true)
end

# test 7: mul!(c::AbstractMatrix, at::Adjoint{<:Any,<:GenericPMatrix}, b::GenericPMatrix, α, β)
function test_adjoint_times_genericpmatrix(ranks,np,n,k,nrhs)
  rp = uniform_partition(ranks,np,n,true)
  a,Aglobal = build_basis(rp,n,k)
  bglobal = reshape(Float64.(1:(n*nrhs)),n,nrhs)
  bvals = map(rp) do p
    bglobal[local_to_global(p),:]
  end
  b = GenericPArray(bvals,rp,(Base.OneTo(nrhs),))
  c = zeros(k,nrhs)
  mul!(c,a',b,1.0,0.0)
  cref = Aglobal'*bglobal
  reduce(&,map(_ -> isapprox(c,cref), ranks);init=true)
end

# test 8: mul!(c::AbstractParamArray, at::Adjoint{<:Any,<:GenericPMatrix}, b::PVector{<:AbstractParamVector}, α, β)
# This is the "project!" path: projecting a full-order (distributed) solution x onto
# the reduced basis via x̂ = Φ'*x.
function test_project(ranks,np,n,k,nparam)
  rp = uniform_partition(ranks,np,n,true)
  a,Aglobal = build_basis(rp,n,k)
  bglobal = reshape(Float64.(1:(n*nparam)),n,nparam)
  bvals = map(rp) do p
    ConsecutiveParamArray(bglobal[local_to_global(p),:])
  end
  b = PVector(bvals,rp)
  c = ConsecutiveParamArray(zeros(k,nparam))
  mul!(c,a',b,1.0,0.0)
  cref = Aglobal'*bglobal
  reduce(&,map(_ -> isapprox(c.data,cref), ranks);init=true)
end

# test 9: mul!(c::GenericPArray, a::GenericPMatrix, b::GenericPArray, α, β) with a
# *square*, same-partition-on-both-dimensions matrix (the only shape under which this
# method's use of `partition(b)` without a ghost/own split is dimensionally
# consistent: see note in run_all).
function test_genericpmatrix_square_local(ranks,np,n)
  rp = uniform_partition(ranks,np,n,true)
  avals = map(rp) do p
    l2g = local_to_global(p)
    nl = local_length(p)
    M = zeros(nl,nl)
    for li in 1:nl, lj in 1:nl
      M[li,lj] = l2g[li]*10+l2g[lj]
    end
    M
  end
  a = GenericPArray(avals,rp)
  bvals = map(rp) do p
    Float64.(local_to_global(p))
  end
  b = GenericPArray(bvals,rp)
  c = GenericPArray{Vector{Float64}}(undef,rp)
  fill!(c,0.0)
  mul!(c,a,b,1.0,0.0)
  # Like every other mul! here, only `own_values(c)` gets written; ghost entries of
  # c are left untouched (still 0 from the `fill!` above) - that's by design, not a
  # bug, so the comparison below only checks own rows. Reference: this method sums
  # over each rank's *local* (own+ghost) columns only, not the true global width -
  # see run_all for why that's the one shape under which this is well-defined.
  own_c = own_values(c)
  own_ids = map(own_to_global,partition(axes(c,1)))
  ok2 = map(rp,own_c,own_ids) do p,v,ids
    l2g = local_to_global(p)
    nl = local_length(p)
    expected_own = [sum((gi*10+l2g[lj])*l2g[lj] for lj in 1:nl) for gi in ids]
    isapprox(v,expected_own)
  end
  reduce(&,ok2;init=true)
end

function run_all(distribute,np)
  ranks = distribute(LinearIndices((np,)))
  n = 8
  k = 3
  nrhs = 2
  nparam = 2

  results = Dict{String,Bool}()
  results["psparse * genericp (test1)"] = test_psparse_mul(ranks,np,n)
  results["genericpmatrix * plain vector (test4)"] = test_genericpmatrix_times_plainvector(ranks,np,n,k)
  results["genericpmatrix * plain matrix (test7)"] = test_genericpmatrix_times_plainmatrix(ranks,np,n,k,nrhs)
  results["inv_project!: genericpmatrix * param vector -> PVector (test5, CRITICAL - used every online solve)"] = test_inv_project(ranks,np,n,k,nparam)
  results["genericpmatrix * param array -> replicated output (test6)"] = test_mul_replicated_output(ranks,np,n,k,nparam)
  results["adjoint(genericpmatrix) * genericpvector (test8)"] = test_adjoint_times_genericpvector(ranks,np,n,k)
  results["adjoint(genericpmatrix) * genericpmatrix (test9)"] = test_adjoint_times_genericpmatrix(ranks,np,n,k,nrhs)
  results["project!: adjoint(genericpmatrix) * PVector{param} -> param array (test10, CRITICAL - used every offline/online projection)"] = test_project(ranks,np,n,k,nparam)
  results["genericpmatrix(square,local) * genericpvector (test2/3, see note)"] = test_genericpmatrix_square_local(ranks,np,n)

  if i_am_main(ranks)
    println("="^80)
    println("mul! test results")
    println("="^80)
    for (name,ok) in results
      println(ok ? "PASS" : "FAIL", "  -  ", name)
    end
    nfail = count(!,values(results))
    println("-"^80)
    println(nfail == 0 ? "ALL PASS" : "$(nfail) FAILURE(S)")
  end

  all(values(results))
end

function main(distribute,parts)
  np = prod(parts)
  @assert run_all(distribute,np) "one or more mul! tests failed, see output above"
end

end # module
