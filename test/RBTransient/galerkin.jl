module GalerkinProjectionTest

using Test
using LinearAlgebra
using SparseArrays

using Gridap
using Gridap.Arrays

using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.ParamODEs
using GridapROMs.RBSteady
using GridapROMs.RBTransient

consecutive_param(data) = ConsecutiveParamArray(data)

function _rand_basis(n,r)
	randn(n,r)
end

function _rand_param_sparse(m,n,np;density=0.4)
	mask = sprand(m,n,density)
	rows,cols,_ = findnz(mask)
	mats = Vector{SparseMatrixCSC{Float64,Int}}(undef,np)
	for i in 1:np
		vals = randn(length(rows))
		mats[i] = sparse(rows,cols,vals,m,n)
	end
	ParamArray(mats)
end

function _check_cache_order_3d(cache,proj)
	@test ndims(cache) == 3
	@test ndims(proj) == 3
	@test size(cache,1) == size(proj,1)
	@test size(cache,2) == size(proj,3)
	@test size(cache,3) == size(proj,2)
	@inbounds for ip in axes(proj,2)
		@test cache[:,:,ip] ≈ proj[:,ip,:]
	end
end

@testset "steady: matrix and param-vector projections" begin
	n,rl,nr,np = 9,3,4,5
	Φl = _rand_basis(n,rl)
	Φr = _rand_basis(n,nr)
	A = randn(n,nr)

	@test galerkin_projection(Φl,A) ≈ Φl' * A

	a_data = randn(n,np)
	a = consecutive_param(a_data)
	proj_a = galerkin_projection(Φl,a)
	@test proj_a ≈ Φl' * a_data

	@test begin
		cache = consecutive_param(zeros(rl,np))
		galerkin_projection!(cache,Φl,a)
		get_all_data(cache) ≈ proj_a
	end

	psm = _rand_param_sparse(n,n,np)
	proj_psm = galerkin_projection(Φl,psm,Φr)
	@test size(proj_psm) == (rl,np,nr)

	for ip in 1:np
		@test proj_psm[:,ip,:] ≈ Φl' * param_getindex(psm,ip) * Φr
	end

	@test begin
		cache3 = consecutive_param(zeros(rl,nr,np))
		galerkin_projection!(cache3,Φl,psm,Φr)
		_check_cache_order_3d(get_all_data(cache3),proj_psm)
		true
	end
end

@testset "transient: TimeCombination matrix projections" begin
	nt,rl,nr,nc = 8,3,4,5
	Φl = _rand_basis(nt,rl)
	Φr = _rand_basis(nt,nr)
	A = randn(nt,nc)

	c = CombinationOrder{1}(ThetaMethodCombination(0.1,0.35))

	# The 2-argument transient overload should reduce to steady projection.
	@test galerkin_projection(Φl,A,c) ≈ Φl' * A

	proj = galerkin_projection(Φl,A,Φr,c)
	θ = get_coefficients(c,nt)
	proj_ref = zeros(rl,nc,nr)

	@inbounds for i = 1:rl,k = 1:nc,j = 1:nr
		s = 0.0
		for γ in eachindex(θ)
			for α in 1:nt
				α + γ > nt + 1 && break
				s += θ[γ] * Φl[α + γ - 1,i] * A[α + γ - 1,k] * Φr[α,j]
			end
		end
		proj_ref[i,k,j] = s
	end

	@test proj ≈ proj_ref

	@test begin
		a = consecutive_param(randn(nt,nc))
		cache = consecutive_param(zeros(rl,nc))
		galerkin_projection!(cache,Φl,a,c)
		get_all_data(cache) ≈ galerkin_projection(Φl,a,c)
	end
end

@testset "multifield: VectorBlock projections" begin
	n1,n2,np = 7,6,4
	rl1,rl2 = 3,2

	Φl = BlockProjection(PODProjection.([_rand_basis(n1,rl1),_rand_basis(n2,rl2)]),Bool[true,true])
	a1 = consecutive_param(randn(n1,np))
	a2 = consecutive_param(randn(n2,np))
	a = ArrayBlock(Any[a1,a2],Bool[true,true])

	proj = galerkin_projection(Φl,a)
	@test get_basis(proj[1]) ≈ get_basis(galerkin_projection(Φl[1],a1))
	@test get_basis(proj[2]) ≈ get_basis(galerkin_projection(Φl[2],a2))

	@test begin
		cache = ArrayBlock(Any[consecutive_param(zeros(rl1,np)),consecutive_param(zeros(rl2,np))],Bool[true,true])
		galerkin_projection!(cache,Φl,a)
		get_all_data(cache[1]) ≈ get_basis(proj[1]) && get_all_data(cache[2]) ≈ get_basis(proj[2])
	end
end

@testset "multifield: MatrixBlock projections with ParamSparseMatrix" begin
	n1,n2,np = 7,6,3
	rl1,rl2 = 3,2
	rr1,rr2 = 2,4

	Φl = BlockProjection(PODProjection.([_rand_basis(n1,rl1),_rand_basis(n2,rl2)]),Bool[true,true])
	Φr = BlockProjection(PODProjection.([_rand_basis(n1,rr1),_rand_basis(n2,rr2)]),Bool[true,true])

	A = Array{Any}(undef,2,2)
	A[1,1] = _rand_param_sparse(n1,n1,np)
	A[1,2] = _rand_param_sparse(n1,n2,np)
	A[2,1] = _rand_param_sparse(n2,n1,np)
	A[2,2] = _rand_param_sparse(n2,n2,np)
	touched = fill(true,2,2)
	Ab = ArrayBlock(A,touched)

	proj = galerkin_projection(Φl,Ab,Φr)

	@test get_basis(proj[1,1]) ≈ get_basis(galerkin_projection(Φl[1],A[1,1],Φr[1]))
	@test get_basis(proj[1,2]) ≈ get_basis(galerkin_projection(Φl[1],A[1,2],Φr[2]))
	@test get_basis(proj[2,1]) ≈ get_basis(galerkin_projection(Φl[2],A[2,1],Φr[1]))
	@test get_basis(proj[2,2]) ≈ get_basis(galerkin_projection(Φl[2],A[2,2],Φr[2]))

	cache_data = Array{Any}(undef,2,2)
	cache_data[1,1] = consecutive_param(zeros(rl1,rr1,np))
	cache_data[1,2] = consecutive_param(zeros(rl1,rr2,np))
	cache_data[2,1] = consecutive_param(zeros(rl2,rr1,np))
	cache_data[2,2] = consecutive_param(zeros(rl2,rr2,np))
	cache = ArrayBlock(cache_data,touched)

	@test begin
		galerkin_projection!(cache,Φl,Ab,Φr)
		_check_cache_order_3d(get_all_data(cache[1,1]),get_basis(proj[1,1]))
		_check_cache_order_3d(get_all_data(cache[1,2]),get_basis(proj[1,2]))
		_check_cache_order_3d(get_all_data(cache[2,1]),get_basis(proj[2,1]))
		_check_cache_order_3d(get_all_data(cache[2,2]),get_basis(proj[2,2]))
		true
	end
end

end # module