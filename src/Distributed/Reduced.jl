# basis construction 

for T in (:GenericPMatrix,:DistributedSnapshots)
  @eval begin
    function RBSteady.tpod(red_style::ReductionStyle,A::$T)
      _method_of_snapshots_row(red_style,A,A'*A)
    end

    function RBSteady.tpod(red_style::ReductionStyle,A::$T,X::PSparseMatrix)
      _method_of_snapshots_row(red_style,A,A'*(X*A))
    end

    RBSteady.gram_schmidt(A::$T) = _qr!(A)
    RBSteady.gram_schmidt(A::$T,X::PSparseMatrix) = _qr!(X*A)
  end
end

function _method_of_snapshots_row(red_style::ReductionStyle,A,AA)
  _,Sr,Vr = RBSteady.truncated_svd(red_style,AA;issquare=true)
  Ur = _weighted_mul(A,Vr,Sr)
  return Ur,Sr,Vr
end

function _weighted_mul(_A,V,S)
  A = _gettr(_A)
  Ta = eltype(A)
  Tv = eltype(V)
  T = typeof(zero(Ta)*zero(Tv)+zero(Ta)*zero(Tv))
  U = GenericPArray{Matrix{T}}(undef,partition(axes(A,1)),axes(V,2))
  D = Diagonal(sqrt.(S).+eps())
  map(partition(U),partition(A),partition(axes(A,1))) do lU,lA,rA
    Uo = view(lU,own_to_local(rA),:)
    Ao = view(lA,own_to_local(rA),:)
    mul!(Uo,Ao,V,one(T),zero(T))
    rdiv!(Uo,D)
  end
  U
end

# projections

struct DistributedProjection <: Projection
  basis::AbstractMatrix
end

function RBSteady.Projection(basis::GenericPMatrix,s::DistributedSnapshots)
  DistributedProjection(basis)
end

function RBSteady.Projection(basis::GenericPMatrix,s::DistributedSparseSnapshots)
  basis′ = recast(basis,s)
  DistributedProjection(basis′)
end

RBSteady.get_basis(a::DistributedProjection) = a.basis
row_partition(a::DistributedProjection) = row_partition(a.basis)
col_partition(a::DistributedProjection) = col_partition(a.basis)

function RBSteady.union_bases(a::DistributedProjection,b::DistributedProjection,args...) 
  union_bases(a,get_basis(b),args...)
end

function RBSteady.union_bases(a::DistributedProjection,basis_b::AbstractMatrix,args...)
  basis_a = get_basis(a)
  basis_ab = gram_schmidt(basis_b,basis_a,args...)
  DistributedProjection(basis_ab)
end

function RBSteady.galerkin_projection(a::DistributedProjection,b::DistributedProjection)
  lb̂ = map(local_views(a),local_views(b)) do ai,bi
    galerkin_projection(get_basis(ai),get_basis(bi))
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

function RBSteady.galerkin_projection(a::DistributedProjection,b::DistributedProjection,c::DistributedProjection,args...)
  lb̂ = map(local_views(a),local_views(b),local_views(c)) do ai,bi,ci
    galerkin_projection(get_basis(ai),get_basis(bi),get_basis(ci),args...)
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

function RBSteady._allocate_projection(red::Reduction,s::DistributedBlockSnapshots{N}) where N
  T = DistributedProjection
  block_basis = Array{T,N}(undef,size(s))
  BlockProjection(block_basis,s.touched)
end

function GridapDistributed.local_views(a::DistributedProjection)
  map(local_views(a.basis)) do basis
    PODProjection(basis)
  end
end

function GridapDistributed.local_views(a::NormedProjection)
  map(local_views(a.projection),local_views(a.norm_matrix)) do proj,norm
    NormedProjection(proj,norm)
  end
end

# reduced basis spaces

const DistributedRBSpace{S<:DistributedFESpace} = RBSpace{S}

function GridapDistributed.local_views(r::DistributedRBSpace)
  map(local_views(r.space),local_views(r.subspace)) do space,subspace
    RBSpace(space,subspace)
  end
end

# hyper-reduction

struct DistributedInterpolation{A} <: Interpolation
  interps::A
end

GridapDistributed.local_views(a::DistributedInterpolation) = local_views(a.interps)

function RBSteady.reduced_triangulation(trian::DistributedTriangulation,a::HRProjection)
  reduced_triangulation(trian,get_interpolation(a))
end

function RBSteady.reduced_triangulation(trian::DistributedTriangulation,a::DistributedInterpolation)
  model = get_background_model(trian)
  trians = map(local_views(trian),local_views(a)) do ti,ai
    reduced_triangulation(ti,ai)
  end
  DistributedTriangulation(trians,model)
end

function RBSteady.reduced_triangulation(trian::DistributedTriangulation,a::EmptyInterpolation)
  model = get_background_model(trian)
  trians = map(local_views(trian)) do ti
    reduced_triangulation(ti,a)
  end
  DistributedTriangulation(trians,model)
end

function RBSteady.Interpolation(red::NoHyperReduction,trian::DistributedTriangulation)
  interps = map(local_views(trian)) do ti
    Interpolation(red,ti)
  end
  DistributedInterpolation(interps)
end

for (T,f) in zip((:DEIMHyperReduction,:SOPTHyperReduction),(:DEIM,:SOPT))
  @eval begin
    function RBSteady.Interpolation(
      red::$T,
      basis::DistributedProjection,
      trian::DistributedTriangulation,
      test::DistributedRBSpace
      )

      interps = map(local_views(basis),local_views(trian),local_views(test)) do bi,ti,testi
        rows,interp = $f(bi)
        factor = lu(interp)
        domain = IntegrationDomain(ti,testi,rows)
        GreedyInterpolation(factor,domain)
      end
      DistributedInterpolation(interps)
    end

    function RBSteady.Interpolation(
      red::$T,
      basis::DistributedProjection,
      trian::DistributedTriangulation,
      trial::DistributedRBSpace,
      test::DistributedRBSpace
      )

      interps = map(local_views(basis),local_views(trian),local_views(trial),local_views(test)) do bi,ti,triali,testi
        (rows,cols),interp = $f(bi)
        factor = lu(interp)
        domain = IntegrationDomain(ti,triali,testi,rows,cols)
        GreedyInterpolation(factor,domain)
      end
      DistributedInterpolation(interps)
    end
  end
end

function RBSteady.Interpolation(
  red::RBFHyperReduction,
  basis::DistributedProjection,
  s::DistributedSnapshots
  )

  interps = map(local_views(basis),local_views(s)) do bi,si
    Interpolation(red,bi,si)
  end
  DistributedInterpolation(interps)
end