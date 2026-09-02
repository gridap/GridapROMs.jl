# basis construction 

for T in (:GenericPMatrix,:DistributedSnapshots)
  @eval begin
    function RBSteady.tpod(red_style::ReductionStyle,A::$T,X::PSparseMatrix)
      _method_of_snapshots_row(red_style,A,A'*(X*A))
    end

    function RBSteady.tpod(red_style::ReductionStyle,A::$T)
      _method_of_snapshots_row(red_style,A,A'*A)
    end
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

struct DistributedProjection{P,A} <: Projection
  projections::A
  function DistributedProjection(projections::A) where {P<:Projection,A<:AbstractArray{<:P}}
    new{P,A}(projections)
  end
end

function RBSteady.Projection(a::AbstractArray{<:Projection})
  DistributedProjection(a)
end

function RBSteady.NormedProjection(a::AbstractArray{<:Projection},b::PSparseMatrix)
  projections = map(local_values(a),local_values(b)) do a,b
    RBSteady.NormedProjection(a,b)
  end
  DistributedProjection(projections)
end

const DistributedBlockProjection{A<:DistributedProjection,N} = BlockProjection{A,N}

function RBSteady.projection(red::Reduction,s::DistributedBlockSnapshots)
  basis = _allocate_projection(red,s)
  for i in eachindex(basis)
    if basis.touched[i]
      basis[i] = projection(red,s[i])
    end
  end
  return basis
end

function RBSteady.projection(red::Reduction,s::DistributedBlockSnapshots,X::BlockPMatrix)
  basis = RBSteady._allocate_projection(red,s,X)
  for i in eachindex(basis)
    if basis.touched[i]
      basis[i] = projection(red,s[i,X[Block(i,i)]])
    end
  end
  return basis
end

function RBSteady._allocate_projection(red::Reduction,s::DistributedBlockSnapshots{N},args...) where N
  T = DistributedProjection
  block_basis = Array{T,N}(undef,size(s))
  BlockProjection(block_basis,s.touched)
end

# reduced basis spaces

const DistributedRBSpace = RBSpace{DistributedFESpace}
const DistributedSingleFieldRBSpace = RBSpace{DistributedSingleFieldFESpace}
const DistributedMultiFieldRBSpace = RBSpace{DistributedMultiFieldFESpace}

# projections (distributed)

function RBSteady.Projection(basis::GenericPArray,s::DistributedSnapshots)
  projections = map(partition(basis),partition(axes(basis,1))) do lb,rp
    PODProjection(copy(view(lb,own_to_local(rp),:)))
  end
  DistributedProjection(projections)
end

RBSteady.num_reduced_dofs(a::DistributedProjection) = num_reduced_dofs(getany(a.projections))
RBSteady.projection_eltype(a::DistributedProjection) = projection_eltype(getany(a.projections))

function RBSteady.galerkin_projection(a::DistributedProjection,b::DistributedProjection)
  G = map(a.projections,b.projections) do ai,bi
    galerkin_projection(get_basis(ai),get_basis(bi))
  end
  b̂ = reduce(+,G)
  return ReducedProjection(b̂)
end

function RBSteady.galerkin_projection(a::DistributedProjection,b::DistributedProjection,c::DistributedProjection)
  G = map(a.projections,b.projections,c.projections) do ai,bi,ci
    galerkin_projection(get_basis(ai),get_basis(bi),get_basis(ci))
  end
  b̂ = reduce(+,G)
  return ReducedProjection(b̂)
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

function RBSteady.reduced_triangulation(trian::DistributedTriangulation,a::RBSteady.EmptyInterpolation)
  model = get_background_model(trian)
  trians = map(local_views(trian)) do ti
    reduced_triangulation(ti,a)
  end
  DistributedTriangulation(trians,model)
end

function RBSteady.Interpolation(red::RBSteady.NoHyperReduction,trian::DistributedTriangulation)
  interps = map(local_views(trian)) do ti
    RBSteady.Interpolation(red,ti)
  end
  DistributedInterpolation(interps)
end

for (T,f) in zip((:DEIMHyperReduction,:SOPTHyperReduction),(:DEIM,:SOPT))
  @eval begin
    function RBSteady.Interpolation(
      red::RBSteady.$T,
      basis::DistributedProjection,
      trian::DistributedTriangulation,
      test::DistributedRBSpace
      )

      interps = map(basis.projections,local_views(trian),local_views(test)) do bi,ti,testi
        RBSteady.Interpolation(red,bi,ti,testi)
      end
      DistributedInterpolation(interps)
    end

    function RBSteady.Interpolation(
      red::RBSteady.$T,
      basis::DistributedProjection,
      trian::DistributedTriangulation,
      trial::DistributedRBSpace,
      test::DistributedRBSpace
      )

      interps = map(basis.projections,local_views(trian),local_views(trial),local_views(test)) do bi,ti,triali,testi
        RBSteady.Interpolation(red,bi,ti,triali,testi)
      end
      DistributedInterpolation(interps)
    end
  end
end

function RBSteady.Interpolation(
  red::RBSteady.RBFHyperReduction,
  basis::DistributedProjection,
  s::DistributedSnapshots
  )

  interps = map(basis.projections,local_views(s)) do bi,si
    RBSteady.Interpolation(red,bi,si)
  end
  DistributedInterpolation(interps)
end