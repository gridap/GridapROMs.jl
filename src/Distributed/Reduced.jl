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

# hyper-reduction

function RBSteady.HRProjection(
  red::Reduction,
  s::DistributedSnapshots,
  trian::DistributedTriangulation,
  test::DistributedRBSpace
  )

  map(local_views(s),local_views(trian),local_views(test)) do s,trian,test
    HRProjection(red,s,trian,test)
  end
end

function RBSteady.HRProjection(
  red::Reduction,
  s::DistributedSnapshots,
  trian::DistributedTriangulation,
  trial::DistributedRBSpace,
  test::DistributedRBSpace
  )

  map(local_views(s),local_views(trian),local_views(trial),local_views(test)) do s,trian,trial,test
    HRProjection(red,s,trian,trial,test)
  end
end