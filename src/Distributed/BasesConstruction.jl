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