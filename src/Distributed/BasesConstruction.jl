for T in (:GenericPMatrix,:DistributedSnapshots)
  @eval begin
    function RBSteady.tpod(red_style::ReductionStyle,A::$T)
      _method_of_snapshots_row(red_style,A,A'*A)
    end

    function RBSteady.tpod(red_style::ReductionStyle,A::$T,X::PSparseMatrix)
      _method_of_snapshots_row(red_style,A,A'*(X*A))
    end

    RBSteady.gram_schmidt(A::$T,X::PSparseMatrix) = gram_schmidt(X*A)
  end
end

function LinearAlgebra.qr!(A::GenericPMatrix,::NoPivot)
  m,n = size(A)
  τ = Vector{eltype(A)}(undef,min(m,n))
  for j = 1:min(m,n)
    τj = _reflector!(A,j:m,j)
    τ[j] = τj
    _reflector_apply!(A,τj,j:m,j+1:n)
  end
  Q = _get_Q(A,τ,m,n)
  R = _get_R(A,n)
  return Q,R,piv
end

function LinearAlgebra.qr!(A::GenericPMatrix,::ColumnNorm)
  m,n = size(A)
  piv = Vector(UnitRange{BlasInt}(1,n))
  τ = Vector{eltype(A)}(undef,min(m,n))
  for j = 1:min(m,n)
    j′ = _indmaxcol(A,j:m,j:n) + j - 1
    if j′ != j
      tmpp = piv[j′]
      piv[j′] = piv[j]
      piv[j] = tmpp
      _swapcols!(A,j,j′)
    end
    τj = _reflector!(A,j:m,j)
    τ[j] = τj
    _reflector_apply!(A,τj,j:m,j+1:n)
  end
  Q = _get_Q(A,τ,m,n)
  R = _get_R(A,n)
  return Q,R,piv
end

function RBTransient.first_unfold(A::GenericPArray{T,3}) where T
  values = map(local_values(A)) do A
    RBTransient.first_unfold(A)
  end
  GenericPArray(values,flat_row_partition(A))
end

# utils 

function _method_of_snapshots_row(red_style::ReductionStyle,A,AA)
  _,Sr,Vr = RBSteady.truncated_svd(red_style,AA;issquare=true)
  Ur = _weighted_mul(A,Vr,Sr)
  return Ur,Sr,Vr
end

function _weighted_mul(A,V,S)
  Ta = eltype(A)
  Tv = eltype(V)
  T = typeof(zero(Ta)*zero(Tv)+zero(Ta)*zero(Tv))
  U = GenericPArray{Matrix{T}}(undef,flat_row_partition(A),axes(V,2))
  D = Diagonal(S.+eps())
  map(own_values(U),own_values(A)) do Uo,Ao
    mul!(Uo,Ao,V)
    rdiv!(Uo,D)
  end
  consistent!(U) |> fetch
  U
end

function _get_Q(A::GenericPMatrix,τ,m,n)
  T = eltype(A)
  Q_parts = map(partition(A)) do local_mat
    zeros(T,size(local_mat,1),n)
  end
  Q = GenericPArray(Q_parts,A.index_partition)
  for j in 1:n
    _set_value_single!(Q,j,j,one(T))
  end
  for j in n:-1:1
    _reflector_apply_cross!(Q,A,τ[j],j:m,j:n)
  end
  consistent!(Q) |> wait
  Q
end

function _get_R(A::GenericPMatrix,n)
  T = eltype(A)
  parts = map(own_values(A),row_partition(A)) do vals,row_idxs
    o2g = own_to_global(row_idxs)
    R = zeros(T,n,n)
    for (oi,gi) in enumerate(o2g)
      gi > n && continue
      for j in gi:n
        R[gi,j] = vals[oi,j]
      end
    end
    R
  end
  reduce(+,parts;init=zeros(T,n,n))
end

function _swapcols!(A,j,j′)
  map(own_values(A)) do A
    for i = axes(A,1)
      tmp = A[i,j′]
      A[i,j′] = A[i,j]
      A[i,j] = tmp
    end
  end
  A
end

function _indmaxcol(A,rows=1:size(A,1),cols=1:size(A,2))
  mm = _colnorm(A,rows,cols[1])
  ii = 1
  for i = 2:length(cols)
    mi = _colnorm(A,rows,cols[i])
    if mi > mm
      mm = mi
      ii = i
    end
  end
  return ii
end

function _colnorm(A,rows,col)
  contribs = map(own_values(A),row_partition(A)) do vals,row_idxs
    o2g = own_to_global(row_idxs)
    local_sum = zero(real(eltype(vals)))
    for (oi,gi) in enumerate(o2g)
      gi < first(rows) && continue
      gi > last(rows) && continue
      local_sum += abs2(vals[oi,col])
    end
    local_sum
  end
  sqrt(reduce(+,contribs;init=zero(eltype(contribs))))
end

function _get_value(A,global_row,global_col)
  v = ()
  map(own_values(A),row_partition(A)) do vals,row_idxs
    g2o = global_to_own(row_idxs)
    lr = g2o[global_row]
    if lr > 0
      v = (v...,vals[lr,global_col])
    end
  end
  @check length(v) == 1
  first(v)
end

function _set_value!(A,val,global_row,global_col)
  map(own_values(A),row_partition(A)) do vals,row_idxs
    g2o = global_to_own(row_idxs)
    lr = g2o[global_row]
    if lr > 0
      vals[lr,global_col] = val
    end
  end
  A
end

function _div_col_range!(A,val,rows,col)
  map(own_values(A),row_partition(A)) do vals,row_idxs
    o2g = own_to_global(row_idxs)
    for (oi,gi) in enumerate(o2g)
      gi < first(rows) && continue
      gi > last(rows) && continue
      vals[oi,col] /= val
    end
  end
  A
end

function _reflector!(A,rows=1:size(A,1),col=1)
  n = length(rows)
  n == 0 && return zero(eltype(A))
  T = eltype(A)
  ξ1 = _get_value(A,first(rows),col)
  normu = _colnorm(A,rows,col)
  if iszero(normu)
    return zero(T)
  end
  ν = T(copysign(normu,real(ξ1)))
  v = ξ1 + ν
  τ = v / ν
  _set_value!(A,-ν,first(rows),col)
  n > 1 && _div_col_range!(A,v,rows[2:end],col)
  return τ
end

function _reflector_apply!(A,τ,rows,cols)
  isempty(rows) && return A
  isempty(cols) && return A
  T = eltype(A)
  refl_col = first(rows)
  ncols = length(cols)
  partial_w = map(own_values(A),row_partition(A)) do vals,row_idxs
    o2g = own_to_global(row_idxs)
    g2o = global_to_own(row_idxs)
    w = zeros(T,ncols)
    lr1 = g2o[first(rows)]
    if lr1 > 0
      for (jj,k) in enumerate(cols)
        w[jj] += vals[lr1,k]
      end
    end
    for (oi,gi) in enumerate(o2g)
      gi <= first(rows) && continue
      gi > last(rows) && continue
      vi = conj(vals[oi,refl_col])
      for (jj,k) in enumerate(cols)
        w[jj] += vi * vals[oi,k]
      end
    end
    w
  end
  w = reduce(+,partial_w;init=zeros(T,ncols))
  vAk = conj(τ) .* w
  map(own_values(A),row_partition(A)) do vals,row_idxs
    o2g = own_to_global(row_idxs)
    g2o = global_to_own(row_idxs)
    lr1 = g2o[first(rows)]
    if lr1 > 0
      for (jj,k) in enumerate(cols)
        vals[lr1,k] -= vAk[jj]
      end
    end
    for (oi,gi) in enumerate(o2g)
      gi <= first(rows) && continue
      gi > last(rows) && continue
      vi = vals[oi,refl_col]
      for (jj,k) in enumerate(cols)
        vals[oi,k] -= vi * vAk[jj]
      end
    end
  end
  A
end

function _reflector_apply_cross!(Q::GenericPMatrix,A::GenericPMatrix,τ,rows,cols)
  isempty(rows) && return Q
  isempty(cols) && return Q
  T = eltype(Q)
  ncols = length(cols)
  partial_w = map(own_values(Q),own_values(A),row_partition(Q)) do Qv,av,row_idxs
    o2g = own_to_global(row_idxs)
    g2o = global_to_own(row_idxs)
    w = zeros(T,ncols)
    lr1 = g2o[first(rows)]
    if lr1 > 0
      for (jj,k) in enumerate(cols)
        w[jj] += Qv[lr1,k]
      end
    end
    for (oi,gi) in enumerate(o2g)
      gi <= first(rows) && continue
      gi > last(rows) && continue
      vi = conj(av[oi,first(rows)])
      for (jj,k) in enumerate(cols)
        w[jj] += vi * Qv[oi,k]
      end
    end
    w
  end
  w = reduce(+,partial_w;init=zeros(T,ncols))
  vAk = conj(τ) .* w
  map(own_values(Q),own_values(A),row_partition(Q)) do Qv,av,row_idxs
    o2g = own_to_global(row_idxs)
    g2o = global_to_own(row_idxs)
    lr1 = g2o[first(rows)]
    if lr1 > 0
      for (jj,k) in enumerate(cols)
        Qv[lr1,k] -= vAk[jj]
      end
    end
    for (oi,gi) in enumerate(o2g)
      gi <= first(rows) && continue
      gi > last(rows) && continue
      vi = av[oi,first(rows)]
      for (jj,k) in enumerate(cols)
        Qv[oi,k] -= vi * vAk[jj]
      end
    end
  end
  Q
end