function RBSteady.gram_solver(X::PSparseMatrix)
  solver = CGSolver(JacobiLinearSolver();maxiter=100,atol=1e-14,rtol=1e-10)
  ss = symbolic_setup(solver,X)
  numerical_setup(ss,X)
end

function LinearAlgebra.ldiv!(S::GenericPMatrix,ns::LinearSolvers.CGNumericalSetup,A::GenericPMatrix)
  mat = _get_matrix(ns)
  S′ = S
  if !PartitionedArrays.matching_ghost_indices(axes(S,1),axes(mat,2))
    S′ = _change_layout(S,partition(axes(mat,2)))
  end
  if !PartitionedArrays.matching_ghost_indices(axes(mat,2),axes(A,1))
    A = _change_layout(A,partition(axes(mat,2)))
  end
  consistent!(A) |> wait
  map(own_values(S′)) do s
    fill!(s,zero(eltype(s)))
  end
  for i in axes(A,2)
    Si = _get_column(S′,i)
    Ai = _get_column(A,i)
    solve!(Si,ns,Ai)
  end
  if S !== S′
    map(own_values(S),own_values(S′)) do s,s′
      copyto!(s,s′)
    end
  end
  S
end

function RBSteady.gram_schmidt(A::AbstractMatrix,ns::NumericalSetup;tol=1e-10)
  Q,R, = weighted_qr!(A,_get_matrix(ns))
  rank = something(findlast(abs.(diag(R)) .> tol),0)
  Qr = RBSteady._truncate_col!(Q,rank)
  return Qr
end

function weighted_qr!(A::GenericPMatrix,X::PSparseMatrix)
  m,n = size(A)
  T = eltype(A)
  XA = X*A
  p = collect(1:n)
  R = zeros(T,n,n)
  colnorms2 = [real(_wdot(A,XA,j,j)) for j in 1:n]
  for j in 1:min(m,n)
    j′ = argmax(view(colnorms2,j:n)) + j - 1
    if j′ != j
      tmp = p[j′]
      p[j′] = p[j]
      p[j] = tmp
      tmp = colnorms2[j′]
      colnorms2[j′] = colnorms2[j]
      colnorms2[j] = tmp
      _swapcols!(A,j,j′)
      _swapcols!(XA,j,j′)
    end
    normj = sqrt(max(colnorms2[j],zero(real(T))))
    R[j,j] = normj
    iszero(normj) && continue
    _wscale_col!(A,inv(normj),j)
    _wscale_col!(XA,inv(normj),j)
    for k in j+1:n
      rjk = _wdot(A,XA,j,k)
      R[j,k] = rjk
      _waxpy_col!(A,-rjk,j,k)
      _waxpy_col!(XA,-rjk,j,k)
      colnorms2[k] = real(_wdot(A,XA,k,k))
    end
  end
  consistent!(A) |> wait
  return A,R,p
end

function _wdot(A::GenericPMatrix,XA::GenericPMatrix,i,j)
  contribs = map(own_values(A),own_values(XA)) do Av,XAv
    s = zero(promote_type(eltype(Av),eltype(XAv)))
    for oi in axes(Av,1)
      s += conj(Av[oi,i]) * XAv[oi,j]
    end
    s
  end
  reduce(+,contribs)
end

function _wscale_col!(A::GenericPMatrix,α,j)
  map(own_values(A)) do Av
    for oi in axes(Av,1)
      Av[oi,j] *= α
    end
  end
  A
end

function _waxpy_col!(A::GenericPMatrix,α,i,j)
  map(own_values(A)) do Av
    for oi in axes(Av,1)
      Av[oi,j] += α*Av[oi,i]
    end
  end
  A
end

struct PQR{A,B,C}
  Q::A
  R::B
  p::C
end

Base.iterate(p::PQR,i...) = iterate((p.Q,p.R,p.p),i...)

function LinearAlgebra.qr!(A::GenericPMatrix,::NoPivot)
  m,n = size(A)
  p = Vector(UnitRange{BlasInt}(1,n))
  τ = Vector{eltype(A)}(undef,min(m,n))
  for j = 1:min(m,n)
    τj = _reflector!(A,j:m,j)
    τ[j] = τj
    _reflector_apply!(A,τj,j:m,j+1:n)
  end
  Q = _get_Q(A,τ,m,n)
  R = _get_R(A,n)
  return PQR(Q,R,p)
end

function LinearAlgebra.qr!(A::GenericPMatrix,::ColumnNorm)
  m,n = size(A)
  p = Vector(UnitRange{BlasInt}(1,n))
  τ = Vector{eltype(A)}(undef,min(m,n))
  for j = 1:min(m,n)
    j′ = _indmaxcol(A,j:m,j:n) + j - 1
    if j′ != j
      tmp = p[j′]
      p[j′] = p[j]
      p[j] = tmp
      _swapcols!(A,j,j′)
    end
    τj = _reflector!(A,j:m,j)
    τ[j] = τj
    _reflector_apply!(A,τj,j:m,j+1:n)
  end
  Q = _get_Q(A,τ,m,n)
  R = _get_R(A,n)
  return PQR(Q,R,p)
end

function RBTransient.first_unfold(A::DistributedSnapshots)
  values = map(local_values(A)) do A
    RBTransient.first_unfold(A)
  end
  GenericPArray(values,flat_row_partition(A))
end

# utils 

function RBSteady._is_rectangular(A::Union{GenericPMatrix,DistributedSnapshots};kwargs...)
  true
end

function RBSteady._reduce_columns(A::Union{GenericPMatrix,DistributedSnapshots};kwargs...)
  true
end

function RBSteady._weighted_mul_row(A::Union{GenericPMatrix,DistributedSnapshots},V,S)
  Ta = eltype(A)
  Tv = eltype(V)
  T = typeof(zero(Ta)*zero(Tv)+zero(Ta)*zero(Tv))
  U = GenericPArray{Matrix{T}}(undef,flat_row_partition(A),axes(V,2))
  D = Diagonal(S.+eps())
  map(own_values(U),own_values(A)) do Uo,Ao
    mul!(Uo,Ao,V)
    rdiv!(Uo,D)
  end
  consistent!(U) |> wait
  U
end

function RBSteady._truncate_col!(A::GenericPMatrix,rank)
  rank == size(A,2) && return A
  values = map(local_values(A)) do A
    RBSteady._truncate_col!(A,rank)
  end
  GenericPArray(values,partition(axes(A,1)))
end

_get_matrix(ns) = @abstractmethod
_get_matrix(ns::LinearSolvers.CGNumericalSetup) = ns.mat

function _get_column(A::GenericPMatrix,i::Integer)
  vals,cache = map(A.array_partition,A.cache) do values,cache
    view(values,:,i),param_getindex(cache,i)
  end |> tuple_of_arrays
  PVector(vals,A.index_partition,cache)
end

function _get_Q(A::GenericPMatrix,τ,m,n)
  T = eltype(A)
  Q_parts = map(partition(A)) do local_mat
    zeros(T,size(local_mat,1),n)
  end
  Q = GenericPArray(Q_parts,A.index_partition)
  for j in 1:n
    _set_value!(Q,one(T),j,j)
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
  reduce(+,parts)
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
    s = zero(real(eltype(vals)))
    for (oi,gi) in enumerate(o2g)
      gi < first(rows) && continue
      gi > last(rows) && continue
      s += abs2(vals[oi,col])
    end
    s
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
  iszero(normu) && return zero(T)
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
  w = reduce(+,partial_w)
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
  w = reduce(+,partial_w)
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