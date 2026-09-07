function _qr!(a::GenericPMatrix)
  m,n = size(a)
  piv = Vector(UnitRange{BlasInt}(1,n))
  τ = Vector{eltype(a)}(undef,min(m,n))
  for j = 1:min(m,n)
    j′ = _indmaxcol(a,j:m,j:n) + j - 1
    if j′ != j
      tmpp = piv[j′]
      piv[j′] = piv[j]
      piv[j] = tmpp
      _swapcols!(a,j,j′)
    end
    τj = _reflector!(a,j:m,j)
    τ[j] = τj
    _reflector_apply!(a,τj,j:m,j+1:n)
  end
  consistent!(a) |> wait
  return a
end

function _swapcols!(a,j,j′)
  map(own_values(a)) do a
    for i = axes(a,1)
      tmp = a[i,j′]
      a[i,j′] = a[i,j]
      a[i,j] = tmp
    end
  end
  a
end

function _indmaxcol(a,rows=1:size(a,1),cols=1:size(a,2))
  mm = _colnorm(a,rows,cols[1])
  ii = 1
  for i = 2:length(cols)
    mi = _colnorm(a,rows,cols[i])
    if mi > mm
      mm = mi
      ii = i
    end
  end
  return ii
end

function _colnorm(a,rows,col)
  contribs = map(own_values(a),row_partition(a)) do vals,row_idxs
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

function _fetch_value(a,global_row,global_col)
  v = ()
  map(own_values(a),row_partition(a)) do vals,row_idxs
    g2o = global_to_own(row_idxs)
    lr = g2o[global_row]
    if lr > 0
      v = (v...,vals[lr,global_col])
    end
  end
  @check length(v) == 1
  first(v)
end

function _set_value_single!(a,global_row,global_col,val)
  map(own_values(a),row_partition(a)) do vals,row_idxs
    g2o = global_to_own(row_idxs)
    lr = g2o[global_row]
    if lr > 0
      vals[lr,global_col] = val
    end
  end
  a
end

function _set_values!(a,global_rows,global_col,global_vals)
  map(own_values(a),row_partition(a)) do vals,row_idxs
    g2o = global_to_own(row_idxs)
    for global_row in global_rows
      lr = g2o[global_row]
      if lr > 0
        vals[lr,global_col] = global_vals[global_row]
      end
    end
  end
  a
end

function _div_col_range!(a,rows,col,d)
  map(own_values(a),row_partition(a)) do vals,row_idxs
    o2g = own_to_global(row_idxs)
    for (oi,gi) in enumerate(o2g)
      gi < first(rows) && continue
      gi > last(rows) && continue
      vals[oi,col] /= d
    end
  end
  a
end

function _reflector!(a,rows=1:size(a,1),col=1)
  n = length(rows)
  n == 0 && return zero(eltype(a))
  T = eltype(a)
  ξ1 = _fetch_value(a,first(rows),col)
  normu = _colnorm(a,rows,col)
  if iszero(normu)
    return zero(T)
  end
  ν = T(copysign(normu,real(ξ1)))
  v1 = ξ1 + ν
  τ = v1 / ν
  _set_value_single!(a,first(rows),col,-ν)
  n > 1 && _div_col_range!(a,rows[2:end],col,v1)
  return τ
end

function _reflector_apply!(a,τ,rows,cols)
  isempty(rows) && return a
  isempty(cols) && return a
  T = eltype(a)
  refl_col = first(rows)
  ncols = length(cols)
  partial_w = map(own_values(a),row_partition(a)) do vals,row_idxs
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
  map(own_values(a),row_partition(a)) do vals,row_idxs
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
  a
end
