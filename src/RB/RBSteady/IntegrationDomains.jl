function empirical_interpolation(basis::AbstractMatrix{T}) where T
  m,n = size(basis)
  res = zeros(T,m)
  I = zeros(Int,n)
  basisI = zeros(T,n,n)
  @inbounds @views begin
    @. res = basis[:,1]
    I[1] = argmax(abs.(res))
    basisI[1,:] = basis[I[1],:]
    for l = 2:n
      U = basis[:,1:l-1]
      P = I[1:l-1]
      PᵀU = U[P,:]
      uₗ = basis[:,l]
      Pᵀuₗ = uₗ[P,:]
      c = vec(PᵀU \ Pᵀuₗ)
      mul!(res,U,c)
      @. res = uₗ - res
      I[l] = argmax(abs.(res))
      basisI[l,:] = basis[I[l],:]
    end
  end
  return I,basisI
end

function s_opt(basis::AbstractMatrix{T}) where T
  m,n = size(basis)
  I = zeros(Int,n)
  basisI = zeros(T,n,n)
  @inbounds @views begin
    I[1] = argmax(abs.(basis[:,1]))
    basisI[1,:] = basis[I[1],:]
    for l in 2:n
      U = basis[:,1:l]
      P = I[1:l-1]
      PᵀU = U[P,:]
      colnorms2 = vec(sum(abs2,PᵀU;dims=1))
      Il = best_s_opt_index(U,P,colnorms2)
      @check Il > 0
      I[l] = Il
      basisI[l,:] = basis[Il,:]
    end
  end
  return I,basisI
end

function best_s_opt_index(U,P,colnorms2)
  m,n = size(U)
  best_i = 0
  best_logS = -Inf
  @inbounds @views for l in 1:m
    l ∈ P && continue
    q = U[l,:]
    A = U[vcat(P,l),:]
    G = A'*A
    logdet_plus = robust_logdet(G)
    colnorms2_plus = colnorms2 .+ abs2.(q)
    # S(A) in log form: (1/n)*( 0.5*logdet - 0.5*Σ log colnorms2 )
    logS = (0.5/n)*(logdet_plus - sum(log,colnorms2_plus))
    if logS > best_logS
      best_logS = logS
      best_i = l
    end
  end
  return best_i
end

function robust_logdet(A::AbstractMatrix{T}) where T
  try
    H = cholesky(Symmetric(A);check=true)
    2.0*sum(log,diag(H.L))
  catch
    _,Σ,_ = svd(A)
    tol = maximum(Σ)*eps(T)+eps(T)
    sum(log,clamp.(Σ,tol,Inf))
  end
end

for f in (:empirical_interpolation,:s_opt)
  @eval begin
    function $f(A::ParamSparseMatrix)
      I,AI = $f(A.data)
      R′,C′ = recast_split_indices(I,param_getindex(A,1))
      return (R′,C′),AI
    end
  end
end

"""
    get_rows_to_cells(
      cell_row_ids::AbstractArray,
      rows::AbstractVector
      ) -> AbstractVector

Returns the list of cells containing the row ids `rows`
"""
function get_rows_to_cells(
  cell_row_ids::AbstractArray,
  rows::AbstractVector
  )

  ncells = length(cell_row_ids)
  cells = fill(false,ncells)
  cache = array_cache(cell_row_ids)
  for cell in 1:ncells
    cellrows = getindex!(cache,cell_row_ids,cell)
    cells[cell] = _isrow(cellrows,rows)
  end
  Int32.(findall(cells))
end


"""
    get_rowcols_to_cells(
      cell_row_ids::AbstractArray,
      cell_col_ids::AbstractArray,
      rows::AbstractVector,
      cols::AbstractVector) -> AbstractVector

Returns the list of cells containing the row ids `rows` and the col ids `cols`
"""
function get_rowcols_to_cells(
  cell_row_ids::AbstractArray,
  cell_col_ids::AbstractArray,
  rows::AbstractVector,
  cols::AbstractVector
  )

  @assert length(cell_row_ids) == length(cell_col_ids)
  ncells = length(cell_row_ids)
  cells = fill(false,ncells)
  rowcache = array_cache(cell_row_ids)
  colcache = array_cache(cell_col_ids)
  for cell in 1:ncells
    cellrows = getindex!(rowcache,cell_row_ids,cell)
    cellcols = getindex!(colcache,cell_col_ids,cell)
    cells[cell] = _isrowcol(cellrows,cellcols,rows,cols)
  end
  Int32.(findall(cells))
end

struct CellsToIDofsMap{A,B,C} <: Map
  cell_dof_ids::A
  cells::B
  rows::C
end

function Arrays.return_cache(k::CellsToIDofsMap,icell::Int)
  dof_cache = array_cache(k.cell_dof_ids)
  rows = getindex!(dof_cache,k.cell_dof_ids,icell)
  idof_cache = _cache(rows)
  unwrap_cache = return_cache(unwrap_and_setsize!,idof_cache,rows)
  return dof_cache,idof_cache,unwrap_cache
end

function Arrays.evaluate!(cache,k::CellsToIDofsMap,icell::Int)
  dof_cache,idof_cache,unwrap_cache = cache
  cell = k.cells[icell]
  celldofs = getindex!(dof_cache,k.cell_dof_ids,cell)
  a = evaluate!(unwrap_cache,unwrap_and_setsize!,idof_cache,celldofs)
  _evaluate!(a,celldofs,k.rows)
end

function get_cells_to_idofs(cell_dof_ids,cells,rows)
  k = CellsToIDofsMap(cell_dof_ids,cells,rows)
  lazy_map(k,1:length(cells))
end

"""
    abstract type IntegrationDomain end

Type representing the set of interpolation rows of a `Projection` subjected
to a EIM approximation.
Subtypes:
- [`VectorDomain`](@ref)
- [`MatrixDomain`](@ref)
- [`TransientIntegrationDomain`](@ref)
"""
abstract type IntegrationDomain end

get_integration_cells(i::IntegrationDomain,args...) = @abstractmethod
get_cell_irows(i::IntegrationDomain) = @abstractmethod
get_cell_icols(i::IntegrationDomain) = @notimplemented

function get_owned_icells(i::IntegrationDomain,cells::AbstractVector)::Vector{Int}
  cellsi = get_integration_cells(i)
  filter(!isnothing,indexin(cellsi,cells))
end

function get_integration_cells(i::IntegrationDomain,trian::Triangulation)
  get_integration_cells(i)
end

function get_integration_cells(i::IntegrationDomain,trian::AppendedTriangulation)
  cells = get_integration_cells(i)
  parent = get_parent(trian)
  parent_ncellsa = num_cells(parent.a)
  cell_to_istriana = zeros(Bool,num_cells(trian))
  for (icell,cell) in enumerate(cells)
    cell_to_istriana[icell] = cell <= parent_ncellsa
  end
  ncellsa = length(findall(cell_to_istriana))
  a = Vector{eltype(cells)}(undef,ncellsa)
  b = Vector{eltype(cells)}(undef,length(cells)-ncellsa)
  na = 0
  nb = 0
  for (icell,cell) in enumerate(cells)
    if cell_to_istriana[icell]
      na += 1
      a[na] = cell
    else
      nb += 1
      b[nb] = cell
    end
  end
  lazy_append(a,b)
end

function IntegrationDomain(args...)
  @abstractmethod
end

"""
    struct VectorDomain{A,B} <: IntegrationDomain
      cells::Vector{Int32}
      cell_irows::A
      rows::B
    end

Integration domain for a projection operator in a steady problem
"""
struct VectorDomain{A,B} <: IntegrationDomain
  cells::Vector{Int32}
  cell_irows::A
  rows::B
end

function IntegrationDomain(
  trian::Triangulation,
  test::FESpace,
  rows::Vector{<:Number}
  )

  cell_row_ids = get_cell_dof_ids(test,trian)
  cells = get_rows_to_cells(cell_row_ids,rows)
  irows = get_cells_to_idofs(cell_row_ids,cells,rows)
  VectorDomain(cells,irows,rows)
end

get_integration_cells(i::VectorDomain) = i.cells
get_cell_irows(i::VectorDomain) = i.cell_irows

function move_integration_domain(
  i::VectorDomain,
  test::FESpace,
  ttrian::Triangulation,
  )

  rows = i.rows
  IntegrationDomain(ttrian,test,rows)
end

"""
    struct MatrixDomain{A,B,C,D} <: IntegrationDomain
      cells::Vector{Int32}
      cell_irows::A
      cell_icols::B
      rows::C
      cols::D
    end

Integration domain for a projection operator in a steady problem
"""
struct MatrixDomain{A,B,C,D} <: IntegrationDomain
  cells::Vector{Int32}
  cell_irows::A
  cell_icols::B
  rows::C
  cols::D
end

function IntegrationDomain(
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::Vector{<:Number},
  cols::Vector{<:Number}
  )

  cell_row_ids = get_cell_dof_ids(test,trian)
  cell_col_ids = get_cell_dof_ids(trial,trian)
  cells = get_rowcols_to_cells(cell_row_ids,cell_col_ids,rows,cols)
  irows = get_cells_to_idofs(cell_row_ids,cells,rows)
  icols = get_cells_to_idofs(cell_col_ids,cells,cols)
  MatrixDomain(cells,irows,icols,rows,cols)
end

get_integration_cells(i::MatrixDomain) = i.cells
get_cell_irows(i::MatrixDomain) = i.cell_irows
get_cell_icols(i::MatrixDomain) = i.cell_icols

function move_integration_domain(
  i::MatrixDomain,
  trial::FESpace,
  test::FESpace,
  ttrian::Triangulation,
  )

  rows = i.rows
  cols = i.cols
  IntegrationDomain(ttrian,trial,test,rows,cols)
end

# utils

function _isrow(cellrows,rows)
  for row in cellrows
    for _row in rows
      if row == _row
        return true
      end
    end
  end
  return false
end

function _isrow(cellrows::VectorBlock,rows)
  for i in eachindex(cellrows)
    if cellrows.touched[i]
      if _isrow(cellrows.array[i],rows)
        return true
      end
    end 
  end
  return false
end

function _isrowcol(cellrows,cellcols,rows,cols)
  for col in cellcols
    for row in cellrows
      for _col in cols
        for _row in rows
          if row == _row && col == _col
            return true
          end
        end
      end
    end
  end
  return false
end

function _isrowcol(cellrows::VectorBlock,cellcols::VectorBlock,rows,cols)
  for j in eachindex(cellcols)
    if cellcols.touched[j]
      for i in eachindex(cellrows)
        if cellrows.touched[i]
          if _isrowcol(cellrows.array[i],cellcols.array[j],rows,cols)
            return true
          end
        end 
      end
    end 
  end
  return false
end

function _cache(ids::AbstractArray)
  _getids(a) = a 
  _getids(a::OIdsToIds) = a.indices
  CachedArray(similar(_getids(ids)))
end

function _cache(ids::ArrayBlock)
  ai = _cache(testitem(ids))
  array = Array{typeof(ai),ndims(ids)}(undef,size(ids))
  for i in eachindex(ids.array)
    if ids.touched[i]
      array[i] = _cache(ids.array[i])
    end
  end
  ArrayBlock(array,ids.touched)
end

function _evaluate!(a,cellrows,rows)
  fill!(a,zero(eltype(a)))
  for (irow,row) in enumerate(rows)
    for (icellrow,cellrow) in enumerate(cellrows)
      if row == cellrow
        a[icellrow] = irow
      end
    end
  end
  a 
end

function _evaluate!(a,cellrows::OIdsToIds,rows)
  fill!(a,zero(eltype(a)))
  for (irow,row) in enumerate(rows)
    for (_icellrow,cellrow) in enumerate(cellrows)
      if row == cellrow
        icellrow = cellrows.terms[_icellrow] 
        a[icellrow] = irow
      end
    end
  end
  a 
end

function _evaluate!(a::VectorBlock,cellrows::VectorBlock,rows)
  @check a.touched == cellrows.touched 
  for i in eachindex(a)
    if a.touched[i]
      _evaluate!(a.array[i],cellrows.array[i],rows)
    end
  end
  a
end

function _evaluate!(a,cellrows,cellcols,rows,cols)
  fill!(a,zero(eltype(a)))
  ncellrows = length(cellrows)
  for (irowcol,rowcol) in enumerate(zip(rows,cols))
    row,col = rowcol
    for (icellrow,cellrow) in enumerate(cellrows)
      for (icellcol,cellcol) in enumerate(cellcols)
        if row == cellrow && col == cellcol
          icellrowcol = icellrow + (icellcol-1)*ncellrows
          a[icellrowcol] = irowcol
        end
      end
    end
  end
  a 
end

function _evaluate!(a,cellrows::OIdsToIds,cellcols::OIdsToIds,rows,cols)
  fill!(a,zero(eltype(a)))
  ncellrows = length(cellrows)
  for (irowcol,rowcol) in enumerate(zip(rows,cols))
    row,col = rowcol
    for (_icellrow,cellrow) in enumerate(cellrows)
      for (_icellcol,cellcol) in enumerate(cellcols)
        if row == cellrow && col == cellcol
          icellrow = cellrows.terms[_icellrow] 
          icellcol = cellcols.terms[_icellcol]
          icellrowcol = icellrow + (icellcol-1)*ncellrows
          a[icellrowcol] = irowcol
        end
      end
    end
  end
  a 
end

function _evaluate!(a::VectorBlock,cellrows::VectorBlock,cellcols::VectorBlock,rows,cols)
  @check a.touched == cellrows.touched == cellcols.touched 
  for i in eachindex(a)
    if a.touched[i]
      _evaluate!(a.array[i],cellrows.array[i],cellcols.array[i],rows,cols)
    end
  end
  a
end