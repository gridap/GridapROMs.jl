function DEIM(basis::AbstractMatrix{T}) where T
  m,n = size(basis)
  res = zeros(T,m)
  I = zeros(Int,n)
  basisI = zeros(T,n,n)
  @inbounds @views begin
    @. res = basis[:,1]
    I[1] = argmax(abs,res)
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
      I[l] = argmax(abs,res)
      basisI[l,:] = basis[I[l],:]
    end
  end
  return I,basisI
end

function SOPT(basis::AbstractMatrix{T}) where T
  m,n = size(basis)
  I = zeros(Int,n)
  basisI = zeros(T,n,n)
  @inbounds @views begin
    I[1] = argmax(abs,basis[:,1])
    basisI[1,:] = basis[I[1],:]
    for l in 2:n
      U = basis[:,1:l]
      P = I[1:l-1]
      PᵀU = U[P,:]
      G = PᵀU'*PᵀU
      colnorms2 = vec(sum(abs2,PᵀU;dims=1))
      Il = best_s_opt_index(U,P,G,colnorms2)
      @check Il > 0
      I[l] = Il
      basisI[l,:] = basis[Il,:]
    end
  end
  return I,basisI
end

function best_s_opt_index(U,P,G,colnorms2)
  m,n = size(U)
  best_i = 0
  best_logS = -Inf
  @inbounds @views for l in 1:m
    l ∈ P && continue
    q = U[l,:]
    logdet_plus = robust_logdet(G)
    colnorms2_plus = colnorms2 .+ abs2.(G + q*q')
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
    H = symcholesky(A)
    2.0*sum(log,diag(H.L))
  catch
    _,Σ,_ = svd(A)
    tol = maximum(Σ)*eps(T)+eps(T)
    sum(log,clamp.(Σ,tol,Inf))
  end
end

for f in (:DEIM,:SOPT)
  @eval begin
    function $f(A::ParamSparseMatrix)
      I,AI = $f(get_all_data(A))
      R′,C′ = recast_split_indices(I,testitem(A))
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

struct CellsToIrowsMap{A,B,C} <: Map
  cell_row_ids::A
  cells::B
  rows::C
end

function Arrays.return_cache(k::CellsToIrowsMap,icell::Int)
  row_cache = array_cache(k.cell_row_ids)
  rows = getindex!(row_cache,k.cell_row_ids,icell)
  irow_cache = _cache(rows)
  unwrap_cache = return_cache(unwrap_and_setsize!,irow_cache,rows)
  return row_cache,irow_cache,unwrap_cache
end

function Arrays.evaluate!(cache,k::CellsToIrowsMap,icell::Int)
  row_cache,irow_cache,unwrap_cache = cache
  cell = k.cells[icell]
  cellrows = getindex!(row_cache,k.cell_row_ids,cell)
  a = evaluate!(unwrap_cache,unwrap_and_setsize!,irow_cache,cellrows)
  _evaluate!(a,cellrows,k.rows)
end

function get_cells_to_irows(cell_row_ids,cells,rows)
  isempty(cells) && return Fill(Int32[],0)
  k = CellsToIrowsMap(cell_row_ids,cells,rows)
  lazy_map(k,1:length(cells))
end

struct CellsToIrowcolsMap{A,B,C,D,E} <: Map
  cell_row_ids::A
  cell_col_ids::B
  cells::C
  rows::D
  cols::E
end

function Arrays.return_cache(k::CellsToIrowcolsMap,icell::Int)
  row_cache = array_cache(k.cell_row_ids)
  col_cache = array_cache(k.cell_col_ids)
  cellrows = getindex!(row_cache,k.cell_row_ids,icell)
  cellcols = getindex!(col_cache,k.cell_col_ids,icell)
  irowcol_cache = _cache(cellrows,cellcols)
  unwrap_cache = return_cache(unwrap_and_setsize!,irowcol_cache,cellrows,cellcols)
  return row_cache,col_cache,irowcol_cache,unwrap_cache
end

function Arrays.evaluate!(cache,k::CellsToIrowcolsMap,icell::Int)
  row_cache,col_cache,irowcol_cache,unwrap_cache = cache
  cell = k.cells[icell]
  cellrows = getindex!(row_cache,k.cell_row_ids,cell)
  cellcols = getindex!(col_cache,k.cell_col_ids,cell)
  a = evaluate!(unwrap_cache,unwrap_and_setsize!,irowcol_cache,cellrows,cellcols)
  _evaluate!(a,cellrows,cellcols,k.rows,k.cols)
end

function get_cells_to_irowcols(cell_row_ids,cell_col_ids,cells,rows,cols)
  isempty(cells) && return Fill(Int32[],0)
  k = CellsToIrowcolsMap(cell_row_ids,cell_col_ids,cells,rows,cols)
  lazy_map(k,1:length(cells))
end

"""
    abstract type IntegrationDomain end

Type representing the set of interpolation DOFs of a `Projection` subjected
to a EIM approximation.
Subtypes:
- [`GenericDomain`](@ref)
- [`TransientIntegrationDomain`](@ref)
"""
abstract type IntegrationDomain end

get_integration_cells(i::IntegrationDomain) = @abstractmethod
get_cell_idofs(i::IntegrationDomain) = @abstractmethod
get_interpolation_dofs(i::IntegrationDomain) = @abstractmethod

function IntegrationDomain(args...)
  @abstractmethod
end

"""
    struct GenericDomain{A,B} <: IntegrationDomain
      cells::Vector{Int32}
      cell_idofs::A
      dofs::B
    end

Integration domain for the hyper-reduction of a residual (vector-valued form) or
a Jacobian (matrix-valued form).

- `cells`: sorted list of cell indices (local to the triangulation) that contain
  at least one interpolation DOF.
- `cell_idofs`: lazy map over `cells`; for the vector case, `cell_idofs[ic][j]` is
  the 1-based index into `dofs` of the `j`-th DOF of cell `cells[ic]`, or `0` if
  that DOF is not an interpolation DOF. For the matrix case, `cell_idofs[ic]` is a
  flat (column-major) array of length `ncellrows * ncellcols`; position
  `r + (c-1)*ncellrows` stores the EIM pair index, or `0` if that (row,col) entry
  is not an interpolation pair.
- `dofs`: for the vector case, a vector of global DOF indices selected as
  interpolation DOFs by EIM/DEIM. For the matrix case, a tuple `(rows, cols)` of
  such vectors.
"""
struct GenericDomain{A,B} <: IntegrationDomain
  cells::Vector{Int32}
  cell_idofs::A
  dofs::B
end

function IntegrationDomain(
  trian::Triangulation,
  test::FESpace,
  rows::AbstractVector
  )

  cell_dof_ids = get_cell_dof_ids(test,trian)
  cells = get_rows_to_cells(cell_dof_ids,rows)
  idofs = get_cells_to_irows(cell_dof_ids,cells,rows)
  GenericDomain(cells,idofs,rows)
end

function IntegrationDomain(
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::AbstractVector,
  cols::AbstractVector
  )

  cell_row_ids = get_cell_dof_ids(test,trian)
  cell_col_ids = get_cell_dof_ids(trial,trian)
  cells = get_rowcols_to_cells(cell_row_ids,cell_col_ids,rows,cols)
  irowcols = get_cells_to_irowcols(cell_row_ids,cell_col_ids,cells,rows,cols)
  GenericDomain(cells,irowcols,(rows,cols))
end

get_integration_cells(i::GenericDomain) = i.cells
get_cell_idofs(i::GenericDomain) = i.cell_idofs
get_interpolation_dofs(i::GenericDomain) = i.dofs

function move_integration_domain(
  i::GenericDomain,
  test::FESpace,
  ttrian::Triangulation,
  )

  rows = i.dofs
  IntegrationDomain(ttrian,test,rows)
end

function move_integration_domain(
  i::GenericDomain,
  trial::FESpace,
  test::FESpace,
  ttrian::Triangulation,
  )

  rows,cols = i.dofs
  IntegrationDomain(ttrian,trial,test,rows,cols)
end

# triangulation utils 

get_integration_cells(t::Triangulation) = @abstractmethod
get_integration_cells(t::ChildTriangulation) = t.cell_to_parent_cell

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

function _cache(rows::AbstractVector{<:Integer})
  CachedArray(similar(rows,Int32))
end

function _cache(rows::VectorBlock)
  ai = _cache(testitem(rows))
  array = Vector{typeof(ai)}(undef,length(rows))
  for i in eachindex(rows.array)
    if rows.touched[i]
      array[i] = _cache(rows.array[i])
    end
  end
  ArrayBlock(array,rows.touched)
end

function _cache(rows::AbstractVector{<:Integer},cols::AbstractVector{<:Integer})
  CachedArray(similar(rows,Int32,length(rows)*length(cols)))
end

function _cache(rows::VectorBlock,cols::VectorBlock)
  ai = _cache(testitem(rows),testitem(cols))
  array = Matrix{typeof(ai)}(undef,length(rows),length(cols))
  touched = fill(false,size(array))
  for j in eachindex(cols.array)
    if cols.touched[j]
      for i in eachindex(rows.array)
        if rows.touched[i]
          array[i,j] = _cache(rows.array[i],cols.array[j])
          touched[i,j] = true
        end
      end 
    end
  end
  ArrayBlock(array,touched)
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

function _evaluate!(a::VectorBlock,cellrows::VectorBlock,rows)
  for i in eachindex(a)
    if a.touched[i]
      @check cellrows.touched[i]
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

function _evaluate!(a::MatrixBlock,cellrows::VectorBlock,cellcols::VectorBlock,rows,cols)
  for j in axes(a,2), i in axes(a,1)
    if a.touched[i,j]
      @check cellrows.touched[i] && cellcols.touched[j]
      _evaluate!(a.array[i,j],cellrows.array[i],cellcols.array[j],rows,cols)
    end
  end
  a
end