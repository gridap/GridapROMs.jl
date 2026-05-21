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

get_idof_correction(a) = (idof,celldofs) -> idof
get_idof_correction(a::OTable) = (idof,celldofs) -> celldofs.terms[idof]
get_idof_correction(a::LazyArray{<:Fill{<:Reindex}}) = get_idof_correction(a.maps[1].values)
get_idof_correction(a::AppendedArray) = get_idof_correction(a.a)

function get_cells_to_irows(
  cell_row_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  rows::AbstractVector
  )

  correct_irow = get_idof_correction(cell_row_ids)
  cache = array_cache(cell_row_ids)

  ncells = length(cells)
  ptrs = Vector{Int32}(undef,ncells+1)
  @inbounds for (icell,cell) in enumerate(cells)
    cellrows = getindex!(cache,cell_row_ids,cell)
    ptrs[icell+1] = length(cellrows)
  end
  length_to_ptrs!(ptrs)

  data = fill(zero(Int32),ptrs[end]-1)
  for (icell,cell) in enumerate(cells)
    cellrows = getindex!(cache,cell_row_ids,cell)
    for (irow,row) in enumerate(rows)
      for (_icellrow,cellrow) in enumerate(cellrows)
        if row == cellrow
          icellrow = correct_irow(_icellrow,cellrows)
          data[ptrs[icell]-1+icellrow] = irow
        end
      end
    end
  end

  Table(data,ptrs)
end

function get_cells_to_irowcols(
  cell_row_ids::AbstractArray{<:AbstractArray},
  cell_col_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  rows::AbstractVector,
  cols::AbstractVector
  )

  correct_irow = get_idof_correction(cell_row_ids)
  correct_icol = get_idof_correction(cell_col_ids)
  rowcache = array_cache(cell_row_ids)
  colcache = array_cache(cell_col_ids)

  ncells = length(cells)
  ptrs = Vector{Int32}(undef,ncells+1)
  @inbounds for (icell,cell) in enumerate(cells)
    cellrows = getindex!(rowcache,cell_row_ids,cell)
    cellcols = getindex!(colcache,cell_col_ids,cell)
    ptrs[icell+1] = length(cellrows)*length(cellcols)
  end
  length_to_ptrs!(ptrs)

  data = fill(zero(Int32),ptrs[end]-1)
  for (icell,cell) in enumerate(cells)
    cellrows = getindex!(rowcache,cell_row_ids,cell)
    cellcols = getindex!(colcache,cell_col_ids,cell)
    ncellrows = length(cellrows)
    for (irowcol,rowcol) in enumerate(zip(rows,cols))
      row,col = rowcol
      for (_icellrow,cellrow) in enumerate(cellrows)
        for (_icellcol,cellcol) in enumerate(cellcols)
          if row == cellrow && col == cellcol
            icellrow = correct_irow(_icellrow,cellrows)
            icellcol = correct_icol(_icellcol,cellcols)
            icellrowcol = icellrow + (icellcol-1)*ncellrows
            data[ptrs[icell]-1+icellrowcol] = irowcol
          end
        end
      end
    end
  end

  Table(data,ptrs)
end

struct FetchBlockMap{A} <: Map
  values::A
  blockid::Int
end

function Arrays.return_cache(k::FetchBlockMap,i...)
  array_cache(k.values)
end

function Arrays.evaluate!(cache,k::FetchBlockMap,i...)
  a = getindex!(cache,k.values,i...)
  a.array[k.blockid]
end

function get_cells_to_irows(
  cell_row_ids::AbstractArray{<:VectorBlock},
  cells::AbstractVector,
  rows::AbstractVector
  )

  item_rows = testitem(cell_row_ids)
  T = typeof(empty_table(Int,Int32,0))
  array = Vector{T}(undef,length(item_rows))
  touched = fill(false,size(array))
  for i in eachindex(item_rows)
    if item_rows.touched[i]
      array[i] = get_cells_to_irows(
        lazy_map(FetchBlockMap(cell_row_ids,i),eachindex(cell_row_ids)),
        cells,rows
      )
      touched[i] = true
    end 
  end
  return ArrayBlock(array,touched)
end

function get_cells_to_irowcols(
  cell_row_ids::AbstractArray{<:VectorBlock},
  cell_col_ids::AbstractArray{<:VectorBlock},
  cells::AbstractVector,
  rows::AbstractVector,
  cols::AbstractVector
  )

  item_rows = testitem(cell_row_ids)
  item_cols = testitem(cell_col_ids)
  T = typeof(empty_table(Int,Int32,0))
  array = Matrix{T}(undef,(length(item_rows),length(item_cols)))
  touched = fill(false,size(array))
  for j in eachindex(item_cols)
    if item_cols.touched[j]
      for i in eachindex(item_rows)
        if item_rows.touched[i]
          array[i,j] = get_cells_to_irowcols(
            lazy_map(FetchBlockMap(cell_row_ids,i),eachindex(cell_row_ids)),
            lazy_map(FetchBlockMap(cell_col_ids,j),eachindex(cell_col_ids)),
            cells,rows,cols
          )
          touched[i,j] = true
        end 
      end
    end 
  end
  return ArrayBlock(array,touched)
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

get_integration_cells(i::IntegrationDomain,args...) = @abstractmethod
get_cell_idofs(i::IntegrationDomain) = @abstractmethod
get_interpolation_dofs(i::IntegrationDomain) = @abstractmethod

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
  rows::Vector{<:Number}
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
  rows::Vector{<:Number},
  cols::Vector{<:Number}
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

get_integration_cells(g::Grid) = @abstractmethod
get_integration_cells(g::Triangulation) = get_integration_cells(get_grid(g))
get_integration_cells(g::Geometry.GridView) = g.cell_to_parent_cell

function get_integration_cells(g::Geometry.AppendedTriangulation)
  pa = get_parent(g.a)
  npa = num_cells(pa)
  ca = get_integration_cells(g.a)
  cb = get_integration_cells(g.b) 
  return lazy_append(ca,cb.+npa)
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