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

function RBTransient.first_unfold(A::GenericPArray{T,3}) where T
  values = map(local_values(A)) do A
    RBTransient.first_unfold(A)
  end
  GenericPArray(values,row_partition(A))
end

function _method_of_snapshots_row(red_style::ReductionStyle,A,AA)
  _,Sr,Vr = RBSteady.truncated_svd(red_style,AA;issquare=true)
  Ur = _weighted_mul(A,Vr,Sr)
  return Ur,Sr,Vr
end

function _weighted_mul(A,V,S)
  Ta = eltype(A)
  Tv = eltype(V)
  T = typeof(zero(Ta)*zero(Tv)+zero(Ta)*zero(Tv))
  U = GenericPArray{Matrix{T}}(undef,row_partition(A),axes(V,2))
  D = Diagonal(S.+eps())
  map(own_values(U),own_values(A)) do Uo,Ao
    mul!(Uo,Ao,V)
    rdiv!(Uo,D)
  end
  consistent!(U) |> fetch
  U
end

# integration domains

function RBSteady.DEIM(basis::GenericPMatrix{T}) where T
  n = size(basis,2)
  I = zeros(Int,n)
  I_local = map(_ -> Int[],partition(axes(basis,1)))
  basisI = zeros(T,n,n)
  res = GenericPArray{Vector{T}}(undef,partition(axes(basis,1)))
  map(own_values(res),own_values(basis)) do ro,bo
    @. ro = bo[:,1]
  end
  I[1] = argmax(abs,res)
  _push_to_local!(I_local,I[1],partition(axes(basis,1)))
  _from_submatrix!(basisI,basis,I,1)
  for l = 2:n
    PᵀU = view(basisI,1:l-1,1:l-1)
    Pᵀuₗ = view(basisI,1:l-1,l)
    c = vec(PᵀU \ Pᵀuₗ)
    map(own_values(res),own_values(basis)) do ro,bo
      @. ro = bo[:,l]
      mul!(ro,view(bo,:,1:l-1),c,-1.0,1.0)
    end
    I[l] = argmax(abs,res)
    _push_to_local!(I_local,I[l],partition(axes(basis,1)))
    _from_submatrix!(basisI,basis,I,l)
  end
  return I_local,basisI
end

function RBSteady.SOPT(basis::GenericPMatrix{T}) where T
  n = size(basis,2)
  I = zeros(Int,n)
  I_local = map(_ -> Int[],partition(axes(basis,1)))
  basisI = zeros(T,n,n)
  res = GenericPArray{Vector{T}}(undef,partition(axes(basis,1)))
  map(own_values(res),own_values(basis)) do ro,bo
    @. ro = bo[:,1]
  end
  I[1] = argmax(abs,res)
  _push_to_local!(I_local,I[1],partition(axes(basis,1)))
  _from_submatrix!(basisI,basis,I,1)
  for l in 2:n
    P = I[1:l-1]
    PᵀU = view(basisI,1:l-1,1:l)
    G = PᵀU'*PᵀU
    colnorms2 = vec(sum(abs2,PᵀU;dims=1))
    Il = _best_s_opt_index(basis,P,G,colnorms2,l)
    @check Il > 0
    I[l] = Il
    _push_to_local!(I_local,I[l],partition(axes(basis,1)))
    _from_submatrix!(basisI,basis,I,l)
  end
  return I_local,basisI
end

# projections

PartitionedArrays.partition(a::Projection) = partition(get_basis(a))
PartitionedArrays.local_values(a::Projection) = local_values(get_basis(a))
PartitionedArrays.own_values(a::Projection) = own_values(get_basis(a))
PartitionedArrays.ghost_values(a::Projection) = ghost_values(get_basis(a))
PartitionedArrays.consistent!(a::Projection) = consistent!(get_basis(a))

struct DistributedPODProjection <: Projection
  basis::AbstractMatrix
end

function RBSteady.PODProjection(basis::GenericPMatrix)
  DistributedPODProjection(basis)
end

function RBSteady.Projection(basis::GenericPMatrix,s::DistributedSparseSnapshots)
  basis′ = recast(basis,s)
  DistributedPODProjection(basis′)
end

function RBTransient.kron_projection(red::KroneckerReduction,s::DistributedSparseSnapshots,args...)
  basis_space,basis_time = tucker(red.reductions,s,args...)
  basis_space′ = recast(basis_space,s)
  projection_space = PODProjection(basis_space′)
  projection_time = PODProjection(basis_time)
  return projection_space,projection_time
end

RBSteady.get_basis(a::DistributedPODProjection) = a.basis
row_partition(a::DistributedPODProjection) = row_partition(a.basis)
col_partition(a::DistributedPODProjection) = col_partition(a.basis)

function RBSteady.union_bases(a::DistributedPODProjection,b::DistributedPODProjection,args...) 
  union_bases(a,get_basis(b),args...)
end

function RBSteady.union_bases(a::DistributedPODProjection,basis_b::AbstractMatrix,args...)
  basis_a = get_basis(a)
  basis_ab = gram_schmidt(basis_b,basis_a,args...)
  DistributedPODProjection(basis_ab)
end

function RBSteady.galerkin_projection(a::DistributedPODProjection,b::DistributedPODProjection)
  lb̂ = map(own_values(a),own_values(b)) do ao,bo
    galerkin_projection(ao,bo)
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

function RBSteady.galerkin_projection(a::DistributedPODProjection,b::DistributedPODProjection,c::DistributedPODProjection,args...)
  lb̂ = map(own_values(a),own_values(b),own_values(c)) do ao,bo,co
    galerkin_projection(ao,bo,co,args...)
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

function RBSteady._allocate_projection(red::Reduction,s::DistributedBlockSnapshots{N}) where N
  T = DistributedPODProjection
  block_basis = Array{T,N}(undef,size(s))
  BlockProjection(block_basis,s.touched)
end

function GridapDistributed.local_views(a::DistributedPODProjection)
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

# for (T,f) in zip((:DEIMHyperReduction,:SOPTHyperReduction),(:DEIM,:SOPT))
#   @eval begin
#     function RBSteady.Interpolation(
#       red::$T,
#       basis::DistributedPODProjection,
#       trian::DistributedTriangulation,
#       test::DistributedRBSpace
#       )

#       interps = map(local_views(basis),local_views(trian),local_views(test)) do bi,ti,testi
#         rows,interp = $f(bi)
#         factor = lu(interp)
#         domain = IntegrationDomain(ti,testi,rows)
#         GreedyInterpolation(factor,domain)
#       end
#       DistributedInterpolation(interps)
#     end

#     function RBSteady.Interpolation(
#       red::$T,
#       basis::DistributedPODProjection,
#       trian::DistributedTriangulation,
#       trial::DistributedRBSpace,
#       test::DistributedRBSpace
#       )

#       interps = map(local_views(basis),local_views(trian),local_views(trial),local_views(test)) do bi,ti,triali,testi
#         (rows,cols),interp = $f(bi)
#         println("  DEIM rows=$rows cols=$cols basis_size=$(size(get_basis(bi)))")
#         cell_row_ids = get_cell_dof_ids(get_fe_space(testi),ti)
#         cell_col_ids = get_cell_dof_ids(get_fe_space(triali),ti)
#         println("  n_cells=$(length(cell_row_ids)) row_ids_sample=$(cell_row_ids[1])")
#         factor = lu(interp)
#         domain = IntegrationDomain(ti,triali,testi,rows,cols)
#         GreedyInterpolation(factor,domain)
#       end
#       DistributedInterpolation(interps)
#     end
#   end
# end

# function RBSteady.Interpolation(
#   red::RBFHyperReduction,
#   basis::DistributedPODProjection,
#   s::DistributedSnapshots
#   )

#   interps = map(local_views(basis),local_views(s)) do bi,si
#     Interpolation(red,bi,si)
#   end
#   DistributedInterpolation(interps)
# end

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

# utils 

function submatrix(a::GenericPMatrix,global_rows,global_cols)
  map(own_values(a),partition(axes(a,1))) do values,ra
    g2o = global_to_own(ra)
    own_rows = filter(>(0),g2o[global_rows])
    view(values,own_rows,global_cols)
  end
end

function submatrix(a::GenericPMatrix,::Colon,global_cols)
  new_parts = map(partition(a)) do values
    view(values,:,global_cols)
  end
  GenericPArray(new_parts,partition(axes(a,1)))
end

function _subfill!(a::AbstractVector,b::AbstractVector,ia,ib)
  a[ia] = b[ib]
end

function _subfill!(a::AbstractMatrix,b::AbstractMatrix,ia,ib)
  @check size(a,2) == size(b,2)
  @inbounds for k in axes(a,2)
    a[ia,k] = b[ib,k]
  end
end

function _from_submatrix!(aI,a,I,l)
  map(own_values(a),partition(axes(a,1))) do oa,ra
    g2o = global_to_own(ra)
    for k in l 
      or = g2o[I[k]]
      or > 0 && _subfill!(aI,oa,k,or)
    end
  end
end

function _push_to_local!(I_local,gi,idx_partition)
  map(I_local,idx_partition) do li,ra
    oi = global_to_own(ra)[gi]
    oi > 0 && push!(li,oi)
  end
end

function _best_s_opt_index(basis::GenericPMatrix,P,G,colnorms2,l)
  best_pairs = map(own_values(basis),partition(axes(basis,1))) do bo,ra
    best_logS = -Inf
    best_gi = 0
    for oi in axes(bo,1)
      gi = own_to_global(ra)[oi]
      gi ∈ P && continue
      q = view(bo,oi,1:l)
      logdet_plus = RBSteady.robust_logdet(G + q*q')
      colnorms2_plus = colnorms2 .+ abs2.(q)
      logS = (0.5/l)*(logdet_plus - sum(log,colnorms2_plus))
      if logS > best_logS
        best_logS = logS
        best_gi = gi
      end
    end
    best_logS => best_gi
  end
  return second(reduce(max,best_pairs,init=(-Inf=>0)))
end