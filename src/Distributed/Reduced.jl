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
  U
end

# projections

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
  # consistent!(a) |> fetch; a should already be consistent
  consistent!(b) |> fetch 
  lb̂ = map(own_values(a),own_values(b)) do ao,bo
    galerkin_projection(ao,bo)
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

function RBSteady.galerkin_projection(a::DistributedPODProjection,b::DistributedPODProjection,c::DistributedPODProjection,args...)
  # consistent!(a) |> fetch; a should already be consistent
  # consistent!(c) |> fetch; c should already be consistent
  consistent!(b) |> fetch
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

PartitionedArrays.partition(a::Projection) = partition(get_basis(a))
PartitionedArrays.local_values(a::Projection) = local_values(get_basis(a))
PartitionedArrays.own_values(a::Projection) = own_values(get_basis(a))
PartitionedArrays.ghost_values(a::Projection) = ghost_values(get_basis(a))

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

PartitionedArrays.consistent!(a::DistributedPODProjection) = consistent!(a.basis)
PartitionedArrays.consistent!(a::NormedProjection) = consistent!(a.projection)

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

for (T,f) in zip((:DEIMHyperReduction,:SOPTHyperReduction),(:DEIM,:SOPT))
  @eval begin
    function RBSteady.Interpolation(
      red::$T,
      basis::DistributedPODProjection,
      trian::DistributedTriangulation,
      test::DistributedRBSpace
      )

      interps = map(local_views(basis),local_views(trian),local_views(test)) do bi,ti,testi
        rows,interp = $f(bi)
        factor = lu(interp)
        domain = IntegrationDomain(ti,testi,rows)
        GreedyInterpolation(factor,domain)
      end
      DistributedInterpolation(interps)
    end

    function RBSteady.Interpolation(
      red::$T,
      basis::DistributedPODProjection,
      trian::DistributedTriangulation,
      trial::DistributedRBSpace,
      test::DistributedRBSpace
      )

      interps = map(local_views(basis),local_views(trian),local_views(trial),local_views(test)) do bi,ti,triali,testi
        (rows,cols),interp = $f(bi)
        println("  DEIM rows=$rows cols=$cols basis_size=$(size(get_basis(bi)))")
        cell_row_ids = get_cell_dof_ids(get_fe_space(testi),ti)
        cell_col_ids = get_cell_dof_ids(get_fe_space(triali),ti)
        println("  n_cells=$(length(cell_row_ids)) row_ids_sample=$(cell_row_ids[1])")
        factor = lu(interp)
        domain = IntegrationDomain(ti,triali,testi,rows,cols)
        GreedyInterpolation(factor,domain)
      end
      DistributedInterpolation(interps)
    end
  end
end

function RBSteady.Interpolation(
  red::RBFHyperReduction,
  basis::DistributedPODProjection,
  s::DistributedSnapshots
  )

  interps = map(local_views(basis),local_views(s)) do bi,si
    Interpolation(red,bi,si)
  end
  DistributedInterpolation(interps)
end