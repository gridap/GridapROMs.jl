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

struct DistributedProjection{P,A,B} <: Projection
  projections::A
  index_partition::B
  function DistributedProjection(projections::A,index_partition::B) where {P<:Projection,A<:AbstractArray{<:P},B}
    new{P,A,B}(projections,index_partition)
  end
end

function RBSteady.Projection(basis::GenericPArray,s::DistributedSnapshots)
  index_partition = row_partition(s)
  projections = map(local_views(basis),local_views(s)) do lb,ls
    Projection(lb,ls)
  end
  DistributedProjection(projections,index_partition)
end

# function RBSteady.Projection(basis::AbstractMatrix,s::DistributedSparseSnapshots)
#   basis′ = recast(basis,s)
#   PODProjection(basis′)
# end

RBSteady.get_basis(a::DistributedProjection) = map(get_basis,a.projections)
RBSteady.num_fe_dofs(a::DistributedProjection) = num_fe_dofs(getany(a.projections))
RBSteady.num_reduced_dofs(a::DistributedProjection) = num_reduced_dofs(getany(a.projections))
RBSteady.projection_eltype(a::DistributedProjection) = projection_eltype(getany(a.projections))

function RBSteady.galerkin_projection(a::DistributedProjection,b::DistributedProjection)
  lb̂ = map(local_views(a),local_views(b)) do ai,bi
    galerkin_projection(get_basis(ai),get_basis(bi))
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

function RBSteady.galerkin_projection(a::DistributedProjection,b::DistributedProjection,c::DistributedProjection,args...)
  lb̂ = map(local_views(a),local_views(b),local_views(c)) do ai,bi,ci
    galerkin_projection(get_basis(ai),get_basis(bi),get_basis(ci),args...)
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

function RBSteady._allocate_projection(red::Reduction,s::DistributedBlockSnapshots{N}) where N
  T = DistributedProjection
  block_basis = Array{T,N}(undef,size(s))
  BlockProjection(block_basis,s.touched)
end

GridapDistributed.local_views(a::DistributedProjection) = a.projections

function GridapDistributed.local_views(a::NormedProjection)
  map(local_views(a.projection),local_views(a.norm_matrix)) do proj,norm
    NormedProjection(proj,norm)
  end
end

# reduced basis spaces

const DistributedRBSpace{S<:DistributedFESpace} = RBSpace{S}

function GridapDistributed.local_views(r::DistributedRBSpace)
  map(local_views(r.space),local_views(r.subspace)) do lspace,lsubspace
    RBSpace(lspace,lsubspace)
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
      basis::DistributedProjection,
      trian::DistributedTriangulation,
      test::DistributedRBSpace
      )

      interps = map(basis.projections,basis.index_partition,local_views(trian),local_views(test)) do bi,rp,ti,testi
        local_rows,interp = $f(bi)
        dof_rows = convert(Vector{Int},own_to_local(rp)[local_rows])
        factor = lu(interp)
        domain = IntegrationDomain(ti,testi,dof_rows)
        GreedyInterpolation(factor,domain)
      end
      DistributedInterpolation(interps)
    end

    function RBSteady.Interpolation(
      red::$T,
      basis::DistributedProjection,
      trian::DistributedTriangulation,
      trial::DistributedRBSpace,
      test::DistributedRBSpace
      )

      interps = map(basis.projections,basis.index_partition,local_views(trian),local_views(trial),local_views(test)) do bi,rp,ti,triali,testi
        (local_rows,local_cols),interp = $f(bi)
        dof_rows = convert(Vector{Int},own_to_local(rp)[local_rows])
        dof_cols = convert(Vector{Int},own_to_local(rp)[local_cols])
        factor = lu(interp)
        domain = IntegrationDomain(ti,triali,testi,dof_rows,dof_cols)
        GreedyInterpolation(factor,domain)
      end
      DistributedInterpolation(interps)
    end
  end
end

function RBSteady.Interpolation(
  red::RBFHyperReduction,
  basis::DistributedProjection,
  s::DistributedSnapshots
  )

  interps = map(basis.projections,local_views(s)) do bi,si
    Interpolation(red,bi,si)
  end
  DistributedInterpolation(interps)
end