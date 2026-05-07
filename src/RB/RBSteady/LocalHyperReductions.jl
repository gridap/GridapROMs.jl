struct LocalInterpolation <: Interpolation
  interp::AbstractMatrix{<:Interpolation}
  k::NTuple{2,KmeansResult}
end

local_vals(a::LocalInterpolation) = a.interp
get_clusters(a::LocalInterpolation) = a.k

function get_local(a::LocalInterpolation,μ::AbstractVector)
  k,l = get_clusters(a)
  labk = get_label(k,μ)
  labl = get_label(l,μ)
  a.interp[labk,labl]
end

Base.ndims(a::LocalInterpolation) = ndims(a.interp)
Base.size(a::LocalInterpolation,args...) = size(a.interp,args...)
Base.axes(a::LocalInterpolation,args...) = axes(a.interp,args...)
Base.length(a::LocalInterpolation) = length(a.interp)
Base.eachindex(a::LocalInterpolation) = eachindex(a.interp)
Base.getindex(a::LocalInterpolation,i...) = a.interp[i...] 
Base.setindex!(a::LocalInterpolation,v,i...) = (a.interp[i...] = v)
Arrays.testitem(a::LocalInterpolation) = first(a.interp)

for f in (:get_cell_irows,:get_cell_icols)
  @eval begin
    function $f(a::LocalInterpolation) 
      data = map($f,a.interp)
      return Table(data)
    end
  end
end

function get_integration_cells(a::LocalInterpolation,args...)
  _union(args...) = @notimplemented
  _union(a::T,b::T) where T<:AbstractVector = union(a,b)
  _union(a::T,b::T) where T<:AppendedArray = lazy_append(union(a.a,b.a),union(a.b,b.b))

  isempty(a.interp) && return Int32[]
  cells = get_integration_cells(a.interp[1],args...)
  for i in 2:length(a)
    cells = _union(cells,get_integration_cells(a.interp[i],args...))
  end
  return cells
end

function get_owned_icells(a::LocalInterpolation,args...)
  cells = get_integration_cells(a,args...)
  get_owned_icells(a,cells)
end

function get_owned_icells(a::LocalInterpolation,cells::AbstractVector) 
  data = map(i -> get_owned_icells(i,cells),a.interp)
  return Table(data)
end

function move_interpolation(a::LocalInterpolation,args...)
  interp = map(i -> move_interpolation(i,args...),a.interp)
  return LocalInterpolation(interp,a.touched)
end

struct LocalHRProjection <: HRProjection{Projection,HyperReduction}
  reductions::AbstractMatrix
  k::NTuple{2,KmeansResult}
end

LocalHRProjection(reductions::AbstractVector,k::KmeansResult) = LocalHRProjection(reductions,(k,))

get_basis(a::LocalHRProjection) = LocalProjection(map(get_basis,a.reductions),a.k)
get_interpolation(a::LocalHRProjection) = LocalInterpolation(map(get_interpolation,a.reductions),a.k)
projection_eltype(a::LocalHRProjection) = promote_type(map(projection_eltype,a.reductions))

local_vals(a::LocalHRProjection) = a.reductions
get_clusters(a::LocalHRProjection) = a.k

function get_local(a::LocalHRProjection,μ::AbstractVector)
  k,l = get_clusters(a)
  labk = get_label(k,μ)
  labl = get_label(l,μ)
  a.reductions[labk,labl]
end

function reduced_form(
  lred::LocalReduction,
  s::Snapshots,
  trian::Triangulation,
  test::SingleFieldRBSpace
  )
  
  red = get_reduction(lred)
  ks = compute_clusters(lred,s)
  kr, = get_clusters(test)
  cs = cluster(s,ks)

  hr = Matrix{HRProjection}(undef,length(ks.counts),length(kr.counts))

  for i in eachindex(ks.counts)
    si = cs[i]
    for (j,centerj) in enumerate(eachcol(kr.centers))
      testj = get_local(test,centerj)
      hyper_redij, = reduced_form(red,si,trian,testj)
      hr[i,j] = hyper_redij
    end
  end

  hyper_red = LocalHRProjection(hr,(ks,kr))
  red_trian = reduced_triangulation(trian,hyper_red)

  return hyper_red,red_trian
end

function reduced_form(
  lred::LocalReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::SingleFieldRBSpace,
  test::SingleFieldRBSpace
  )
  
  red = get_reduction(lred)
  ks = compute_clusters(lred,s)
  kr, = get_clusters(test)
  cs = cluster(s,ks)

  hr = Matrix{HRProjection}(undef,length(ks.counts),length(kr.counts))

  for i in eachindex(ks.counts)
    si = cs[i]
    for (j,centerj) in enumerate(eachcol(kr.centers))
      trialj = get_local(trial,centerj)
      testj = get_local(test,centerj)
      hyper_redij, = reduced_form(red,si,trian,trialj,testj)
      hr[i,j] = hyper_redij
    end
  end

  hyper_red = LocalHRProjection(hr,(ks,kr))
  red_trian = reduced_triangulation(trian,hyper_red)

  return hyper_red,red_trian
end