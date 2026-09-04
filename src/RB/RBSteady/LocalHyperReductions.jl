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
  local_vals(a)[labk,labl]
end

function get_cell_idofs(a::LocalInterpolation) 
  data = map(get_cell_idofs,local_vals(a))
  return Table(data)
end

function get_integration_cells(a::LocalInterpolation)
  _union(args...) = @notimplemented
  _union(a::T,b::T) where T<:AbstractVector = union(a,b)
  _union(a::T,b::T) where T<:AppendedArray = lazy_append(union(a.a,b.a),union(a.b,b.b))

  vals = local_vals(a)
  isempty(vals) && return Int32[]
  cells = get_integration_cells(vals[1])
  for i in 2:length(vals)
    cells = _union(cells,get_integration_cells(vals[i]))
  end
  return cells
end

function get_owned_icells(a::LocalInterpolation,cells::AbstractVector) 
  data = map(i -> get_owned_icells(i,cells),local_vals(a))
  return Table(data)
end

struct LocalHRProjection <: HRProjection{Projection,HyperReduction}
  reductions::AbstractMatrix
  k::NTuple{2,KmeansResult}
end

LocalHRProjection(reductions::AbstractVector,k::KmeansResult) = LocalHRProjection(reductions,(k,))

get_basis(a::LocalHRProjection) = LocalProjection(map(get_basis,local_vals(a)),get_clusters(a))
get_interpolation(a::LocalHRProjection) = LocalInterpolation(map(get_interpolation,local_vals(a)),get_clusters(a))
projection_eltype(a::LocalHRProjection) = promote_type(map(projection_eltype,local_vals(a)))

local_vals(a::LocalHRProjection) = a.reductions

function local_vals(a::BlockHRProjection)
  litems = map(local_vals,a.array)
  nlitems = length(first(litems))
  map(1:nlitems) do i
    BlockHRProjection(getindex.(litems,i),a.touched)
  end
end

get_clusters(a::LocalHRProjection) = a.k

get_local(a::HRProjection,μ::AbstractVector) = a

function get_local(a::LocalHRProjection,μ::AbstractVector)
  k,l = get_clusters(a)
  labk = get_label(k,μ)
  labl = get_label(l,μ)
  local_vals(a)[labk,labl]
end

function get_local(a::BlockHRProjection,μ::AbstractVector)
  BlockHRProjection(map(p -> get_local(p,μ),a.array),a.touched)
end

function get_local(a::AffineContribution,μ::AbstractVector)
  contribution(get_domains(a)) do trian
    get_local(a[trian],μ)
  end
end

const LocalHRContribution = AffineContribution{<:LocalHRProjection}

function reduced_form(lred::LocalReduction,s,trian,test)
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

function reduced_form(lred::LocalReduction,s,trian,trial,test)
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

function reduced_form(lred::LocalReduction,s::AbstractBlockSnapshots,trian,test)
  @check length(s) == length(test)

  hyper_reds = map(eachindex(s)) do i
    hyper_red, = reduced_form(lred,s[i],trian,test[i])
    hyper_red
  end

  hyper_red = BlockHRProjection(hyper_reds,s.touched)
  red_trian = reduced_triangulation(trian,hyper_red)

  return hyper_red,red_trian
end

function reduced_form(lred::LocalReduction,s::AbstractBlockSnapshots,trian,trial,test)
  @check size(s,1) == length(test)
  @check size(s,2) == length(trial)

  hyper_reds = map(Iterators.product(axes(s)...)) do (i,j)
    hyper_red, = reduced_form(lred,s[i,j],trian,trial[j],test[i])
    hyper_red
  end

  hyper_red = BlockHRProjection(hyper_reds,s.touched)
  red_trian = reduced_triangulation(trian,hyper_red)

  return hyper_red,red_trian
end