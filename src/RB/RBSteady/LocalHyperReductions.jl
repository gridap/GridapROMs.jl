struct LocalHRProjection{A<:Projection,B<:HyperReduction} <: HRProjection{A,B}
  projections::AbstractMatrix{<:HRProjection{A,B}}
  k::NTuple{2,KmeansResult}
end

LocalHRProjection(projections::AbstractVector,k::KmeansResult) = LocalHRProjection(projections,(k,))

local_vals(a::LocalHRProjection) = a.projections

function local_vals(a::BlockHRProjection)
  litems = map(local_vals,a.array)
  nlitems = length(first(litems))
  map(1:nlitems) do i
    BlockHRProjection(getindex.(litems,i),a.touched)
  end
end

get_clusters(a::LocalHRProjection) = a.k
get_clusters(a::BlockHRProjection) = get_clusters(testitem(a))

function get_local(a::LocalHRProjection,μ::AbstractVector)
  k,l = get_clusters(a)
  labk = get_label(k,μ)
  labl = get_label(l,μ)
  a.projections[labk,labl]
end

function get_local(a::BlockHRProjection,μ::AbstractVector)
  BlockHRProjection(map(p -> get_local(p,μ),a.array),a.touched)
end

for (S,T) in zip((:Snapshots,:BlockSnapshots),(:SingleFieldRBSpace,:MultiFieldRBSpace))
  @eval begin
    function reduced_form(
      lred::LocalReduction,
      s::$S,
      trian::Triangulation,
      test::$T
      )
      
      red = get_reduction(lred)
      ks = compute_clusters(lred,s)
      kr, = get_clusters(test)
      cs = cluster(s,ks)

      H = _hr_type(red)
      hr = Matrix{H}(undef,length(ks.counts),length(kr.counts))

      for i in eachindex(ks.counts)
        si = cs[i]
        for (j,centerj) in enumerate(eachcol(kr.centers))
          testj = get_local(test,centerj)
          hr[i,j] = reduced_form(red,si,trian,testj)
        end
      end

      LocalHRProjection(hr,(ks,kr))
    end

    function reduced_form(
      lred::LocalReduction,
      s,
      trian::Triangulation,
      trial::$T,
      test::$T
      )
      
      red = get_reduction(lred)
      ks = compute_clusters(lred,s)
      kr, = get_clusters(test)
      cs = cluster(s,ks)

      H = _hr_type(red)
      hr = Matrix{H}(undef,length(ks.counts),length(kr.counts))

      for i in eachindex(ks.counts)
        si = cs[i]
        for (j,centerj) in enumerate(eachcol(kr.centers))
          trialj = get_local(trial,centerj)
          testj = get_local(test,centerj)
          hr[i,j] = reduced_form(red,si,trian,trialj,testj)
        end
      end

      LocalHRProjection(hr,(ks,kr))
    end
  end
end