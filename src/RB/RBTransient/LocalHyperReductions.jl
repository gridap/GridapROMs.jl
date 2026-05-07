RBSteady.get_local(a::TupOfAffineContribution,μ::AbstractVector) = a

const TupOfLocalHRContribution = Tuple{Vararg{LocalHRContribution}}

function RBSteady.get_local(a::TupOfLocalHRContribution,μ::AbstractVector)
  map(a -> get_local(a,μ),a)
end