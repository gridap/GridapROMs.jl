#TODO Gridap bug fix
function Arrays.testvalue(::Type{Fields.LinearCombinationField{V,F}}) where {V<:AbstractVector,F}
  fields = testvalue(F)
  values = zeros(eltype(V),length(fields))
  Gridap.Fields.LinearCombinationField(values,fields,1)
end

function Arrays.Reindex(v::V) where {T<:Number,V<:AppendedArray{T}}
  !isconcretetype(T) && return Reindex(_to_concrete_eltype(v))
  return Reindex{V}(v)  
end

function _to_concrete_eltype(v::AppendedArray{T,A,B}) where {T,A,B}
  V = promote_type(A,B)
  AppendedArray(convert(V,v.a),convert(V,v.b))
end