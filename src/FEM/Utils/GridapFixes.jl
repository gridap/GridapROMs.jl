function Arrays.Reindex(v::V) where {T<:Number,V<:AppendedArray{T}}
  !isconcretetype(T) && return Reindex(_to_concrete_eltype(v))
  return Reindex{V}(v)  
end

function _to_concrete_eltype(v::AppendedArray{T,A,B}) where {T,A,B} 
  V = promote_type(A,B)
  AppendedArray(convert(V,v.a),convert(V,v.b))
end