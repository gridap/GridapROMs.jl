for I in (:Int32,:Int64)
  @eval begin
    Base.@propagate_inbounds function Base.getindex(v::AppendedArray{T},i::$I) where T
      l = length(v.a)
      result = i > l ? v.b[i-l] : v.a[i]
      convert(T,result)
    end

    Base.@propagate_inbounds function Arrays.getindex!(cache,v::AppendedArray{T},i::$I) where T
      ca,cb = cache
      l = length(v.a)
      result = i > l ? Arrays.getindex!(cb,v.b,i-l) : Arrays.getindex!(ca,v.a,i)
      convert(T,result)
    end
  end
end

Base.@propagate_inbounds function Arrays.evaluate(k::Reindex{<:AppendedArray{T}},i...) where T
  convert(T,k.values[i...])
end

function Arrays.lazy_map(k::Reindex{<:AppendedArray{T}},j_to_i::AbstractArray) where T
  Arrays.lazy_map(k,T,j_to_i)
end