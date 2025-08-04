struct PosZeroNegReindex{A,B} <: Map
  values_pos::A
  values_neg::B
end

function Arrays.testargs(k::PosZeroNegReindex,i::Integer)
  @check length(k.values_pos) !=0 || length(k.values_neg) != 0 "This map has empty domain"
  @check eltype(k.values_pos) == eltype(k.values_neg) "This map is type-unstable"
  length(k.values_pos) !=0 ? (one(i),) : (-one(i))
end

function Arrays.return_value(k::PosZeroNegReindex,i::Integer)
  if length(k.values_pos)==0 && length(k.values_neg)==0
    @check eltype(k.values_pos) == eltype(k.values_neg) "This map is type-unstable"
    testitem(k.values_pos)
  else
    evaluate(k,testargs(k,i)...)
  end
end

function Arrays.return_cache(k::PosZeroNegReindex,i::Integer)
  c_p = array_cache(k.values_pos)
  c_n = array_cache(k.values_neg)
  z = zero(eltype(k.values_pos))
  c_p,c_n,z
end

function Arrays.evaluate!(cache,k::PosZeroNegReindex,i::Integer)
  c_p,c_n,z = cache
  i>0 ? getindex!(c_p,k.values_pos,i) : (i<0 ? getindex!(c_n,k.values_neg,-i) : z)
end

function Arrays.evaluate(k::PosZeroNegReindex,i::Integer)
  i>0 ? k.values_pos[i] : (i<0 ? k.values_neg[-i] : zero(eltype(k.values_pos)))
end
