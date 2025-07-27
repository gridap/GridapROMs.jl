module ParamGeometry

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields
using Gridap.Geometry
using Gridap.Helpers
using Gridap.ReferenceFEs

using GridapEmbedded
using GridapEmbedded.Interfaces
using GridapEmbedded.LevelSetCutters

using GridapROMs.Utils
using GridapROMs.ParamDataStructures

import FillArrays: Fill
import Gridap.CellData: get_data
import Gridap.Fields: AffineMap,ConstantMap

export ParamMappedGrid
export ParamMappedDiscreteModel
export ParamUnstructuredGrid
export mapped_grid
include("ParamGrids.jl")

export ParamSubCellData
include("SubCellTriangulations.jl")

export ParamSubFacetData
include("SubFacetTriangulations.jl")

# utils

function Fields.AffineField(gradients::ParamBlock,origins::ParamBlock)
  data = map(AffineField,get_param_data(gradients),get_param_data(origins))
  GenericParamBlock(data)
end

function Arrays.return_value(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x)

  @check param_length(gradients) == param_length(origins)
  gi = testitem(gradients)
  oi = testitem(origins)
  vi = return_value(k,gi,oi,x)
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    g[i] = return_value(k,param_getindex(gradients,i),param_getindex(origins,i),x)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x)

  @check param_length(gradients) == param_length(origins)
  gi = testitem(gradients)
  oi = testitem(origins)
  li = return_cache(k,gi,oi,x)
  vi = evaluate!(li,k,gi,oi,x)
  l = Vector{typeof(li)}(undef,param_length(gradients))
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    l[i] = return_cache(k,param_getindex(gradients,i),param_getindex(origins,i),x)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(
  cache,k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x)

  @check param_length(gradients) == param_length(origins)
  g,l = cache
  for i in param_eachindex(gradients)
    g.data[i] = evaluate!(l[i],k,param_getindex(gradients,i),param_getindex(origins,i),x)
  end
  g
end

function Arrays.return_value(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x::ParamBlock)

  @check param_length(gradients) == param_length(origins) == param_length(x)
  gi = testitem(gradients)
  oi = testitem(origins)
  xi = testitem(x)
  vi = return_value(k,gi,oi,xi)
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    g[i] = return_value(k,param_getindex(gradients,i),param_getindex(origins,i),param_getindex(x,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x::ParamBlock)

  @check param_length(gradients) == param_length(origins) == param_length(x)
  gi = testitem(gradients)
  oi = testitem(origins)
  xi = testitem(x)
  li = return_cache(k,gi,oi,xi)
  vi = evaluate!(li,k,gi,oi,xi)
  l = Vector{typeof(li)}(undef,param_length(gradients))
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    l[i] = return_cache(k,param_getindex(gradients,i),param_getindex(origins,i),param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(
  cache,k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x::ParamBlock)

  @check param_length(gradients) == param_length(origins) == param_length(x)
  g,l = cache
  for i in param_eachindex(gradients)
    g.data[i] = evaluate!(l[i],k,param_getindex(gradients,i),param_getindex(origins,i),param_getindex(x,i))
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ConstantMap},a::ParamBlock,x)
  ai = testitem(a)
  vi = return_value(k,ai,x)
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    g[i] = return_value(k,param_getindex(a,i),x)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ConstantMap},a::ParamBlock,x)
  ai = testitem(a)
  li = return_cache(k,ai,x)
  vi = evaluate!(li,k,ai,x)
  l = Vector{typeof(li)}(undef,param_length(a))
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    l[i] = return_cache(k,param_getindex(a,i),x)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ConstantMap},a::ParamBlock,x)
  g,l = cache
  for i in param_eachindex(a)
    g.data[i] = evaluate!(l[i],k,param_getindex(a,i),x)
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ConstantMap},a::ParamBlock,x::ParamBlock)
  @check param_length(a) == param_length(x)
  ai = testitem(a)
  xi = testitem(x)
  vi = return_value(k,ai,xi)
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    g[i] = return_value(k,param_getindex(a,i),param_getindex(x,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ConstantMap},a::ParamBlock,x::ParamBlock)
  @check param_length(a) == param_length(x)
  ai = testitem(a)
  xi = testitem(x)
  li = return_cache(k,ai,xi)
  vi = evaluate!(li,k,ai,xi)
  l = Vector{typeof(li)}(undef,param_length(a))
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    l[i] = return_cache(k,param_getindex(a,i),param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ConstantMap},a::ParamBlock,x::ParamBlock)
  @check param_length(a) == param_length(x)
  g,l = cache
  for i in param_eachindex(a)
    g.data[i] = evaluate!(l[i],k,param_getindex(a,i),param_getindex(x,i))
  end
  g
end

function Arrays.return_value(f::Field,x::ParamBlock)
  xi = testitem(x)
  vi = return_value(f,xi)
  g = Vector{typeof(vi)}(undef,param_length(x))
  for i in param_eachindex(x)
    g[i] = return_value(f,param_getindex(x,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(f::Field,x::ParamBlock)
  xi = testitem(x)
  li = return_cache(f,xi)
  vi = evaluate!(li,f,xi)
  l = Vector{typeof(li)}(undef,param_length(x))
  g = Vector{typeof(vi)}(undef,param_length(x))
  for i in param_eachindex(x)
    l[i] = return_cache(f,param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,f::Field,x::ParamBlock)
  g,l = cache
  for i in param_eachindex(x)
    g.data[i] = evaluate!(l[i],f,param_getindex(x,i))
  end
  g
end

for f in (:(Fields.GenericField),:(Fields.ZeroField),:(Fields.ConstantField),:(Fields.inverse_map))
  @eval begin
    $f(a::ParamBlock) = GenericParamBlock(map($f,get_param_data(a)))
  end
end

end
