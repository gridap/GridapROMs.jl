struct TProductCellPoint{DS<:DomainStyle,A<:CellPoint{DS},B<:AbstractVector{<:CellPoint{DS}}} <: CellDatum
  point::A
  single_points::B
end

get_tp_data(f::TProductCellPoint) = f.single_points

function CellData.get_triangulation(f::TProductCellPoint)
  s1 = first(f.single_points)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.single_points))
  trian
end

CellData.DomainStyle(::Type{<:TProductCellPoint{DS}}) where DS = DS()

# default behavior

function Arrays.evaluate!(cache,f::CellField,x::TProductCellPoint)
  evaluate!(cache,f,x.point)
end

abstract type TProductCellField <: CellField end

get_tp_data(f::AbstractVector{<:AbstractArray}) = f
get_tp_data(f::AbstractVector{<:CellField}) = f
get_tp_data(f::TProductCellField) = @abstractmethod

get_diff_data(f::TProductCellField) = @abstractmethod

get_tp_diff_data(f::AbstractVector{<:AbstractArray}) = f
get_tp_diff_data(f::AbstractVector{<:CellField}) = f
get_tp_diff_data(f::TProductCellField) = @abstractmethod

struct GenericTProductCellField{DS<:DomainStyle,A,B} <: TProductCellField
  single_fields::A
  trian::B
  domain_style::DS

  function GenericTProductCellField(single_fields::A,trian::B) where {A<:Vector{<:CellField},B<:TProductTriangulation}
    @assert length(single_fields) > 0
    domain_style = DomainStyle(first(single_fields))
    @check all(DomainStyle(sf)==domain_style for sf in single_fields)
    DS = typeof(domain_style)
    new{DS,A,B}(single_fields,trian,domain_style)
  end
end

get_tp_data(f::GenericTProductCellField) = f.single_fields

CellData.get_triangulation(f::GenericTProductCellField) = f.trian

CellData.DomainStyle(::Type{<:GenericTProductCellField{DS}}) where DS = DS()

# differentiation

abstract type TProductDiffCellField <: TProductCellField end

struct GenericTProductDiffCellField{O,A,B,C} <: TProductDiffCellField
  op::O
  cell_data::A
  diff_cell_data::B
  summation::C
end

function GenericTProductDiffCellField(op,cell_data,diff_cell_data)
  GenericTProductDiffCellField(op,cell_data,diff_cell_data,nothing)
end

const GradientTProductCellField{A,B,C} = GenericTProductDiffCellField{typeof(gradient),A,B,C}

CellData.get_data(f::GenericTProductDiffCellField) = f.cell_data
get_diff_data(f::GenericTProductDiffCellField) = f.diff_cell_data

get_tp_data(f::GenericTProductDiffCellField) = get_tp_data(f.cell_data)
get_tp_diff_data(f::GenericTProductDiffCellField) = get_tp_data(f.diff_cell_data)

CellData.get_triangulation(f::GenericTProductDiffCellField) = get_triangulation(f.cell_data)

CellData.DomainStyle(f::GenericTProductDiffCellField) = DomainStyle(f.cell_data)

function Fields.gradient(f::TProductCellField)
  g = GenericTProductCellField(gradient.(f.single_fields),f.trian)
  return GenericTProductDiffCellField(gradient,f,g)
end

# stores the evaluation of a GenericTProductDiffCellField on a quadrature
struct GenericTProductDiffEval{O,A,B,C}
  op::O
  f::A
  g::B
  summation::C
end

function GenericTProductDiffEval(op,f,g)
  GenericTProductDiffEval(op,f,g,nothing)
end

function GenericTProductDiffEval(op,f::Vector{DomainContribution},g::Vector{DomainContribution})
  s = _block_operation(nothing,testitem(first(f)),testitem(first(g)))
  op′ = _block_operation(op,testitem(first(f)),testitem(first(g)))
  GenericTProductDiffEval(op′,f,g,s)
end

CellData.get_data(f::GenericTProductDiffEval) = f.f
get_diff_data(f::GenericTProductDiffEval) = f.g

get_tp_data(f::GenericTProductDiffEval) = get_tp_data(f.f)
get_tp_diff_data(f::GenericTProductDiffEval) = get_tp_data(f.g)

# fe basis

struct TProductFEBasis{DS,BS,A,B} <: FEBasis
  basis::A
  trian::B
  domain_style::DS
  basis_style::BS
end

function TProductFEBasis(basis::Vector{<:FESpaces.FEBasis},trian::TProductTriangulation)
  b1 = testitem(basis)
  DS = DomainStyle(b1)
  BS = BasisStyle(b1)
  @check all(map(i -> DS===DomainStyle(i) && BS===BasisStyle(i),basis))
  TProductFEBasis(basis,trian,DS,BS)
end

function TProductFEBasis(basis::Vector{<:MultiField.MultiFieldCellField},trian::TProductTriangulation)
  b1 = testitem(basis)
  DS = DomainStyle(b1)
  BS = BasisStyle(first(b1))
  @check all(map(i -> DS===DomainStyle(i) && BS===BasisStyle(first(i)),basis))
  TProductFEBasis(basis,trian,DS,BS)
end

get_tp_data(f::TProductFEBasis) = f.basis
CellData.get_triangulation(f::TProductFEBasis) = f.trian
FESpaces.BasisStyle(::Type{<:TProductFEBasis{DS,BS}}) where {DS,BS} = BS()
CellData.DomainStyle(::Type{<:TProductFEBasis{DS,BS}}) where {DS,BS} = DS()

function Fields.gradient(f::TProductFEBasis)
  dbasis = map(gradient,f.basis)
  trian = get_triangulation(f)
  g = TProductFEBasis(dbasis,trian)
  return GenericTProductDiffCellField(gradient,f,g)
end

const MultiFieldTProductFEBasis{DS,BS,B} = TProductFEBasis{DS,BS,Vector{MultiFieldCellField{DS}},B}

MultiField.num_fields(f::MultiFieldTProductFEBasis) = length(f.basis[1])

Base.length(f::MultiFieldTProductFEBasis) = num_fields(f)

function Base.getindex(f::MultiFieldTProductFEBasis,i::Integer)
  TProductFEBasis(getindex.(f.basis,i),f.trian,f.domain_style,f.basis_style)
end

function Base.iterate(f::MultiFieldTProductFEBasis,state=1)
  if state > num_fields(f)
    return nothing
  end
  astate = TProductFEBasis(getindex.(f.basis,state),f.trian,f.domain_style,f.basis_style)
  return astate,state+1
end

# evaluations

const TProductCellDatum = Union{TProductFEBasis,TProductCellField}

function Arrays.return_cache(f::TProductCellDatum,x::TProductCellPoint)
  @assert length(get_tp_data(f)) == length(get_tp_data(x))
  fitem = testitem(get_tp_data(f))
  xitem = testitem(get_tp_data(x))
  c1 = return_cache(fitem,xitem)
  fx1 = evaluate(fitem,xitem)
  cache = Vector{typeof(c1)}(undef,length(get_tp_data(f)))
  array = Vector{typeof(fx1)}(undef,length(get_tp_data(f)))
  return cache,array
end

function Arrays.evaluate!(_cache,f::TProductCellDatum,x::TProductCellPoint)
  cache,b = _cache
  @inbounds for i = eachindex(get_tp_data(f))
    b[i] = evaluate!(cache[i],get_tp_data(f)[i],get_tp_data(x)[i])
  end
  return b
end

function Arrays.return_cache(k::Operation,f::TProductCellDatum...)
  D = length(get_tp_data(first(f)))
  @assert all(map(fi -> length(get_tp_data(fi)) == D,f))
  fitem = map(testitem,get_tp_data.(f))
  c1 = return_cache(k,fitem...)
  Fill(c1,D)
end

function Arrays.evaluate!(cache,k::Operation,α::TProductCellDatum,β::TProductCellDatum)
  αβ = map(evaluate!,cache,Fill(k,length(cache)),get_tp_data(α),get_tp_data(β))
  GenericTProductCellField(αβ,get_triangulation(α))
end

function Arrays.return_cache(f::GenericTProductDiffCellField,x::TProductCellPoint)
  cache = return_cache(get_data(f),x)
  diff_cache = return_cache(get_diff_data(f),x)
  return cache,diff_cache
end

function Arrays.evaluate!(_cache,f::GenericTProductDiffCellField,x::TProductCellPoint)
  cache,diff_cache = _cache
  fx = evaluate!(cache,get_data(f),x)
  dfx = evaluate!(diff_cache,get_diff_data(f),x)
  return GenericTProductDiffEval(f.op,fx,dfx)
end

function Arrays.return_cache(k::Operation,α::GradientTProductCellField,β::GradientTProductCellField)
  cache = return_cache(k,get_data(α),get_data(β))
  diff_cache = return_cache(k,get_diff_data(α),get_diff_data(β))
  return cache,diff_cache
end

function Arrays.evaluate!(_cache,k::Operation,α::GradientTProductCellField,β::GradientTProductCellField)
  cache,diff_cache = _cache
  αβ = evaluate!(cache,k,get_data(α),get_data(β))
  dαβ = evaluate!(diff_cache,k,get_diff_data(α),get_diff_data(β))
  return GenericTProductDiffCellField(gradient,αβ,dαβ)
end

# partial derivatives

function Utils.PartialDerivative{N}(f::TProductCellField) where N
  g = GenericTProductCellField(PartialDerivative{1}.(f.single_fields),f.trian)
  return GenericTProductDiffCellField(PartialDerivative{N}(),f,g)
end

function Utils.PartialDerivative{N}(f::TProductFEBasis) where N
  dbasis = PartialDerivative{1}.(f.basis)
  g = GenericTProductCellField(dbasis,f.trian)
  return GenericTProductDiffCellField(PartialDerivative{N}(),f,g)
end

const PartialDerivativeTProductCellField{N,A,B,C} = GenericTProductDiffCellField{PartialDerivative{N},A,B,C}

function Arrays.return_cache(k::Operation{typeof(*)},α::TProductCellDatum,β::PartialDerivativeTProductCellField)
  cache = return_cache(k,α,get_data(β))
  diff_cache = return_cache(k,α,get_diff_data(β))
  return cache,diff_cache
end

function Arrays.return_cache(k::Operation{typeof(*)},α::PartialDerivativeTProductCellField,β::TProductCellDatum)
  cache = return_cache(k,get_data(α),β)
  diff_cache = return_cache(k,get_diff_data(α),β)
  return cache,diff_cache
end

function Arrays.evaluate!(_cache,k::Operation{typeof(*)},α::TProductCellDatum,β::PartialDerivativeTProductCellField{N}) where N
  cache,diff_cache = _cache
  αβ = evaluate!(cache,k,α,get_data(β))
  dαβ = evaluate!(cache,k,α,get_diff_data(β))
  return GenericTProductDiffCellField(β.op,αβ,dαβ)
end

function Arrays.evaluate!(_cache,k::Operation{typeof(*)},α::PartialDerivativeTProductCellField{N},β::TProductCellDatum) where N
  cache,diff_cache = _cache
  αβ = evaluate!(cache,k,get_data(α),β)
  dαβ = evaluate!(cache,k,get_diff_data(α),β)
  return GenericTProductDiffCellField(α.op,αβ,dαβ)
end

# integration

function CellData.integrate(f::TProductCellDatum,a::TProductMeasure)
  map(integrate,get_tp_data(f),a.measures_1d)
end

function CellData.integrate(f::GenericTProductDiffCellField,a::TProductMeasure)
  fi = map(integrate,get_tp_data(f),a.measures_1d)
  dfi = map(integrate,get_tp_diff_data(f),a.measures_1d)
  GenericTProductDiffEval(f.op,fi,dfi)
end

for f in (:+,:-)
  @eval ($f)(a::AbstractArray,b::GenericTProductDiffEval) = _add_tp_cell_data($f,a,b)
  @eval ($f)(a::GenericTProductDiffEval,b::AbstractArray) = _add_tp_cell_data($f,a,b)
  @eval ($f)(a::GenericTProductDiffEval,b::GenericTProductDiffEval) = _add_tp_cell_data($f,a,b)
end

Arrays.testitem(a::DomainContribution) = testitem(a[first([get_domains(a)...])])

function _add_tp_cell_data(f,a::AbstractVector{<:DomainContribution},b::GenericTProductDiffEval)
  a1 = testitem(a[1])
  b1 = testitem(b.f[1])
  summation = _block_operation(f,a1,b1,b.summation)
  if _is_different_block(a1,b1)
    GenericTProductDiffEval(b.op,f(a,b.f),b.g,summation)
  else
    GenericTProductDiffEval(b.op,a,b.g,summation)
  end
end

function _add_tp_cell_data(f,a::GenericTProductDiffEval,b::AbstractVector{<:DomainContribution})
  a1 = testitem(a.f[1])
  b1 = testitem(b[1])
  summation = _block_operation(f,a1,b1,a.summation)
  if _is_different_block(a1,b1)
    GenericTProductDiffEval(a.op,f(a.f,b),a.g,summation)
  else
    GenericTProductDiffEval(a.op,a.f,a.g,summation)
  end
end

function _add_tp_cell_data(f,a::GenericTProductDiffEval{Oa},b::GenericTProductDiffEval{Ob}) where {Oa,Ob}
  af1,ag1 = testitem(a.f[1]),testitem(a.g[1])
  bf1,bg1 = testitem(b.f[1]),testitem(b.g[1])
  bf = _is_different_block(af1,bf1)
  bg = _is_different_block(ag1,bg1)
  if !(bf || bg)
    return _add_tp_div_cell_data(ag1,bg1,f,a,b)
  end
  op = _disjoint_block_operation(a.op,ag1,bg1,b.op)
  summation = _block_operation(ag1,bg1,f,af1,bf1,a.summation)
  if bf && bg
    GenericTProductDiffEval(op,f(a.f,b.f),f(a.g,b.g),summation)
  elseif bf
    GenericTProductDiffEval(op,f(a.f,b.f),a.g,summation)
  elseif bg
    GenericTProductDiffEval(op,a.f,f(a.g,b.g),summation)
  else
    @notimplemented
  end
end

#TODO probably this function is wrong in general
function _add_tp_div_cell_data(
  ag1,bg1,f,
  a::GenericTProductDiffEval{PartialDerivative{A}},
  b::GenericTProductDiffEval{PartialDerivative{B}}
  ) where {A,B}

  op = PartialDerivative{(A...,B...)}()
  s = nothing
  GenericTProductDiffEval(op,a.f,a.g,s)
end

function _add_tp_div_cell_data(
  ag1,bg1,f,
  a::GenericTProductDiffEval{ArrayBlock{PartialDerivative{A},N}},
  b::GenericTProductDiffEval{ArrayBlock{PartialDerivative{B},N}}
  ) where {A,B,N}

  f′ = PartialDerivative{(A...,B...)}()
  op = _block_operation(f′,ag1,bg1)
  s = _block_operation(nothing,ag1,bg1)
  GenericTProductDiffEval(op,a.f,a.g,s)
end

function _is_different_block(a1,b1)
  false
end

function _is_different_block(a1::ArrayBlock,b1::ArrayBlock)
  all(findall(a1.touched) .!= findall(b1.touched))
end

function _block_operation(f,a1,b1,args...)
  f
end

function _disjoint_block_operation(f,a1,b1,args...)
  f
end

function _block_operation(f,a1::ArrayBlock{A,N},b1::ArrayBlock{A,N}) where {A,N}
  @check size(a1) == size(b1)
  overlap = CartesianIndex{N}[]
  for ib in findall(b1.touched)
    if ib ∈ findall(a1.touched)
      push!(overlap,ib)
    end
  end
  block_map = BlockMap(size(a1.touched),overlap)
  return_cache(block_map,fill(f,length(overlap))...)
end

function _block_operation(f,a1::ArrayBlock{A,N},b1::ArrayBlock{A,N},fprev::ArrayBlock{B,N}) where {A,B,N}
  @check size(a1) == size(b1)
  overlap = CartesianIndex{N}[]
  for ib in findall(b1.touched)
    if ib ∈ findall(a1.touched)
      push!(overlap,ib)
    end
  end
  sum_overlap = vcat(findall(fprev.touched)...,overlap...)
  block_map = BlockMap(size(a1.touched),sum_overlap)
  return_cache(block_map,fill(f,length(sum_overlap))...)
end

function _disjoint_block_operation(f,a1::ArrayBlock{A,N},b1::ArrayBlock{A,N},fprev::ArrayBlock{B,N}) where {A,B,N}
  @check size(a1) == size(b1)
  touched_a = findall(a1.touched)
  touched_b = findall(b1.touched)
  @check isempty(intersect(touched_a,touched_b))
  array_a = f[touched_a...]
  array_b = fprev[touched_b...]
  union_touched = vcat(touched_a,touched_b)
  union_array = vcat(array_a,array_b)
  block_map = BlockMap(size(a1.touched),union_touched)
  return_cache(block_map,union_array...)
end
