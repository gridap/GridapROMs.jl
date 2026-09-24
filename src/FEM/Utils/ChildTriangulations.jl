struct ChildTriangulation{Dc,Dp,A,B} <: Triangulation{Dc,Dp}
  parent::A
  cell_to_parent_cell::B
  function ChildTriangulation(parent::Triangulation,cell_to_parent_cell::AbstractArray)
    Dc = num_cell_dims(parent)
    Dp = num_point_dims(parent)
    A = typeof(parent)
    B = typeof(cell_to_parent_cell)
    new{Dc,Dp,A,B}(parent,cell_to_parent_cell)
  end
end

function ChildTriangulation(parent::Triangulation,parent_cell_to_mask::AbstractArray{Bool})
  cell_to_parent_cell = findall(collect1d(parent_cell_to_mask))
  ChildTriangulation(parent,cell_to_parent_cell)
end

function ChildTriangulation(parent::Triangulation,parent_cell_to_mask::AbstractVector{Bool})
  cell_to_parent_cell = findall(parent_cell_to_mask)
  ChildTriangulation(parent,cell_to_parent_cell)
end

function Geometry.get_background_model(trian::ChildTriangulation)
  get_background_model(trian.parent)
end

function Geometry.get_grid(trian::ChildTriangulation)
  get_grid(trian.parent)
end

function Geometry.get_glue(t::ChildTriangulation,::Val{d}) where d
  get_glue(t.parent,Val(d))
end

function Geometry.get_facet_normal(trian::ChildTriangulation)
  get_facet_normal(trian.parent)
end

function Geometry.get_cell_map(trian::ChildTriangulation)
  get_cell_map(trian.parent)
end

# relationships

function is_parent(parent::Triangulation,child::Triangulation)
  false 
end

function is_parent(parent::Triangulation,child::ChildTriangulation)
  parent === child.parent
end

function find_trian_permutation(a,b,cmp::Function)
  map(a -> findfirst(b -> cmp(a,b),b),a)
end

function find_trian_permutation(a,b)
  cmp(a,b) = a === b || is_parent(a,b)
  find_trian_permutation(a,b,cmp)
end

"""
    order_domains(
      parents::Tuple{Vararg{Triangulation}},
      children::Tuple{Vararg{Triangulation}}
      ) -> Tuple{Vararg{Triangulation}}

Orders the triangulation children in the same way as the triangulation parents
"""
function order_domains(parents,children)
  @check length(parents) == length(children)
  perm = find_trian_permutation(parents,children)
  map(p->children[p],perm)
end

function change_triangulation(old::Tuple,new::Tuple)
  perm = find_trian_permutation(old,new)
  new′ = ()
  for p in perm
    new′ = (new′...,new[p])
  end
  return new′
end

function change_triangulation(old::Tuple,new::AbstractArray{<:Tuple})
  map(n -> change_triangulation(old,n),new)
end

function change_triangulation(old::AbstractArray{<:Tuple},new::Tuple)
  map(o -> change_triangulation(o,new),old)
end

function change_triangulation(old::AbstractArray{<:Tuple},new::AbstractArray{<:Tuple})
  map(change_triangulation,old,new)
end