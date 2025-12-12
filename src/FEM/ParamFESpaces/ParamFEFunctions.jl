"""
    abstract type ParamFEFunction <: FEFunction end

Parametric extension of a `FEFunction` in [`Gridap`](@ref).
Subtypes:
- [`SingleFieldParamFEFunction`](@ref)
- [`MultiFieldParamFEFunction`](@ref)
"""
abstract type ParamFEFunction <: FEFunction end

"""
    struct SingleFieldParamFEFunction{T<:CellField} <: ParamFEFunction
      cell_field::T
      cell_dof_values::AbstractArray{<:ParamBlock}
      free_values::AbstractParamVector{<:Number}
      dirichlet_values::AbstractParamVector{<:Number}
      fe_space::SingleFieldFESpace
    end
"""
struct SingleFieldParamFEFunction{T<:CellField} <: ParamFEFunction
  cell_field::T
  cell_dof_values::AbstractArray{<:ParamBlock}
  free_values::AbstractParamVector{<:Number}
  dirichlet_values::AbstractParamVector{<:Number}
  fe_space::SingleFieldFESpace
end

function FESpaces.SingleFieldFEFunction(
  cell_field::T,
  cell_dof_values::AbstractArray{<:ParamBlock},
  free_values::AbstractParamVector{<:Number},
  dirichlet_values::AbstractParamVector{<:Number},
  fs::SingleFieldFESpace) where T

  SingleFieldParamFEFunction(cell_field,cell_dof_values,free_values,dirichlet_values,fs)
end

CellData.get_data(f::SingleFieldParamFEFunction) = get_data(f.cell_field)
FESpaces.get_triangulation(f::SingleFieldParamFEFunction) = get_triangulation(f.cell_field)
CellData.DomainStyle(::Type{SingleFieldParamFEFunction{T}}) where T = DomainStyle(T)
FESpaces.get_free_dof_values(f::SingleFieldParamFEFunction) = f.free_values
FESpaces.get_cell_dof_values(f::SingleFieldParamFEFunction) = f.cell_dof_values
FESpaces.get_fe_space(f::SingleFieldParamFEFunction) = f.fe_space

# audodiff

function FESpaces._change_argument(op,f,trian,uh::SingleFieldParamFEFunction)
  U = get_fe_space(uh)
  function g(cell_u)
    cf = CellField(U,cell_u)
    cell_grad = f(cf)
    CellData.get_contribution(cell_grad,trian)
  end
  g
end

# for visualization/testing purposes

function ParamDataStructures.param_getindex(f::GenericCellField,index::Integer)
  data = get_data(f)
  trian = get_triangulation(f)
  DS = DomainStyle(f)
  di = param_getindex.(data,index)
  GenericCellField(di,trian,DS)
end

function ParamDataStructures.param_length(f::SingleFieldParamFEFunction)
  @assert param_length(f.free_values) == param_length(f.dirichlet_values)
  param_length(f.free_values)
end

function ParamDataStructures.param_getindex(f::SingleFieldParamFEFunction,index::Integer)
  cf = param_getindex(f.cell_field,index)
  fs = param_getindex(f.fe_space,index)
  cv = lazy_map(x->param_getindex(x,index),f.cell_dof_values)
  fv = param_getindex(f.free_values,index)
  dv = param_getindex(f.dirichlet_values,index)
  SingleFieldFEFunction(cf,cv,fv,dv,fs)
end

"""
    struct MultiFieldParamFEFunction{T<:MultiFieldCellField} <: ParamFEFunction
      single_fe_functions::Vector{<:SingleFieldParamFEFunction}
      free_values::AbstractArray
      fe_space::MultiFieldFESpace
      multi_cell_field::T
    end
"""
struct MultiFieldParamFEFunction{T<:MultiFieldCellField} <: ParamFEFunction
  single_fe_functions::Vector{<:SingleFieldParamFEFunction}
  free_values::AbstractArray
  fe_space::MultiFieldFESpace
  multi_cell_field::T

  function MultiFieldParamFEFunction(
    free_values::AbstractVector,
    space::MultiFieldFESpace,
    single_fe_functions::Vector{<:SingleFieldParamFEFunction})

    multi_cell_field = MultiFieldCellField(map(i->i.cell_field,single_fe_functions))
    T = typeof(multi_cell_field)

    new{T}(
      single_fe_functions,
      free_values,
      space,
      multi_cell_field)
  end
end

function MultiField.MultiFieldFEFunction(
  free_values::AbstractVector,
  space::MultiFieldFESpace,
  single_fe_functions::Vector{<:SingleFieldParamFEFunction})

  MultiFieldParamFEFunction(free_values,space,single_fe_functions)
end

CellData.get_data(f::MultiFieldParamFEFunction) = get_data(f.multi_cell_field)
FESpaces.get_triangulation(f::MultiFieldParamFEFunction) = get_triangulation(f.multi_cell_field)
CellData.DomainStyle(::Type{MultiFieldParamFEFunction{T}}) where T = DomainStyle(T)
FESpaces.get_free_dof_values(f::MultiFieldParamFEFunction) = f.free_values
FESpaces.get_fe_space(f::MultiFieldParamFEFunction) = f.fe_space

function FESpaces.get_cell_dof_values(f::MultiFieldParamFEFunction)
  msg = """\n
  This method does not make sense for multi-field
  since each field can be defined on a different triangulation.
  Pass a triangulation in the second argument to get the DOF values
  on top of the corresponding cells.
  """
  trians = map(get_triangulation,f.fe_space.spaces)
  trian = first(trians)
  @check all(map(t->is_change_possible(t,trian),trians)) msg
  get_cell_dof_values(f,trian)
end

function FESpaces.get_cell_dof_values(f::MultiFieldParamFEFunction,trian::Triangulation)
  uhs = f.single_fe_functions
  blockmask = [is_change_possible(get_triangulation(uh),trian) for uh in uhs]
  active_block_ids = findall(blockmask)
  active_block_data = [get_cell_dof_values(uhs[i],trian) for i in active_block_ids]
  nblocks = length(uhs)
  lazy_map(BlockMap(nblocks,active_block_ids),active_block_data...)
end

MultiField.num_fields(m::MultiFieldParamFEFunction) = length(m.single_fe_functions)
Base.length(m::MultiFieldParamFEFunction) = num_fields(m)
Base.iterate(m::MultiFieldParamFEFunction) = iterate(m.single_fe_functions)
Base.iterate(m::MultiFieldParamFEFunction,state) = iterate(m.single_fe_functions,state)
Base.getindex(m::MultiFieldParamFEFunction,field_id::Integer) = m.single_fe_functions[field_id]

# for visualization/testing purposes

function FESpaces.test_fe_function(f::MultiFieldParamFEFunction)
  map(test_fe_function,f.single_fe_functions)
end

ParamDataStructures.param_length(f::MultiFieldParamFEFunction) = param_length(first(f.single_fe_functions))

function ParamDataStructures.param_getindex(f::MultiFieldParamFEFunction,index::Integer)
  style = f.fe_space.multi_field_style
  sff = map(f->param_getindex(f,index),f.single_fe_functions)
  mfs = MultiFieldFESpace(map(f->param_getindex(f,index),f.fe_space);style)
  fv = param_getindex(f.free_values,index)
  MultiFieldFEFunction(fv,mfs,sff)
end
