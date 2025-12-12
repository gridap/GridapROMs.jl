struct BlockMultiFieldExtensionStyle{NB,SB,P} <: MultiFieldStyle end

BlockMultiFieldExtensionStyle() = BlockMultiFieldExtensionStyle{0,0,0}()
function BlockMultiFieldExtensionStyle(NB::Integer,SB::Tuple,P::Tuple)
  @check length(SB) == NB
  @check sum(SB) == length(P)
  return BlockMultiFieldExtensionStyle{NB,SB,P}()
end

function BlockMultiFieldExtensionStyle(NB::Integer,SB::Tuple)
  P = Tuple([1:sum(SB)...])
  return BlockMultiFieldExtensionStyle(NB,SB,P)
end

function BlockMultiFieldExtensionStyle(NB::Integer)
  SB = Tuple(fill(1,NB))
  return BlockMultiFieldExtensionStyle(NB,SB)
end

function BlockMultiFieldExtensionStyle(::BlockMultiFieldExtensionStyle{NB,SB,P},spaces) where {NB,SB,P}
  @check length(spaces) == sum(SB)
  return BlockMultiFieldExtensionStyle(NB,SB,P)
end

function BlockMultiFieldExtensionStyle(::BlockMultiFieldExtensionStyle{0,0,0},spaces)
  NB = length(spaces)
  return BlockMultiFieldExtensionStyle(NB)
end

@inline MultiField.get_block_parameters(::BlockMultiFieldExtensionStyle{NB,SB,P}) where {NB,SB,P} = (NB,SB,P)
function MultiFieldExtensionFESpace(
  spaces::Vector{<:SingleFieldFESpace}; style = BlockMultiFieldExtensionStyle()
)
  @check style == BlockMultiFieldExtensionStyle()
  style = BlockMultiFieldExtensionStyle(style,spaces)
  VT = typeof(mortar(map(zero_free_values,spaces)))
  MultiFieldFESpace(VT,spaces,style)
end

function BlockArrays.blocks(f::MultiFieldFESpace{<:BlockMultiFieldExtensionStyle})
  block_ranges = MultiField.get_block_ranges(get_block_parameters(MultiFieldStyle(f))...)
  block_spaces = map(block_ranges) do range
    isone(length(range)) ? f[range[1]] : MultiFieldFESpace(f.spaces[range])
  end
  return block_spaces
end

function FESpaces.get_free_dof_ids(f::MultiFieldFESpace,::BlockMultiFieldExtensionStyle{NB,SB,P}) where {NB,SB,P}
  block_ranges   = MultiField.get_block_ranges(NB,SB,P)
  block_num_dofs = map(range->sum(map(num_free_dofs,f.spaces[range])),block_ranges)
  return BlockArrays.blockedrange(block_num_dofs)
end

function MultiField._restrict_to_field(
  f,
  ::Union{<:ConsecutiveMultiFieldStyle,<:BlockMultiFieldExtensionStyle},
  free_values,
  field
)
  U = f.spaces
  offsets = MultiField._compute_field_offsets(U)
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  view(free_values,pini:pend)
end

function MultiField._restrict_to_field(
  f,
  mfs::BlockMultiFieldExtensionStyle{NB,SB,P},
  free_values::BlockVector,
  field
) where {NB,SB,P}
  @check blocklength(free_values) == NB
  U = f.spaces

  # Find the block for this field
  block_ranges = MultiField.get_block_ranges(NB,SB,P)
  block_idx    = findfirst(range -> field ∈ range,block_ranges)
  block_free_values = blocks(free_values)[block_idx]

  # Within the block,restrict to field
  offsets = MultiField.compute_field_offsets(f,mfs)
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  return view(block_free_values,pini:pend)
end

function MultiField.compute_field_offsets(f::MultiFieldFESpace,::BlockMultiFieldExtensionStyle{NB,SB,P}) where {NB,SB,P}
  U = f.spaces
  block_ranges  = MultiField.get_block_ranges(NB,SB,P)
  block_offsets = vcat(map(range-> MultiField._compute_field_offsets(U[range]),block_ranges)...)
  offsets = map(p->block_offsets[p],P)
  return offsets
end

function FESpaces.get_cell_dof_ids(f::MultiFieldFESpace,
                                   trian::Triangulation,
                                   ::Union{<:ConsecutiveMultiFieldStyle,<:BlockMultiFieldExtensionStyle})
  offsets = MultiField.compute_field_offsets(f)
  nfields = length(f.spaces)
  blockmask = [ is_change_possible(get_triangulation(Vi),trian) for Vi in f.spaces ]
  active_block_ids = findall(blockmask)
  active_block_data = Any[]
  for i in active_block_ids
    cell_dofs_i = get_cell_dof_ids(f.spaces[i],trian)
    if i == 1
      push!(active_block_data,cell_dofs_i)
    else
      offset = Int32(offsets[i])
      o = Fill(offset,length(cell_dofs_i))
      cell_dofs_i_b = lazy_map(Broadcasting(_sum_if_first_positive),cell_dofs_i,o)
      push!(active_block_data,cell_dofs_i_b)
    end
  end
  lazy_map(BlockMap(nfields,active_block_ids),active_block_data...)
end

function BlockExtensionSparseMatrixAssembler(
  trial::MultiFieldFESpace,
  test::MultiFieldFESpace,
  matrix_builder,
  vector_builder,
  strategy=FESpaces.DefaultAssemblyStrategy()
)
  msg = "BlockSparseMatrixAssembler: trial and test spaces must have BlockMultiFieldExtensionStyle"
  @assert isa(MultiFieldStyle(trial),BlockMultiFieldExtensionStyle) msg
  @assert isa(MultiFieldStyle(test),BlockMultiFieldExtensionStyle) msg

  NBr,SBr,Pr = MultiField.get_block_parameters(MultiFieldStyle(test))
  NBc,SBc,Pc = MultiField.get_block_parameters(MultiFieldStyle(trial))

  # Count block rows/cols
  block_rows = [sum(num_free_dofs,test.spaces[r]) for r in MultiField.get_block_ranges(NBr,SBr,Pr)]
  block_cols = [sum(num_free_dofs,trial.spaces[r]) for r in MultiField.get_block_ranges(NBc,SBc,Pc)]

  # Extension dofs
  test_dof_to_bg_dofs = [vcat(get_dof_to_bg_dof.(test.spaces[r])...) for r in MultiField.get_block_ranges(NBr,SBr,Pr)]
  trial_dof_to_bg_dofs = [vcat(get_dof_to_bg_dof.(trial.spaces[r])...) for r in MultiField.get_block_ranges(NBc,SBc,Pc)]

  # Create block assemblers
  block_idx = CartesianIndices((NBr,NBc))
  block_assemblers = map(block_idx) do idx
    rows = Base.OneTo(block_rows[idx[1]])
    cols = Base.OneTo(block_cols[idx[2]])
    assem = FESpaces.GenericSparseMatrixAssembler(
      matrix_builder,vector_builder,rows,cols,strategy
    )
    ExtensionAssembler(assem,trial_dof_to_bg_dofs[idx[2]],test_dof_to_bg_dofs[idx[1]])
  end

  R,C = (NBr,SBr,Pr),(NBc,SBc,Pc)
  return BlockSparseMatrixAssembler{R,C}(block_assemblers)
end

function FESpaces.SparseMatrixAssembler(
  mat,vec,
  trial::MultiFieldFESpace{<:BlockMultiFieldExtensionStyle},
  test ::MultiFieldFESpace{<:BlockMultiFieldExtensionStyle},
  strategy::AssemblyStrategy=DefaultAssemblyStrategy()
)
  BlockExtensionSparseMatrixAssembler(
    trial,test,SparseMatrixBuilder(mat),ArrayBuilder(vec),strategy
  )
end

function ParamFESpaces.MultiFieldParamFESpace(
  spaces::Vector{<:DirectSumTrialFESpace};
  style = BlockMultiFieldExtensionStyle())

  @notimplementedif !isa(style,BlockMultiFieldExtensionStyle)
  style = BlockMultiFieldExtensionStyle(style,spaces)
  fv = mortar(map(zero_free_values,spaces))
  V = typeof(fv)
  MultiFieldFESpace(V,spaces,style)
end

function MultiField._restrict_to_field(
  f,
  ::BlockMultiFieldExtensionStyle,
  free_values::AbstractParamVector,
  field)

  U = f.spaces
  offsets = MultiField._compute_field_offsets(U)
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  get_param_entry(free_values,pini:pend)
end

function MultiField._restrict_to_field(
  f,
  mfs::BlockMultiFieldExtensionStyle{NB,SB,P},
  free_values::BlockParamVector,
  field
  ) where {NB,SB,P}

  @check blocklength(free_values) == NB
  U = f.spaces

  # Find the block for this field
  block_ranges = MultiField.get_block_ranges(NB,SB,P)
  block_idx    = findfirst(range -> field ∈ range,block_ranges)
  block_free_values = blocks(free_values)[block_idx]

  # Within the block,restrict to field
  offsets = MultiField.compute_field_offsets(f,mfs)
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  return get_param_entry(block_free_values,pini:pend)
end

function FESpaces.zero_free_values(f::MultiFieldParamFESpace{<:BlockMultiFieldExtensionStyle})
  mortar(map(zero_free_values,f.spaces))
end

function FESpaces.zero_dirichlet_values(f::MultiFieldParamFESpace{<:BlockMultiFieldExtensionStyle})
  mortar(map(zero_dirichlet_values,f.spaces))
end