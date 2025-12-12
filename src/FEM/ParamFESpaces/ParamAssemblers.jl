function FESpaces.SparseMatrixAssembler(
  matrix_builder,
  vector_builder,
  rows::AbstractUnitRange,
  cols::AbstractUnitRange,
  strategy::AssemblyStrategy
  )

  GenericSparseMatrixAssembler(
    matrix_builder,
    vector_builder,
    rows,
    cols,
    strategy
    )
end

function FESpaces.SparseMatrixAssembler(
  trial::SingleFieldParamFESpace,
  test::SingleFieldFESpace
  )

  assem = SparseMatrixAssembler(get_fe_space(trial),test)
  parameterize(assem,param_length(trial))
end

"""
    parameterize(a::SparseMatrixAssembler,plength::Int) -> SparseMatrixAssembler

Returns an assembler that also stores the parametric length of `r`. This function
is to be used to assemble parametric residuals and Jacobians. The assembly routines
follow the same pipeline as in [`Gridap`](@ref)
"""
function ParamDataStructures.parameterize(a::SparseMatrixAssembler,plength::Int)
  matrix_builder = parameterize(get_matrix_builder(a),plength)
  vector_builder = parameterize(get_vector_builder(a),plength)
  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  strategy = FESpaces.get_assembly_strategy(a)
  GenericSparseMatrixAssembler(matrix_builder,vector_builder,rows,cols,strategy)
end

function ParamDataStructures.parameterize(
  a::MultiField.BlockSparseMatrixAssembler{R,C},
  plength::Int) where {R,C}

  matrix_builder = get_matrix_builder(a)
  vector_builder = get_vector_builder(a)
  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  strategy = FESpaces.get_assembly_strategy(a)
  block_idx = CartesianIndices((NB,NB))
  block_assemblers = map(block_idx) do idx
    mb = matrix_builder[idx[1],idx[2]]
    vb = vector_builder[idx[1]]
    r = rows[idx[1]]
    c = cols[idx[2]]
    s = strategy[idx[1],idx[2]]
    assem = SparseMatrixAssembler(mb,vb,r,c,s)
    parameterize(assem,plength)
  end
  MultiField.BlockSparseMatrixAssembler{R,C}(block_assemblers)
end

function FESpaces.assemble_vector_add!(
  b::BlockParamVector,
  a::MultiField.BlockSparseMatrixAssembler,
  vecdata)
  b1 = ArrayBlock(blocks(b),fill(true,blocksize(b)))
  b2 = MultiField.expand_blocks(a,b1)
  FESpaces.assemble_vector_add!(b2,a,vecdata)
end

function FESpaces.assemble_matrix_add!(
  mat::BlockParamMatrix,
  a::MultiField.BlockSparseMatrixAssembler,
  matdata)
  m1 = ArrayBlock(blocks(mat),fill(true,blocksize(mat)))
  m2 = MultiField.expand_blocks(a,m1)
  FESpaces.assemble_matrix_add!(m2,a,matdata)
end

function FESpaces.assemble_matrix_and_vector_add!(
  A::BlockParamMatrix,
  b::BlockParamVector,
  a::MultiField.BlockSparseMatrixAssembler,
  data)
  m1 = ArrayBlock(blocks(A),fill(true,blocksize(A)))
  m2 = MultiField.expand_blocks(a,m1)
  b1 = ArrayBlock(blocks(b),fill(true,blocksize(b)))
  b2 = MultiField.expand_blocks(a,b1)
  FESpaces.assemble_matrix_and_vector_add!(m2,b2,a,data)
end
