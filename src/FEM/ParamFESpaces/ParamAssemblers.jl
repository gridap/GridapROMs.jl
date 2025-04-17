"""
    parameterize(a::SparseMatrixAssembler,r::AbstractRealization) -> SparseMatrixAssembler

Returns an assembler that also stores the parametric length of `r`. This function
is to be used to assemble parametric residuals and Jacobians. The assembly routines
follow the same pipeline as in [`Gridap`](@ref)
"""
function ParamDataStructures.parameterize(a::SparseMatrixAssembler,r::AbstractRealization)
  matrix_builder = parameterize(get_matrix_builder(a),r)
  vector_builder = parameterize(get_vector_builder(a),r)
  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  strategy = FESpaces.get_assembly_strategy(a)
  GenericSparseMatrixAssembler(matrix_builder,vector_builder,rows,cols,strategy)
end

function ParamDataStructures.parameterize(
  a::MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P},
  r::AbstractRealization) where {NB,NV,SB,P}

  matrix_builder = parameterize(_getfirst(get_matrix_builder(a)),r)
  vector_builder = parameterize(_getfirst(get_vector_builder(a)),r)
  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  strategy = FESpaces.get_assembly_strategy(a)
  block_idx = CartesianIndices((NB,NB))
  block_assemblers = map(block_idx) do idx
    r = rows[idx[1]]
    c = cols[idx[2]]
    s = strategy[idx[1],idx[2]]
    GenericSparseMatrixAssembler(matrix_builder,vector_builder,r,c,s)
  end
  MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P}(block_assemblers)
end

_getfirst(a::Fields.ArrayBlock) = a[findfirst(a.touched)]
_getfirst(a::Fields.ArrayBlockView) = _getfirst(a.array)

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
