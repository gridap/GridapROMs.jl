abstract type ExtensionAssemblerStyle end



"""
    struct ExtensionAssembler <: SparseMatrixAssembler
      assem::SparseMatrixAssembler
      trial_dof_to_bg_dofs::NTuple{2,AbstractVector}
      test_dof_to_bg_dofs::NTuple{2,AbstractVector}
    end

Structure that allows to decouple the assembly of FE matrices/vectors from the
integration of weak formulations, to be used exclusively in the context of a trial/test
couple of [`DirectSumFESpace`](@ref). After performing integration on the FE space,
the assembly is done on the background space. The latter step can be done by
exploiting the fields
- `assem`: a SparseMatrixAssembler defined on the FE space
- `trial_dof_to_bg_dofs`: index maps from the free/Dirichlet dofs on the trial FE space
  to those on the background trial space
- `test_dof_to_bg_dofs`: index maps from the free/Dirichlet dofs on the test FE space
  to those on the background test space
"""
struct ExtensionAssembler <: SparseMatrixAssembler
  assem::SparseMatrixAssembler
  trial_dof_to_bg_dofs::NTuple{2,AbstractVector}
  test_dof_to_bg_dofs::NTuple{2,AbstractVector}
end

function ExtensionAssembler(trial::SingleFieldFESpace,test::SingleFieldFESpace)
  bg_trial = get_bg_space(trial)
  bg_test = get_bg_space(test)
  assem = SparseMatrixAssembler(bg_trial,bg_test)
  trial_dof_to_bg_dofs = get_dof_to_bg_dof(trial)
  test_dof_to_bg_dofs = get_dof_to_bg_dof(test)
  ExtensionAssembler(assem,trial_dof_to_bg_dofs,test_dof_to_bg_dofs)
end

function ExtensionAssembler(
  ::BlockMultiFieldStyle{R,C},
  trial::MultiFieldFESpace,
  test::MultiFieldFESpace
  ) where {R,C}

  NV = length(test.spaces)
  block_idx = CartesianIndices((NB,NB))
  block_assem = map(block_idx) do idx
    ExtensionAssembler(trial[idx[2]],test[idx[1]])
  end
  BlockSparseMatrixAssembler{R,C}(block_assem)
end

function ExtensionAssembler(trial::MultiFieldFESpace,test::MultiFieldFESpace)
  mfs = MultiFieldStyle(test)
  ExtensionAssembler(mfs,trial,test)
end

FESpaces.get_vector_type(a::ExtensionAssembler) = get_vector_type(a.assem)
FESpaces.get_matrix_type(a::ExtensionAssembler) = get_matrix_type(a.assem)
FESpaces.num_rows(a::ExtensionAssembler) = FESpaces.num_rows(a.assem)
FESpaces.num_cols(a::ExtensionAssembler) = FESpaces.num_cols(a.assem)
FESpaces.get_rows(a::ExtensionAssembler) = FESpaces.get_rows(a.assem)
FESpaces.get_cols(a::ExtensionAssembler) = FESpaces.get_cols(a.assem)
FESpaces.get_assembly_strategy(a::ExtensionAssembler) = FESpaces.get_assembly_strategy(a.assem)
FESpaces.get_matrix_builder(a::ExtensionAssembler)= get_matrix_builder(a.assem)
FESpaces.get_vector_builder(a::ExtensionAssembler) = get_vector_builder(a.assem)


const BlockExtensionAssembler{R,C} = BlockSparseMatrixAssembler{R,C,ExtensionAssembler}
const AbstractExtensionAssembler = Union{ExtensionAssembler,BlockExtensionAssembler}

get_assem(a::ExtensionAssembler) = a.assem
get_rows_to_bg_rows(a::ExtensionAssembler) = a.test_dof_to_bg_dofs[1]
get_cols_to_bg_cols(a::ExtensionAssembler) = a.trial_dof_to_bg_dofs[1]
get_drows_to_bg_drows(a::ExtensionAssembler) = a.test_dof_to_bg_dofs[2]
get_dcols_to_bg_dcols(a::ExtensionAssembler) = a.trial_dof_to_bg_dofs[2]

function get_assem(a::BlockExtensionAssembler{R,C}) where {R,C}
  block_assem = map(get_assem,a.block_assemblers)
  BlockSparseMatrixAssembler{R,C}(block_assem)
end

for f in (:get_rows_to_bg_rows,:get_drows_to_bg_drows)
  @eval begin
    function $f(a::BlockExtensionAssembler{NB}) where NB
      ArrayBlock(map(i -> $f(a.block_assemblers[i,1]),1:NB),fill(true,NB))
    end
  end
end

for f in (:get_cols_to_bg_cols,:get_dcols_to_bg_dcols)
  @eval begin
    function $f(a::BlockExtensionAssembler{NB}) where NB
      ArrayBlock(map(i -> $f(a.block_assemblers[1,i]),1:NB),fill(true,NB))
    end
  end
end

function extend_vecdata(a::AbstractExtensionAssembler,act_vecdata)
  cellvals,cellrows = act_vecdata
  for k in eachindex(cellrows)
    cellrows[k] = to_bg_cellrows(cellrows[k],a)
  end
  return (cellvals,cellrows)
end

function extend_matdata(a::AbstractExtensionAssembler,act_matdata)
  cellvals,cellrows,cellcols = act_matdata
  for k in eachindex(cellrows)
    cellrows[k] = to_bg_cellrows(cellrows[k],a)
    cellcols[k] = to_bg_cellcols(cellcols[k],a)
  end
  return (cellvals,cellrows,cellcols)
end

function FESpaces.allocate_vector(a::AbstractExtensionAssembler,vecdata)
  bg_vecdata = extend_vecdata(a,vecdata)
  allocate_vector(get_assem(a),bg_vecdata)
end

function FESpaces.assemble_vector(a::AbstractExtensionAssembler,vecdata)
  bg_vecdata = extend_vecdata(a,vecdata)
  assemble_vector(get_assem(a),bg_vecdata)
end

function FESpaces.allocate_matrix(a::AbstractExtensionAssembler,matdata)
  bg_matdata = extend_matdata(a,matdata)
  allocate_matrix(get_assem(a),bg_matdata)
end

function FESpaces.assemble_matrix(a::AbstractExtensionAssembler,matdata)
  bg_matdata = extend_matdata(a,matdata)
  assemble_matrix(get_assem(a),bg_matdata)
end

function FESpaces.assemble_vector!(b,a::ExtensionAssembler,vecdata)
  bg_vecdata = extend_vecdata(a,vecdata)
  assemble_vector!(b,get_assem(a),bg_vecdata)
  b
end

function FESpaces.assemble_vector_add!(b,a::ExtensionAssembler,vecdata)
  bg_vecdata = extend_vecdata(a,vecdata)
  assemble_vector_add!(b,get_assem(a),bg_vecdata)
  b
end

function FESpaces.assemble_matrix!(A,a::ExtensionAssembler,matdata)
  bg_matdata = extend_matdata(a,matdata)
  assemble_matrix!(A,get_assem(a),bg_matdata)
  A
end

function FESpaces.assemble_matrix_add!(A,a::ExtensionAssembler,matdata)
  bg_matdata = extend_matdata(a,matdata)
  assemble_matrix_add!(A,get_assem(a),bg_matdata)
  A
end

for T in (:AbstractBlockVector,:BlockParamVector)
  @eval begin
    function FESpaces.assemble_vector!(b::$T,a::BlockExtensionAssembler,vecdata)
      bg_vecdata = extend_vecdata(a,vecdata)
      assemble_vector!(b,get_assem(a),bg_vecdata)
      b
    end

    function FESpaces.assemble_vector_add!(b::$T,a::BlockExtensionAssembler,vecdata)
      bg_vecdata = extend_vecdata(a,vecdata)
      assemble_vector_add!(b,get_assem(a),bg_vecdata)
      b
    end
  end
end

for T in (:AbstractBlockMatrix,:BlockParamMatrix)
  @eval begin
    function FESpaces.assemble_matrix!(A::$T,a::BlockExtensionAssembler,matdata)
      bg_matdata = extend_matdata(a,matdata)
      assemble_matrix!(A,get_assem(a),bg_matdata)
      A
    end

    function FESpaces.assemble_matrix_add!(A::$T,a::BlockExtensionAssembler,matdata)
      bg_matdata = extend_matdata(a,matdata)
      assemble_matrix_add!(A,get_assem(a),bg_matdata)
      A
    end
  end
end

# utils

function to_bg_cellrows(cellids,a::AbstractExtensionAssembler)
  k = BGCellDofIds(cellids,get_rows_to_bg_rows(a),get_drows_to_bg_drows(a))
  lazy_map(k,1:length(cellids))
end

function to_bg_cellcols(cellids,a::AbstractExtensionAssembler)
  k = BGCellDofIds(cellids,get_cols_to_bg_cols(a),get_dcols_to_bg_dcols(a))
  lazy_map(k,1:length(cellids))
end

function ParamDataStructures.parameterize(a::ExtensionAssembler,plength::Int)
  assem = parameterize(get_assem(a),plength)
  ExtensionAssembler(assem,a.trial_dof_to_bg_dofs,a.test_dof_to_bg_dofs)
end

function ParamDataStructures.parameterize(
  a::BlockExtensionAssembler{R,C},
  plength::Int) where {R,C}

  block_assemblers = map(eachindex(a.block_assemblers)) do idx
    parameterize(a.block_assemblers[idx],plength)
  end
  BlockSparseMatrixAssembler{R,C}(block_assemblers)
end
