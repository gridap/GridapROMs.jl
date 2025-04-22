struct ExtensionAssembler <: SparseMatrixAssembler
  assem::SparseMatrixAssembler
  trial_fdof_to_bg_fdofs::AbstractVector
  test_fdof_to_bg_fdofs::AbstractVector
end

function ExtensionAssembler(trial::FESpace,test::FESpace)
  bg_trial = get_bg_space(trial)
  bg_test = get_bg_space(test)
  assem = SparseMatrixAssembler(bg_trial,bg_test)
  trial_fdof_to_bg_fdofs = get_fdof_to_bg_fdof(trial)
  test_fdof_to_bg_fdofs = get_fdof_to_bg_fdof(test)
  ExtensionAssembler(assem,trial_fdof_to_bg_fdofs,test_fdof_to_bg_fdofs)
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

function extend_vecdata(a::ExtensionAssembler,act_vecdata)
  cellvals,cellrows = act_vecdata
  for k in eachindex(cellrows)
    cellrows[k] = to_bg_cellrows(cellrows[k],a)
  end
  return (cellvals,cellrows)
end

function extend_matdata(a::ExtensionAssembler,act_matdata)
  cellvals,cellrows,cellcols = act_matdata
  for k in eachindex(cellrows)
    cellrows[k] = to_bg_cellrows(cellrows[k],a)
    cellcols[k] = to_bg_cellcols(cellcols[k],a)
  end
  return (cellvals,cellrows,cellcols)
end

function FESpaces.allocate_vector(a::ExtensionAssembler,vecdata)
  bg_vecdata = extend_vecdata(a,vecdata)
  allocate_vector(a.assem,bg_vecdata)
end

function FESpaces.assemble_vector(a::ExtensionAssembler,vecdata)
  bg_vecdata = extend_vecdata(a,vecdata)
  assemble_vector(a.assem,bg_vecdata)
end

function FESpaces.assemble_vector!(b,a::ExtensionAssembler,vecdata)
  bg_vecdata = extend_vecdata(a,vecdata)
  assemble_vector!(b,a.assem,bg_vecdata)
  b
end

function FESpaces.assemble_vector_add!(b,a::ExtensionAssembler,vecdata)
  bg_vecdata = extend_vecdata(a,vecdata)
  assemble_vector_add!(b,a.assem,bg_vecdata)
  b
end

function FESpaces.allocate_matrix(a::ExtensionAssembler,matdata)
  bg_matdata = extend_matdata(a,matdata)
  allocate_matrix(a.assem,bg_matdata)
end

function FESpaces.assemble_matrix(a::ExtensionAssembler,matdata)
  bg_matdata = extend_matdata(a,matdata)
  assemble_matrix(a.assem,bg_matdata)
end

function FESpaces.assemble_matrix!(A,a::ExtensionAssembler,matdata)
  bg_matdata = extend_matdata(a,matdata)
  assemble_matrix!(A,a.assem,bg_matdata)
  A
end

function FESpaces.assemble_matrix_add!(A,a::ExtensionAssembler,matdata)
  bg_matdata = extend_matdata(a,matdata)
  assemble_matrix_add!(A,a.assem,bg_matdata)
  A
end

# utils

function to_bg_cellrows(cellids,a::ExtensionAssembler)
  k = BGCellDofIds(cellids,a.test_fdof_to_bg_fdofs)
  lazy_map(k,1:length(cellids))
end

function to_bg_cellcols(cellids,a::ExtensionAssembler)
  k = BGCellDofIds(cellids,a.trial_fdof_to_bg_fdofs)
  lazy_map(k,1:length(cellids))
end

function ParamDataStructures.parameterize(a::ExtensionAssembler,r::AbstractRealization)
  assem = parameterize(a.assem,r)
  ExtensionAssembler(assem,a.trial_fdof_to_bg_fdofs,a.test_fdof_to_bg_fdofs)
end
