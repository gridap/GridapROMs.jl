FESpaces.num_rows(ext::Extension) = num_free_dofs(get_fe_space(ext))
FESpaces.num_cols(ext::Extension) = num_free_dofs(get_fe_space(ext))

abstract type OutValsStyle end
struct InsertIn <: OutValsStyle end
struct InsertOut <: OutValsStyle end
struct InsertInOut <: OutValsStyle end

struct ExtensionAssembler{S<:OutValsStyle} <: SparseMatrixAssembler
  style::S
  assem::SparseMatrixAssembler
  extension::Extension
  trial_fdof_to_bg_fdofs::AbstractVector
  test_fdof_to_bg_fdofs::AbstractVector
end

function ExtensionAssembler(trial::FESpace,test::FESpace,style=InsertInOut())
  bg_trial = get_bg_fe_space(trial)
  bg_test = get_bg_fe_space(test)
  assem = SparseMatrixAssembler(bg_trial,bg_test)
  extension = get_extension(trial) # must be equal to the test extension
  trial_fdof_to_bg_fdofs = get_in_fdof_to_bg_fdofs(trial)
  test_fdof_to_bg_fdofs = get_in_fdof_to_bg_fdofs(test)
  ExtensionAssembler(style,assem,extension,trial_fdof_to_bg_fdofs,test_fdof_to_bg_fdofs)
end

function ExtensionAssemblerInsertIn(trial::FESpace,test::FESpace,args...)
  ExtensionAssembler(trial,test,InsertIn())
end

function ExtensionAssemblerInsertOut(trial::FESpace,test::FESpace,args...)
  ExtensionAssembler(trial,test,InsertOut())
end

function ExtensionAssemblerInsertInOut(trial::FESpace,test::FESpace,args...)
  ExtensionAssembler(trial,test,InsertInOut())
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

function extend_vecdata(a::ExtensionAssembler{InsertIn},incut_vecdata)
  cellvals,cellrows = incut_vecdata
  for k in eachindex(cellrows)
    cellrows[k] = to_bg_cellrows(cellrows[k],a)
  end
  return (cellvals,cellrows)
end

function extend_matdata(a::ExtensionAssembler{InsertIn},incut_matdata)
  cellvals,cellrows,cellcols = incut_matdata
  for k in eachindex(cellrows)
    cellrows[k] = to_bg_cellrows(cellrows[k],a)
    cellcols[k] = to_bg_cellcols(cellcols[k],a)
  end
  return (cellvals,cellrows,cellcols)
end

#TODO since the extension is mutable, I cannot overwrite cellrows. Think about
# potential optimizations
function extend_vecdata(a::ExtensionAssembler{InsertOut},incut_vecdata)
  cellvals,out_cellrows = a.extension.vecdata
  cellrows = Any[to_bg_cellrows(out_cellrows[k],a.extension) for k in eachindex(out_cellrows)]
  return (cellvals,cellrows)
end

#TODO since the extension is mutable, I cannot overwrite cellrows/cellcols. Think about
# potential optimizations
function extend_matdata(a::ExtensionAssembler{InsertOut},incut_matdata)
  cellvals,out_cellrows,out_cellcols = a.extension.matdata
  cellrows = Any[to_bg_cellrows(out_cellrows[k],a.extension) for k in eachindex(out_cellrows)]
  cellcols = Any[to_bg_cellcols(out_cellcols[k],a.extension) for k in eachindex(out_cellrows)]
  return (cellvals,cellrows,cellcols)
end

function extend_vecdata(a::ExtensionAssembler{InsertInOut},incut_vecdata)
  cellvals,cellrows = incut_vecdata
  out_cellvals,out_cellrows = a.extension.vecdata
  for k in eachindex(cellrows)
    cellrows[k] = to_bg_cellrows(cellrows[k],a)
  end
  for (out_cellval,out_cellrow) in zip(out_cellvals,out_cellrows)
    push!(cellvals,out_cellval)
    push!(cellrows,to_bg_cellrows(out_cellrow,a.extension))
  end
  return (cellvals,cellrows)
end

function extend_matdata(a::ExtensionAssembler{InsertInOut},incut_matdata)
  cellvals,cellrows,cellcols = incut_matdata
  out_cellvals,out_cellrows,out_cellcols = a.extension.matdata
  for k in eachindex(cellrows)
    cellrows[k] = to_bg_cellrows(cellrows[k],a)
    cellcols[k] = to_bg_cellcols(cellcols[k],a)
  end
  for (out_cellval,out_cellrow,out_cellcol) in zip(out_cellvals,out_cellrows,out_cellcols)
    push!(cellvals,out_cellval)
    push!(cellrows,to_bg_cellrows(out_cellrow,a.extension))
    push!(cellcols,to_bg_cellcols(out_cellcol,a.extension))
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

# function FESpaces.assemble_vector(ext::Extension)
#   f = get_fe_space(ext)
#   assem = SparseMatrixAssembler(f,f)
#   assemble_vector(assem,ext.vecdata)
# end

# function assemble_extended_vector(ext::Extension)
#   f = get_fe_space(ext)
#   assem = ExtensionAssemblerInsertOut(f,f)
#   assemble_vector(assem,ext.vecdata)
# end

# function FESpaces.assemble_matrix(ext::Extension)
#   f = get_fe_space(ext)
#   assem = SparseMatrixAssembler(f,f)
#   assemble_matrix(assem,ext.matdata)
# end

# function assemble_extended_matrix(ext::Extension)
#   f = get_fe_space(ext)
#   assem = ExtensionAssemblerInsertOut(f,f)
#   assemble_matrix(assem,ext.matdata)
# end

# utils

function to_bg_cellrows(cellids,a::ExtensionAssembler)
  k = BGCellDofIds(cellids,a.test_fdof_to_bg_fdofs)
  lazy_map(k,1:length(cellids))
end

function to_bg_cellcols(cellids,a::ExtensionAssembler)
  k = BGCellDofIds(cellids,a.trial_fdof_to_bg_fdofs)
  lazy_map(k,1:length(cellids))
end

function to_bg_cellrows(cellids,ext::Extension)
  k = BGCellDofIds(cellids,ext.fdof_to_bg_fdofs)
  lazy_map(k,1:length(cellids))
end

function to_bg_cellcols(cellids,ext::Extension)
  k = BGCellDofIds(cellids,ext.fdof_to_bg_fdofs)
  lazy_map(k,1:length(cellids))
end

function ParamDataStructures.parameterize(a::ExtensionAssembler,r::AbstractRealization)
  assem = parameterize(a.assem,r)
  extension = a.extension(r)
  ExtensionAssembler(a.style,assem,extension,a.trial_fdof_to_bg_fdofs,a.test_fdof_to_bg_fdofs)
end

function ParamSteady._assemble_matrix(f,U::FESpace,V::ExtensionFESpace)
  ParamSteady._assemble_matrix(f,get_bg_fe_space(U),get_bg_fe_space(V))
end

function DofMaps.SparsityPattern(
  U::SingleFieldFESpace,
  V::ExtensionFESpace,
  trian=DofMaps._get_common_domain(U,V)
  )

  a = ExtensionAssembler(U,V)
  m1 = nz_counter(FESpaces.get_matrix_builder(a),(FESpaces.get_rows(a),FESpaces.get_cols(a)))
  cellidsrows = get_bg_cell_dof_ids(V,trian)
  cellidscols = get_bg_cell_dof_ids(U,trian)
  DofMaps.trivial_symbolic_loop_matrix!(m1,cellidsrows,cellidscols)
  m2 = Algebra.nz_allocation(m1)
  DofMaps.trivial_symbolic_loop_matrix!(m2,cellidsrows,cellidscols)
  m3 = Algebra.create_from_nz(m2)
  SparsityPattern(m3)
end
