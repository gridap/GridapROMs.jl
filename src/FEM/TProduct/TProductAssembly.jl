"""
    struct TProductSparseMatrixAssembler{A<:SparseMatrixAssembler} <: SparseMatrixAssembler
      assems_1d::Vector{A}
    end

A `SparseMatrixAssembler` for tensor product FE spaces. Wraps `D` 1D
`SparseMatrixAssembler`s, one per spatial direction.

Assembly operates direction-by-direction: integrating a bilinear form against
a [`TProductMeasure`](@ref) yields a vector of `D` `DomainContribution`s, and
the assembler collects and assembles each into a 1D sparse matrix. The results
are combined into an [`AbstractRankTensor`](@ref):

- A [`Rank1Tensor`](@ref) for forms without derivatives (e.g. mass matrix).
- A [`GenericRankTensor`](@ref) for forms involving `gradient` or
  `PartialDerivative` (e.g. stiffness matrix), where the rank equals the
  spatial dimension `D`.

# Construction

    TProductSparseMatrixAssembler(trial::TProductFESpace,test::TProductFESpace)
    TProductSparseMatrixAssembler(mat,trial::TProductFESpace,test::TProductFESpace)
    TProductSparseMatrixAssembler(mat,vec,trial,test[,strategy])

For multi-field scenarios use [`TProductBlockSparseMatrixAssembler`](@ref).
"""
struct TProductSparseMatrixAssembler{A<:SparseMatrixAssembler} <: SparseMatrixAssembler
  assems_1d::Vector{A}
end

function TProductSparseMatrixAssembler(
  mat,
  vec,
  trial::TProductFESpace,
  test::TProductFESpace,
  strategy::AssemblyStrategy=DefaultAssemblyStrategy())

  assems_1d = map(trial.spaces_1d,test.spaces_1d) do U,V
    SparseMatrixAssembler(mat,vec,U,V,strategy)
  end
  TProductSparseMatrixAssembler(assems_1d)
end

function TProductSparseMatrixAssembler(mat,trial::TProductFESpace,test::TProductFESpace)
  mat_builder = SparseMatrixBuilder(mat)
  T = eltype(get_array_type(mat_builder))
  TProductSparseMatrixAssembler(mat_builder,Vector{T},trial,test)
end

function TProductSparseMatrixAssembler(trial::TProductFESpace,test::TProductFESpace)
  T = get_dof_value_type(trial)
  matrix_type = SparseMatrixCSC{T,Int}
  vector_type = Vector{T}
  TProductSparseMatrixAssembler(matrix_type,vector_type,trial,test)
end

function FESpaces.collect_cell_matrix(
  trial::TProductFESpace,
  test::TProductFESpace,
  a::Vector{<:DomainContribution})

  map(collect_cell_matrix,trial.spaces_1d,test.spaces_1d,a)
end

function FESpaces.collect_cell_vector(
  test::TProductFESpace,
  a::Vector{<:DomainContribution})

  map(collect_cell_vector,test.spaces_1d,a)
end

function FESpaces.collect_cell_matrix(
  trial::MultiFieldFESpace,
  test::MultiFieldFESpace,
  a::Vector{<:DomainContribution})

  map(eachindex(a)) do d
    trials_d = map(f->_remove_trial(f).spaces_1d[d],trial.spaces)
    tests_d = map(f->f.spaces_1d[d],test.spaces)
    trial′ = MultiFieldFESpace(trial.vector_type,trials_d,trial.multi_field_style)
    test′ = MultiFieldFESpace(test.vector_type,tests_d,test.multi_field_style)
    collect_cell_matrix(trial′,test′,a[d])
  end
end

function FESpaces.collect_cell_vector(
  test::MultiFieldFESpace,
  a::Vector{<:DomainContribution})

  map(eachindex(a)) do d
    tests_d = map(f->f.spaces_1d[d],test.spaces)
    test′ = MultiFieldFESpace(test.vector_type,tests_d,test.multi_field_style)
    collect_cell_vector(test′,a[d])
  end
end

function FESpaces.collect_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::GenericTProductDiffEval)

  f = collect_cell_matrix(trial,test,get_data(a))
  g = collect_cell_matrix(trial,test,get_diff_data(a))
  GenericTProductDiffEval(a.op,f,g,a.summation)
end

function FESpaces.collect_cell_vector(
  test::FESpace,
  a::GenericTProductDiffEval)

  f = collect_cell_vector(test,get_data(a))
  g = collect_cell_vector(test,get_diff_data(a))
  GenericTProductDiffEval(a.op,f,g,a.summation)
end

function FESpaces.allocate_vector(a::TProductSparseMatrixAssembler,vecdata::Vector)
  vecs_1d = map(allocate_vector,a.assems_1d,vecdata)
  return tproduct_array(vecs_1d)
end

function FESpaces.assemble_vector!(b::Rank1Tensor,a::TProductSparseMatrixAssembler,vecdata::Vector)
  map(assemble_vector!,get_factors(b),a.assems_1d,vecdata)
end

function FESpaces.assemble_vector_add!(b::Rank1Tensor,a::TProductSparseMatrixAssembler,vecdata::Vector)
  map(assemble_vector_add!,get_factors(b),a.assems_1d,vecdata)
end

function FESpaces.assemble_vector(a::TProductSparseMatrixAssembler,vecdata::Vector)
  vecs_1d = map(assemble_vector,a.assems_1d,vecdata)
  return tproduct_array(vecs_1d)
end

function FESpaces.allocate_matrix(a::TProductSparseMatrixAssembler,matdata::Vector)
  mats_1d = map(allocate_matrix,a.assems_1d,matdata)
  return tproduct_array(mats_1d)
end

function FESpaces.assemble_matrix!(A::Rank1Tensor,a::TProductSparseMatrixAssembler,matdata::Vector)
  map(assemble_matrix!,get_factors(A),a.assems_1d,matdata)
end

function FESpaces.assemble_matrix_add!(A::Rank1Tensor,a::TProductSparseMatrixAssembler,matdata::Vector)
  map(assemble_matrix_add!,get_factors(A),a.assems_1d,matdata)
end

function FESpaces.assemble_matrix(a::TProductSparseMatrixAssembler,matdata::Vector)
  mats_1d = map(assemble_matrix,a.assems_1d,matdata)
  return tproduct_array(mats_1d)
end

function FESpaces.allocate_vector(a::TProductSparseMatrixAssembler,vecdata::GenericTProductDiffEval)
  vecs_1d = map(allocate_vector,a.assems_1d,vecdata.f)
  gradvecs_1d = map(allocate_vector,a.assems_1d,vecdata.g)
  return tproduct_array(vecdata.op,vecs_1d,gradvecs_1d,vecdata.summation)
end

function FESpaces.assemble_vector!(b::GenericRankTensor,a::TProductSparseMatrixAssembler,vecdata::GenericTProductDiffEval)
  map(assemble_vector!,get_arrays_1d(b),a.assems_1d,vecdata.f)
  map(assemble_vector!,get_gradients_1d(b),a.assems_1d,vecdata.g)
end

function FESpaces.assemble_vector_add!(b::GenericRankTensor,a::TProductSparseMatrixAssembler,vecdata::GenericTProductDiffEval)
  map(assemble_vector_add!,get_arrays_1d(b),a.assems_1d,vecdata.f)
  map(assemble_vector_add!,get_gradients_1d(b),a.assems_1d,vecdata.g)
end

function FESpaces.assemble_vector(a::TProductSparseMatrixAssembler,vecdata::GenericTProductDiffEval)
  vecs_1d = map(assemble_vector,a.assems_1d,vecdata.f)
  gradvecs_1d = map(assemble_vector,a.assems_1d,vecdata.g)
  return tproduct_array(vecdata.op,vecs_1d,gradvecs_1d,vecdata.summation)
end

function FESpaces.allocate_matrix(a::TProductSparseMatrixAssembler,matdata::GenericTProductDiffEval)
  mats_1d = map(allocate_matrix,a.assems_1d,matdata.f)
  gradmats_1d = map(allocate_matrix,a.assems_1d,matdata.g)
  return tproduct_array(matdata.op,mats_1d,gradmats_1d,matdata.summation)
end

function FESpaces.assemble_matrix!(A::GenericRankTensor,a::TProductSparseMatrixAssembler,matdata::GenericTProductDiffEval)
  map(assemble_matrix!,get_arrays_1d(A),a.assems_1d,matdata.f)
  map(assemble_matrix!,get_gradients_1d(A),a.assems_1d,matdata.g)
end

function FESpaces.assemble_matrix_add!(A::GenericRankTensor,a::TProductSparseMatrixAssembler,matdata::GenericTProductDiffEval)
  map(assemble_matrix_add!,get_arrays_1d(A),a.assems_1d,matdata.f)
  map(assemble_matrix_add!,get_gradients_1d(A),a.assems_1d,matdata.g)
end

function FESpaces.assemble_matrix(a::TProductSparseMatrixAssembler,matdata::GenericTProductDiffEval)
  mats_1d = map(assemble_matrix,a.assems_1d,matdata.f)
  gradmats_1d = map(assemble_matrix,a.assems_1d,matdata.g)
  return tproduct_array(matdata.op,mats_1d,gradmats_1d,matdata.summation)
end

# multi field

"""
    TProductBlockSparseMatrixAssembler(trial::MultiFieldFESpace,test::MultiFieldFESpace
      ) -> TProductSparseMatrixAssembler

Returns a [`TProductSparseMatrixAssembler`](@ref) in a MultiField scenario
"""
function TProductBlockSparseMatrixAssembler(trial::MultiFieldFESpace,test::MultiFieldFESpace)
  assems_1d = map(eachindex(test.spaces[1].spaces_1d)) do d
    trials_d = map(f->_remove_trial(f).spaces_1d[d],trial.spaces)
    tests_d = map(f->f.spaces_1d[d],test.spaces)
    trial′ = MultiFieldFESpace(trial.vector_type,trials_d,trial.multi_field_style)
    test′ = MultiFieldFESpace(test.vector_type,tests_d,test.multi_field_style)
    SparseMatrixAssembler(trial′,test′)
  end
  TProductSparseMatrixAssembler(assems_1d)
end
