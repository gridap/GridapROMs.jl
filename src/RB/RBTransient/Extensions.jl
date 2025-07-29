for T in (
  :(RBSpace{<:DirectSumFESpace}),
  :(RBSpace{<:AbstractTrialFESpace{<:DirectSumFESpace}})
  )
  @eval begin

    function RBSteady.IntegrationDomain(
      ::Type{<:SequentialProjection},
      trian::Triangulation,
      test::$T,
      rows::AbstractVector,
      indices_time::AbstractVector)

      cell_row_ids = get_bg_cell_dof_ids(test,trian)
      cells = RBSteady.get_rows_to_cells(cell_row_ids,rows)
      irows = get_spacetime_irows(cell_row_ids,cells,rows,indices_times)
      domain_space = IntegrationDomain(cells,irows,rows)
      TransientIntegrationDomain(SequentialDomain(),domain_space,indices_time)
    end

    function RBSteady.IntegrationDomain(
      ::Type{<:SequentialProjection},
      trian::Triangulation,
      trial::$T,
      test::$T,
      rows::AbstractVector,
      cols::AbstractVector,
      indices_time::AbstractVector
      )

      cell_row_ids = get_bg_cell_dof_ids(test,trian)
      cell_col_ids = get_bg_cell_dof_ids(trial,trian)
      cells = RBSteady.get_rowcols_to_cells(cell_row_ids,cell_col_ids,rows,cols)
      irowcols = get_spacetime_irowcols(cell_row_ids,cell_col_ids,cells,rows,cols,indices_time)
      domain_space = IntegrationDomain(cells,irowcols,(rows,cols))
      TransientIntegrationDomain(SequentialDomain(),domain_space,indices_time)
    end
  end
end
