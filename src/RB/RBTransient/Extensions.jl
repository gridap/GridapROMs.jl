for T in (
  :(RBSpace{<:DirectSumFESpace}),
  :(RBSpace{<:AbstractTrialFESpace{<:DirectSumFESpace}})
  )
  @eval begin
    function reduced_spacetime_idofs(
      f::$T,
      trian::Triangulation,
      cells::AbstractVector,
      dofs::AbstractVector,
      indices_times::AbstractVector)

      cell_dof_ids = get_bg_cell_dof_ids(f,trian)
      idofs = get_cells_to_spacetime_idofs(cell_dof_ids,cells,dofs,indices_times)
      return idofs
    end
  end
end
