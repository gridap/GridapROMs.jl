function RBSteady.hr_error(
  red::HighDimHyperReduction,
  trial::SingleFieldRBSpace,
  test::SingleFieldRBSpace,
  A::AbstractParamMatrix,
  Â::AbstractParamMatrix,
  μ::AbstractRealisation
  )

  basis_left = get_basis(test)
  pdata = get_param_data(A)
  data = get_all_data(pdata)
  basis_right = get_basis(trial)
  c = get_time_combination(red)
  Aproj = galerkin_projection(basis_left,data,basis_right,c)
  
  i = VectorDofMap(size(testitem(pdata)))
  Âsnaps = Snapshots(Â,i,μ)
  Asnaps = Snapshots(Aproj,i,μ)
  
  compute_relative_error(Âsnaps,Asnaps)
end