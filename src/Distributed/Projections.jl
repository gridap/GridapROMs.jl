function RBSteady.reduced_subspace(space::DistributedSingleFieldFESpace,subspace::Projection)
  SingleFieldRBSpace(space,subspace)
end

function RBSteady.reduced_subspace(space::DistributedMultiFieldFESpace,subspace::BlockProjection)
  MultiFieldRBSpace(space,subspace)
end