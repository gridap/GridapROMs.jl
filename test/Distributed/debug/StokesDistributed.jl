module StokesDistributedDebug

using PartitionedArrays, GridapPETSc
include("../StokesDistributed.jl")

petsc_options = "-ksp_error_if_not_converged true -sub_pc_type jacobi"

with_debug() do distribute
  GridapPETSc.with(;args=split(petsc_options)) do
    StokesDistributed.main(distribute,(2,2))
    GridapPETSc.gridap_petsc_gc()
  end
end

end
