module StokesDistributedMPI

using MPI, PartitionedArrays, GridapPETSc
include("../StokesDistributed.jl")

petsc_options = "-sub_pc_type jacobi"

with_mpi() do distribute
  GridapPETSc.with(;args=split(petsc_options)) do
    StokesDistributed.main(distribute,(2,2))
    GridapPETSc.gridap_petsc_gc()
  end
end

end
