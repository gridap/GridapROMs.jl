module StokesDistributedMPI

using MPI, PartitionedArrays, GridapPETSc
include("../StokesDistributed.jl")

petsc_options = "-sub_pc_type jacobi"

with_mpi() do distribute
  GridapPETSc.with(;args=split(petsc_options)) do
    for compression in (:local,:global), hypred_strategy in (:none,:affine,:deim,:sopt,:rbf)
      StokesDistributed.main(distribute,(2,2),compression,hypred_strategy)
      GridapPETSc.gridap_petsc_gc()
    end
  end
end

end
