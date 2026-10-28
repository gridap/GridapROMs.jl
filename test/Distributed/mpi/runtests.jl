using Test
using MPI

function run_tests(testdir,procs=4)
  istest(f) = endswith(f,".jl") && !(f == "runtests.jl")
  testfiles = sort(filter(istest,readdir(testdir)))
  @time @testset "$f" for f in testfiles
    MPI.mpiexec() do cmd
      if MPI.MPI_LIBRARY == "OpenMPI" || (isdefined(MPI,:OpenMPI) && MPI.MPI_LIBRARY == MPI.OpenMPI)
        run(`$cmd -n $procs --oversubscribe $(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(testdir,f))`)
      else
        run(`$cmd -n $procs $(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(testdir,f))`)
      end
      # Reached only if the subprocess launched by `run` exits without error.
      @test true
    end
  end
end

run_tests(@__DIR__)
