module SaveSteadyOperator

using Test
using GridapROMs
using GridapROMs.RBSteady

@testset "load_operator method for LinearNonlinearParamOperator" begin
  # Verify the internal helper function used by load_operator exists
  @test isdefined(RBSteady, :_load_fixed_operator_parts)

  # Verify the typo'd name does NOT exist (regression guard)
  @test !isdefined(RBSteady, :_fixed_operator_parts)

  # Verify load_operator is defined for LinearNonlinearParamOperator
  @test hasmethod(load_operator, Tuple{String, LinearNonlinearParamOperator})

  # Verify the source calls _load_fixed_operator_parts, not _fixed_operator_parts
  src = read(joinpath(pkgdir(GridapROMs), "src", "RB", "RBSteady", "PostProcess.jl"), String)
  m = match(r"function load_operator\(dir,feop::LinearNonlinearParamOperator.*?\nend"s, src)
  @test m !== nothing
  @test occursin("_load_fixed_operator_parts(", m.match)
  @test !occursin(r"= _fixed_operator_parts\(", m.match)
end

end
