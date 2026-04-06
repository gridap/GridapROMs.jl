module BlockProjectionNormMatrixTest

using Test
using GridapROMs
using GridapROMs.RBSteady
using BlockArrays
using LinearAlgebra

function main()
  basis1 = rand(10,3)
  basis2 = rand(10,2)
  proj1 = PODProjection(basis1)
  proj2 = PODProjection(basis2)

  array = [proj1,proj2]
  touched = [true,true]
  bp = BlockProjection(array,touched)

  norm_matrix = get_norm_matrix(bp)
  @test norm_matrix isa AbstractMatrix
  @test size(norm_matrix) == (20,20)
  @test norm_matrix[Block(1,1)] == I(10)
  @test norm_matrix[Block(2,2)] == I(10)
end

main()

end
