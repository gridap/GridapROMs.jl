module SnapshotsTests

using Test
using GridapROMs

# Regression test: get_mode2 was constructing ModeTransientSnapshots with s.data
# (an N-dimensional array) instead of the computed m2 matrix (2D). For any standard
# TransientSnapshots backed by 3D+ data this caused a MethodError at the
# ModeTransientSnapshots constructor, which requires A<:AbstractMatrix{T}.
@testset "get_mode2 uses computed m2 matrix" begin
  ns, np, nt = 5, 3, 4
  params = Realization([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  r = TransientRealization(params, [0.1, 0.2, 0.3, 0.4], 0.0)
  i = VectorDofMap(ns)
  data = reshape(collect(1.0:Float64(ns*np*nt)), ns, np, nt)
  s = GenericSnapshots(data, i, r)

  m2 = get_mode2(s)

  @test isa(m2, ModeTransientSnapshots)
  @test isa(get_all_data(m2), AbstractMatrix)
  @test size(get_all_data(m2)) == (nt, ns * np)
  @test get_all_data(m2) == change_mode(reshape(data, ns, :), np)
end

end # module
