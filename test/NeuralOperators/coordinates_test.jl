using Gridap
using Gridap.FESpaces
using Gridap.Geometry
using Gridap.ReferenceFEs
using GridapROMs
using GridapROMs.Utils
using GridapROMs.DofMaps

function get_coords_method1(space::SingleFieldFESpace)
  orders = GridapROMs.Utils.get_polynomial_orders(space)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  get_coords_method1(model,orders)
end

function get_coords_method1(
  model::CartesianDiscreteModel{D},
  orders::NTuple{D,Int}
) where {D}
  desc = get_cartesian_descriptor(model)
  periodic = desc.isperiodic
  ncells = desc.partition
  sizes = desc.sizes
  x0 = desc.origin
  cells = CartesianIndices(ncells)
  nodes = CartesianIndices(orders .* ncells .+ 1 .- periodic)
  coords = Array{NTuple{D,Float64}}(undef,size(nodes))
  for cell in cells
    first_new_node = orders .* (Tuple(cell) .- 1) .+ 1
    nodes_range = map(enumerate(first_new_node)) do (i,ni)
      ni:(ni+orders[i])
    end
    for inode in Iterators.product(nodes_range...)
      _is_periodic_node(inode,nodes) && continue
      coords[inode...] = ntuple(d -> x0[d] + (inode[d]-1)*sizes[d],Val{D}())
    end
  end
  return coords
end

function _is_periodic_node(inode,nodes)
    try
        nodes[inode]
        return false
    catch
        return true
    end
end

function get_coords_method2(V::SingleFieldFESpace)
  # Retrieve the underlying triangulation and its physical dimensionality (1D,2D,3D)
  trian = get_triangulation(V)
  D_phys = length(get_node_coordinates(trian)[1])
  
  # Get the exact number of free DoFs (automatically excluding Dirichlet boundaries)
  N_dofs = num_free_dofs(V)

  # Initialize the tensor that will feed the Trunk Net: shape (D_phys,N_dofs)
  x_raw = zeros(Float32,D_phys,N_dofs)
  
  # Extract coordinates dimension by dimension. This avoids TypeErrors 
  # when interpolating a physical vector (Point) into a purely scalar FESpace.
  for d in 1:D_phys
    # Define a scalar spatial function for the d-th physical dimension
    coord_d(x) = x[d]
    
    # Interpolate the coordinate field over the entire FESpace.
    # This maps the physical space to the algebraic DoF numbering.
    coord_fn = interpolate_everywhere(coord_d,V)
    
    # Extract only the values corresponding to the free DoFs
    free_coords = get_free_dof_values(coord_fn)
    
    # Populate the corresponding row in our Trunk Net input tensor
    for i in 1:N_dofs
      x_raw[d,i] = Float32(free_coords[i])
    end
  end
  
  return x_raw
end

println("===========================================================")
println(" TEST 1: 1D Mesh with Full Dirichlet Boundaries")
println("===========================================================")
# Domain [0, 1] with 4 elements -> Nodes: 0.0, 0.25, 0.5, 0.75, 1.0
model_1d = CartesianDiscreteModel((0.0, 1.0), (4,))
reffe_1d = ReferenceFE(lagrangian, Float64, 1)

# Apply Dirichlet BC on the boundaries (x=0 and x=1 will be fixed)
V_1d = TestFESpace(model_1d, reffe_1d, dirichlet_tags="boundary")

println("Domain Setup     : 1D [0.0, 1.0] with partition (4,)")
println("Total Grid Nodes : 5")
println("Free DoFs Count  : 3")
println("EXPECTED DOFS (Ground Truth) : [0.25, 0.5, 0.75]")

println("\n--- Method 1 (Topological) ---")
coords_m1_1d = get_coords_method1(V_1d)
println("Raw Shape : ", size(coords_m1_1d))
println("Raw Values: ", coords_m1_1d)

# Manually stripping the boundary nodes
coords_m1_1d_stripped = coords_m1_1d[2:end-1]
println("Manually Stripped Shape : ", size(coords_m1_1d_stripped))
println("Manually Stripped Values: ", coords_m1_1d_stripped)

println("\n--- Method 2 (Algebraic) ---")
coords_m2_1d = get_coords_method2(V_1d)
println("Shape : ", size(coords_m2_1d))
println("Values: ", coords_m2_1d)

println("\n>>> ANALYSIS 1:")
println("Method 1 retrieves the full Cartesian grid. To match the algebraic free DoFs,")
println("manual slicing is required. Method 2 naturally retrieves only the free DoFs.")


println("\n\n===========================================================")
println(" TEST 2: 2D Mesh (No Boundaries) with OrderedFESpace")
println("===========================================================")
# Domain [0, 1] x [0, 1] with 2x2 elements -> 9 nodes total
model_2d_free = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (2, 2))
reffe_2d = ReferenceFE(lagrangian, Float64, 1)

# Pure free space, no Dirichlet conditions
V_2d_free_std = TestFESpace(model_2d_free, reffe_2d)
V_2d_free = OrderedFESpace(V_2d_free_std)

println("Domain Setup     : 2D [0.0, 1.0] x [0.0, 1.0] with partition (2, 2)")
println("Total Grid Nodes : 9")
println("Free DoFs Count  : 9")
println("EXPECTED DOFS (Ground Truth) : All 9 grid points from (0.0, 0.0) to (1.0, 1.0)")

println("\n--- Method 1 (Topological) ---")
coords_m1_2dfree = get_coords_method1(V_2d_free)
println("Raw Shape : ", size(coords_m1_2dfree))
println("Flattened via `vec()` (Column-Major):")
println(vec(coords_m1_2dfree))

println("\n--- Method 2 (Algebraic) ---")
coords_m2_2dfree = get_coords_method2(V_2d_free)
println("Shape : ", size(coords_m2_2dfree))
println("Columns representing DoFs:")
for i in 1:size(coords_m2_2dfree, 2)
    print("(", coords_m2_2dfree[1, i], ", ", coords_m2_2dfree[2, i], ") ")
end
println()

println("\n>>> ANALYSIS 2:")
println("Both methods return 9 coordinates matching Ground Truth. However, flattening Method 1 via `vec()`")
println("follows standard column-major traversal. The `OrderedFESpace` in Gridap rearranges the internal")
println("DoF numbering for bandwidth optimization. Method 2 implicitly follows this new ordering,")
println("ensuring that Coordinate 'j' strictly corresponds to Solution 'j'.")


println("\n\n===========================================================")
println(" TEST 3: 2D Mesh with Partial Dirichlet Boundaries")
println("===========================================================")
# 3x3 elements on [0,1]x[0,1] -> 16 nodes total (step = 0.3333...)
model_2d_partial = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (3, 3))

# Applying Dirichlet only on one specific boundary tag (tag 7: interior of left wall x=0.0)
V_2d_partial = TestFESpace(model_2d_partial, reffe_2d, dirichlet_tags=[7])

println("Domain Setup     : 2D [0.0, 1.0] x [0.0, 1.0] with partition (3, 3)")
println("Total Grid Nodes : 16")
println("Free DoFs Count  : 14 (Gridap tag 7 fixes only the 2 interior nodes of the left edge, leaving corners free)")
println("EXPECTED DOFS (Ground Truth) : All grid points EXCEPT (0.0, 0.333) and (0.0, 0.667)")

println("\n--- Method 1 (Topological) ---")
coords_m1_partial = get_coords_method1(V_2d_partial)
println("Raw Shape : ", size(coords_m1_partial))
println("Raw Values (Grid Matrix):")
display(coords_m1_partial)

println("\n--- Method 2 (Algebraic) ---")
coords_m2_partial = get_coords_method2(V_2d_partial)
println("Shape : ", size(coords_m2_partial))
println("Extracted Free DoFs Coordinates:")
for i in 1:size(coords_m2_partial, 2)
    print("(", round(coords_m2_partial[1, i], digits=3), ", ", round(coords_m2_partial[2, i], digits=3), ") ")
end
println()

println("\n>>> ANALYSIS 3:")
println("Gridap assigns distinct tags to corners and edges. By fixing tag 7, corners (0.0, 0.0) and (0.0, 1.0)")
println("remain free. Guessing the exact array indices to manually slice Method 1's Cartesian output becomes")
println("extremely error-prone. Method 2 relies purely on the algebraic FESpace logic, bypassing topological complexities.")


println("\n\n===========================================================")
println(" TEST 4: Unstructured Mesh (Simplex / Triangles)")
println("===========================================================")
# Convert a Cartesian model into an unstructured model of triangles
model_unstructured = simplexify(CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (2, 2)))
V_unstructured = TestFESpace(model_unstructured, reffe_2d)

println("Domain Setup     : 2D [0.0, 1.0] x [0.0, 1.0] with partition (2, 2) converted to simplices (triangles)")
println("Total Grid Nodes : 9 (Unstructured Triangular Mesh)")
println("Free DoFs Count  : 9")
println("EXPECTED DOFS (Ground Truth) : All 9 grid points from (0.0, 0.0) to (1.0, 1.0)")

println("\n--- Method 2 (Algebraic) ---")
coords_m2_uns = get_coords_method2(V_unstructured)
println("Shape : ", size(coords_m2_uns))
println("Extracted Free DoFs Coordinates:")
for i in 1:size(coords_m2_uns, 2)
    print("(", round(coords_m2_uns[1, i], digits=1), ", ", round(coords_m2_uns[2, i], digits=1), ") ")
end
println()