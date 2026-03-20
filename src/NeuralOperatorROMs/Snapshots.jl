# Snapshot collection: solve a parametric PDE at sampled parameters,
# extract free DOF vectors as training data for neural operators.
#
# This mirrors the role of GridapROMs.jl's `solution_snapshots` but in a
# minimal, non-intrusive form: we only need the DOF vectors and coordinates,
# not the residual/Jacobian snapshots that Galerkin projection requires.

"""
    SnapshotData

Training dataset for a neural operator ROM. Stores parameter vectors,
the corresponding FEM solution DOF vectors, and mesh coordinate data.

Fields:
- `parameters`:  d_param × M matrix, each column is one parameter sample μ
- `solutions`:   N_dofs × M matrix, each column is get_free_dof_values(uh(μ))
- `coordinates`: D × N_free matrix, spatial coordinates for each free DOF
"""
struct SnapshotData
  parameters::Matrix{Float64}
  solutions::Matrix{Float64}
  coordinates::Matrix{Float64}
end

"""
    extract_coordinates(trial::FESpace) -> Matrix{Float64}

Extract spatial coordinates for each free DOF as a D × N_free matrix.
These serve as input to the trunk network in DeepONet.

For Lagrangian elements, each DOF corresponds to a mesh node. We build
a mapping from free DOF index to node coordinate using the cell-level
DOF and node connectivity. This ensures the coordinate matrix has exactly
N_free columns (matching the solution DOF vector length), excluding
Dirichlet boundary DOFs.
"""
function extract_coordinates(trial::FESpace)
  trian = get_triangulation(trial)
  all_coords = get_node_coordinates(trian)
  D = length(first(all_coords))
  N_free = num_free_dofs(trial)

  # Build free-DOF → node-coordinate mapping via cell connectivity
  cell_dof_ids = get_cell_dof_ids(trial)
  cell_node_ids = Gridap.Geometry.get_cell_node_ids(trian)

  coord_matrix = zeros(Float64,D,N_free)
  filled = falses(N_free)

  for cell in 1:length(cell_dof_ids)
    dofs = cell_dof_ids[cell]
    nodes = cell_node_ids[cell]
    for (local_i,dof_id) in enumerate(dofs)
      if dof_id > 0 && !filled[dof_id]
        c = all_coords[nodes[local_i]]
        for d in 1:D
          coord_matrix[d,dof_id] = c[d]
        end
        filled[dof_id] = true
      end
    end
  end

  return coord_matrix
end

"""
    collect_snapshots(
      solver_fn,
      param_samples::Vector{<:AbstractVector};
      trial::FESpace
    ) -> SnapshotData

Solve a parametric PDE for each parameter sample and collect DOF vectors.

Arguments:
- `solver_fn`:      a function μ -> uh::FEFunction that solves the PDE at parameter μ
- `param_samples`:  vector of parameter vectors [{μ₁}, {μ₂}, ...]
- `trial`:          the trial FESpace (needed for coordinate extraction)

The `solver_fn` encapsulates the entire Gridap problem setup: mesh, spaces,
weak form, and linear solve. This keeps the snapshot collector non-intrusive —
it treats the FEM solver as a black box, which is exactly the philosophy
behind neural operator ROMs.
"""
function collect_snapshots(
  solver_fn,
  param_samples::Vector{<:AbstractVector};
  trial::FESpace
)
  M = length(param_samples)
  d_param = length(first(param_samples))

  # Solve for first sample to determine DOF dimension
  uh_first = solver_fn(param_samples[1])
  dofs_first = get_free_dof_values(uh_first)
  N_dofs = length(dofs_first)

  # Allocate storage
  parameters = zeros(Float64,d_param,M)
  solutions = zeros(Float64,N_dofs,M)

  # Store first solution
  parameters[:,1] .= param_samples[1]
  solutions[:,1] .= dofs_first

  # Solve remaining
  for i in 2:M
    μ = param_samples[i]
    uh = solver_fn(μ)
    parameters[:,i] .= μ
    solutions[:,i] .= get_free_dof_values(uh)
  end

  coordinates = extract_coordinates(trial)
  return SnapshotData(parameters,solutions,coordinates)
end

"""
    sample_parameters(bounds::Vector{Tuple{Float64,Float64}}, n::Int) -> Vector{Vector{Float64}}

Latin hypercube sampling over a box-constrained parameter space.
Each element of `bounds` is (lower, upper) for one parameter dimension.
"""
function sample_parameters(bounds::Vector{Tuple{Float64,Float64}},n::Int)
  D = length(bounds)
  samples = Vector{Vector{Float64}}(undef,n)
  perms = [randperm(n) for _ in 1:D]
  for i in 1:n
    μ = zeros(D)
    for d in 1:D
      lo,hi = bounds[d]
      μ[d] = lo + (perms[d][i] - rand()) / n * (hi - lo)
    end
    samples[i] = μ
  end
  return samples
end
