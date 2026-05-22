# DeepONet implementation for parametric PDE ROMs using Lux.jl.
#
# Key insight for PDE ROMs: the trunk network evaluates at fixed DOF
# coordinates that don't change between queries. So we precompute the
# trunk matrix T ∈ R^{N_dofs × p} once, and online prediction reduces to:
#
#   û(μ) = T · b(μ) + bias
#
# where b(μ) = branch(μ) ∈ R^p. This is an O(N·p) matrix-vector product,
# independent of the FEM assembly cost.

"""
    DeepONetLayer{B,T} <: Lux.AbstractLuxContainerLayer{(:branch,:trunk)}

Custom Lux layer implementing DeepONet for parametric PDE surrogate modelling.

The branch network encodes the parameter vector μ into a latent code b(μ).
The trunk network encodes spatial coordinates x into basis functions t(x).
The output at DOF location xᵢ is: û_i = Σₖ bₖ(μ) · tₖ(xᵢ) + bias_i.
"""
struct DeepONetLayer{B,T} <: Lux.AbstractLuxContainerLayer{(:branch,:trunk)}
  branch::B
  trunk::T
  latent_dim::Int
  n_dofs::Int
end

"""
    build_deeponet(;
      param_dim, n_dofs, spatial_dim,
      latent_dim=32, branch_width=64, trunk_width=64,
      n_branch_layers=2, n_trunk_layers=2,
      activation=Lux.gelu
    ) -> DeepONetLayer

Construct a DeepONet for mapping parameter vectors to FEM DOF vectors.

- `param_dim`:   dimension of the parameter space (branch input)
- `n_dofs`:      number of free DOFs in the FEM discretization (output dim)
- `spatial_dim`: spatial dimension D of the mesh (trunk input)
- `latent_dim`:  dimension p of the shared latent space
"""
function build_deeponet(;
  param_dim::Int,
  n_dofs::Int,
  spatial_dim::Int,
  latent_dim::Int=32,
  branch_width::Int=64,
  trunk_width::Int=64,
  n_branch_layers::Int=2,
  n_trunk_layers::Int=2,
  activation=Lux.gelu
)
  # Branch: μ ∈ R^d → b(μ) ∈ R^p
  branch_layers = Any[Dense(param_dim,branch_width,activation)]
  for _ in 2:n_branch_layers
    push!(branch_layers,Dense(branch_width,branch_width,activation))
  end
  push!(branch_layers,Dense(branch_width,latent_dim))
  branch = Chain(branch_layers...)

  # Trunk: x ∈ R^D → t(x) ∈ R^p
  trunk_layers = Any[Dense(spatial_dim,trunk_width,activation)]
  for _ in 2:n_trunk_layers
    push!(trunk_layers,Dense(trunk_width,trunk_width,activation))
  end
  push!(trunk_layers,Dense(trunk_width,latent_dim))
  trunk = Chain(trunk_layers...)

  return DeepONetLayer(branch,trunk,latent_dim,n_dofs)
end

"""
    precompute_trunk_matrix(model, coord_matrix, ps, st) -> Matrix{Float32}

Evaluate the trunk network at all DOF coordinates once.
Returns T ∈ R^{N_dofs × latent_dim} so that prediction is T * b(μ) + bias.
"""
function precompute_trunk_matrix(model::DeepONetLayer,coord_matrix::AbstractMatrix,ps,st)
  # coord_matrix: D × N_nodes (Float32)
  trunk_out,_ = Lux.apply(model.trunk,coord_matrix,ps.trunk,st.trunk)
  # trunk_out: latent_dim × N_nodes
  return Matrix(transpose(trunk_out))  # N_nodes × latent_dim
end

# Forward pass: branch encodes μ, then multiply with precomputed trunk matrix.
# Input:  x = (mu, trunk_matrix) packed as a tuple
# Output: û ∈ R^{N_dofs × batch}
function (l::DeepONetLayer)(x::Tuple,ps,st)
  mu,trunk_matrix = x
  # mu: d_param × batch
  b,new_st_branch = Lux.apply(l.branch,mu,ps.branch,st.branch)
  # b: latent_dim × batch

  # û = T * b + bias  →  (N_dofs × batch)
  u_hat = trunk_matrix * b

  new_st = (branch=new_st_branch,trunk=st.trunk)
  return u_hat,new_st
end
