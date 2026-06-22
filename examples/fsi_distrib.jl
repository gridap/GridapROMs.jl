using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.CellData
using Gridap.MultiField
using GridapDistributed
using PartitionedArrays
using GridapPETSc

# Gridap upstream bug fix: testvalue(LinearCombinationField{V,F}) always creates
# a Matrix for values regardless of V. When V<:AbstractVector and testitem is called
# on an empty array (e.g. when ALL cells on a rank are void for a surface FE space
# in a MultiField with disjoint surface domains), this produces the wrong concrete
# type and triggers a TypeError. Override with the vector-correct version.
function Gridap.Arrays.testvalue(
  ::Type{Gridap.Fields.LinearCombinationField{V,F}}
) where {V<:AbstractVector,F}
  fields = Gridap.Arrays.testvalue(F)
  values = zeros(eltype(V), length(fields))
  Gridap.Fields.LinearCombinationField(values, fields, 1)
end

# Fixed parameters
Lb = 12.5
m = 8.36
EI₁ = 47100.0
EI₂ = 471.0
β = 0.2
H = 1.1

# Domain size
Ld = Lb
LΩ = 2Ld + 2Lb
x₀ = 0.0
xdᵢₙ = x₀ + Ld
xb₀ = xdᵢₙ + Lb/2
xbⱼ = xb₀ + β*Lb
xb₁ = xb₀ + Lb
xdₒᵤₜ = LΩ - Ld

# Physics
g = 9.81
ρ = 1025
d₀ = m/ρ
a₁ = EI₁/ρ
a₂ = EI₂/ρ

# Numerics constants
nx = 20
ny = 5
nx_total = Int(ceil(nx/β)*ceil(LΩ/Lb))
h = LΩ / nx_total
order = 4
γ = 1.0*order*(order-1)/h
βₕ = 0.5

# Damping
μ₀ = 2.5
μ₁ᵢₙ(x)  = μ₀*(1.0 - sin(π/2*(x[1])/Ld))
μ₁ₒᵤₜ(x) = μ₀*(1.0 - cos(π/2*(x[1]-xdₒᵤₜ)/Ld))

# Fixed wave parameter (midpoint of the parameter domain (20h,30h))
η₀ = 0.01
λ_val   = 25*h
k_val   = 2π/λ_val
ω_val   = √(g*k_val*tanh(k_val*H))
αₕ_val  = -im*ω_val/g * (1-βₕ)/βₕ   # scalar constant

μ₂ᵢₙ(x)  = μ₁ᵢₙ(x)*k_val
μ₂ₒᵤₜ(x) = μ₁ₒᵤₜ(x)*k_val

vᵢₙ(x)  = (η₀*ω_val)*(cosh(k_val*x[2])/sinh(k_val*H))*exp(im*k_val*x[1])
vzᵢₙ(x) = -im*ω_val*η₀*exp(im*k_val*x[1])
ηd(x)   = μ₂ᵢₙ(x)*η₀*exp(im*k_val*x[1])
∇ₙϕd(x) = μ₁ᵢₙ(x)*vzᵢₙ(x)

# Mesh mapping
domain = (x₀, LΩ, 0.0, H)
partition_size = (nx_total, ny)
function f_y(x)
  x == H && return H
  i = x/(H/ny)
  return H - H/(2.5^i)
end
map_Ω(x) = VectorValue(x[1], f_y(x[2]))

# Surface-region classifiers (centroid-based; x[2] ≈ H guards interior edges)
is_beam1(xs)    = (x = sum(xs)/length(xs); (xb₀  <= x[1] <= xbⱼ)  & (x[2] ≈ H))
is_beam2(xs)    = (x = sum(xs)/length(xs); (xbⱼ  <= x[1] <= xb₁)  & (x[2] ≈ H))
is_damping1(xs) = (x = sum(xs)/length(xs); (x₀   <= x[1] <= xdᵢₙ) & (x[2] ≈ H))
is_damping2(xs) = (x = sum(xs)/length(xs); (xdₒᵤₜ<= x[1])         & (x[2] ≈ H))

function main(distribute, parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  options = "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"
  GridapPETSc.with(args=split(options)) do

    # ------------------------------------------------------------------
    # Distributed model + tag assignment
    # ------------------------------------------------------------------
    𝒯_Ω = CartesianDiscreteModel(ranks, parts, domain, partition_size, map=map_Ω)

    # All tag logic lives inside local_views: get_cell_coordinates(Γ) is not
    # available on distributed triangulations, so we classify edges by their
    # local coordinates and add tags directly to each rank's FaceLabeling.
    # Entity IDs 10-13 are globally fixed (safe: CartesianDiscreteModel 2D max = 9).
    map(local_views(𝒯_Ω)) do local_model
      labels = get_face_labeling(local_model)

      add_tag_from_tags!(labels, "surface",  [3, 4, 6])
      add_tag_from_tags!(labels, "bottom",   [1, 2, 5])
      add_tag_from_tags!(labels, "inlet",    [7])
      add_tag_from_tags!(labels, "outlet",   [8])
      add_tag_from_tags!(labels, "water",    [9])

      # Reclassify surface 1-faces (edges).
      # d_to_dface_to_entity is 1-indexed: [1]=vertices, [2]=edges, [3]=cells.
      edge_to_entity = labels.d_to_dface_to_entity[2]
      edge_coords    = get_cell_coordinates(Grid(ReferenceFE{1}, local_model))
      for (i, xs) in enumerate(edge_coords)
        if     is_beam1(xs);    edge_to_entity[i] = 10
        elseif is_beam2(xs);    edge_to_entity[i] = 11
        elseif is_damping1(xs); edge_to_entity[i] = 12
        elseif is_damping2(xs); edge_to_entity[i] = 13
        end
      end

      add_tag!(labels, "beam1",    [10])
      add_tag!(labels, "beam2",    [11])
      add_tag!(labels, "damping1", [12])
      add_tag!(labels, "damping2", [13])

      # After reclassification entity 6 = pure free surface.
      # eta_surface   = beam1 ∪ beam2          (η dofs)
      # kappa_surface = free surface ∪ damping (κ dofs)
      add_tag_from_tags!(labels, "eta_surface",   ["beam1", "beam2"])
      add_tag_from_tags!(labels, "kappa_surface", ["surface", "damping1", "damping2"])
    end

    # ------------------------------------------------------------------
    # Triangulations
    # ------------------------------------------------------------------
    Ω   = Interior(𝒯_Ω)
    Γin = Boundary(𝒯_Ω, tags="inlet")
    Γb1 = Boundary(𝒯_Ω, tags="beam1")
    Γb2 = Boundary(𝒯_Ω, tags="beam2")
    Γd1 = Boundary(𝒯_Ω, tags="damping1")
    Γd2 = Boundary(𝒯_Ω, tags="damping2")
    Γfs = Boundary(𝒯_Ω, tags="surface")
    Γη  = Boundary(𝒯_Ω, tags="eta_surface")
    Γκ  = Boundary(𝒯_Ω, tags="kappa_surface")
    Λb1 = Skeleton(Γb1)
    Λb2 = Skeleton(Γb2)

    # ------------------------------------------------------------------
    # Measures and normals
    # ------------------------------------------------------------------
    degree = 2*order
    dΩ   = Measure(Ω,   degree)
    dΓb1 = Measure(Γb1, degree)
    dΓb2 = Measure(Γb2, degree)
    dΓd1 = Measure(Γd1, degree)
    dΓd2 = Measure(Γd2, degree)
    dΓfs = Measure(Γfs, degree)
    dΓin = Measure(Γin, degree)
    dΛb1 = Measure(Λb1, degree)
    dΛb2 = Measure(Λb2, degree)
    nΛb1 = get_normal_vector(Λb1)
    nΛb2 = get_normal_vector(Λb2)

    # ------------------------------------------------------------------
    # FE spaces  (no Dirichlet BCs: all boundaries are Neumann)
    # ------------------------------------------------------------------
    reffe = ReferenceFE(lagrangian, Float64, order)
    V_Ω  = TestFESpace(Ω,  reffe, conformity=:H1, vector_type=Vector{ComplexF64})
    V_Γκ = TestFESpace(Γκ, reffe, conformity=:H1, vector_type=Vector{ComplexF64})
    V_Γη = TestFESpace(Γη, reffe, conformity=:H1, vector_type=Vector{ComplexF64})
    U_Ω  = TrialFESpace(V_Ω)
    U_Γκ = TrialFESpace(V_Γκ)
    U_Γη = TrialFESpace(V_Γη)
    X = MultiFieldFESpace([U_Ω, U_Γκ, U_Γη])
    Y = MultiFieldFESpace([V_Ω, V_Γκ, V_Γη])

    # ------------------------------------------------------------------
    # Weak form  (joint term omitted: kᵣ = 0 for ξ = 0)
    # ------------------------------------------------------------------
    ∇ₙ(ϕ) = ∇(ϕ)⋅VectorValue(0.0, 1.0)

    a((ϕ,κ,η),(w,u,v)) =
      ∫( ∇(w)⋅∇(ϕ) )dΩ +
      ∫( βₕ*(u + αₕ_val*w)*(g*κ - im*ω_val*ϕ) + im*ω_val*w*κ )dΓfs +
      ∫( βₕ*(u + αₕ_val*w)*(g*κ - im*ω_val*ϕ) + im*ω_val*w*κ
         - μ₂ᵢₙ*κ*w + μ₁ᵢₙ*∇ₙ(ϕ)*(u + αₕ_val*w) )dΓd1 +
      ∫( βₕ*(u + αₕ_val*w)*(g*κ - im*ω_val*ϕ) + im*ω_val*w*κ
         - μ₂ₒᵤₜ*κ*w + μ₁ₒᵤₜ*∇ₙ(ϕ)*(u + αₕ_val*w) )dΓd2 +
      ∫( v*((-ω_val^2*d₀ + g)*η - im*ω_val*ϕ) + a₁*Δ(v)*Δ(η) + im*ω_val*w*η )dΓb1 +
      ∫( v*((-ω_val^2*d₀ + g)*η - im*ω_val*ϕ) + a₂*Δ(v)*Δ(η) + im*ω_val*w*η )dΓb2 +
      ∫( a₁*( - jump(∇(v)⋅nΛb1)*mean(Δ(η)) - mean(Δ(v))*jump(∇(η)⋅nΛb1)
               + γ*(jump(∇(v)⋅nΛb1)*jump(∇(η)⋅nΛb1)) ) )dΛb1 +
      ∫( a₂*( - jump(∇(v)⋅nΛb2)*mean(Δ(η)) - mean(Δ(v))*jump(∇(η)⋅nΛb2)
               + γ*(jump(∇(v)⋅nΛb2)*jump(∇(η)⋅nΛb2)) ) )dΛb2

    l((w,u,v)) =
      ∫( w*vᵢₙ )dΓin -
      ∫( ηd*w - ∇ₙϕd*(u + αₕ_val*w) )dΓd1

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    op = AffineFEOperator(a, l, X, Y)
    solver = PETScLinearSolver()
    ϕh, κh, ηh = solve(solver, op)

    writevtk(Ω,"output_path/results_ex1",
      cellfields=["ϕh"=>ϕh,"κh"=>κh,"ηh"=>ηh])
  end
end

with_mpi() do distribute
  main(distribute, (2, 2))
end
