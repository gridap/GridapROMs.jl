using DrWatson
using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.CellData
using Gridap.MultiField
using GridapDistributed
using PartitionedArrays
using GridapPETSc
using GridapROMs
using GridapROMs.Utils
using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures

include("ExamplesInterface.jl")

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
ξ = 0.0
kᵣ = ξ*a₁/Lb   # = 0; joint rotational-spring term vanishes

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

# (Parametric) wave properties
pdomain = (20*h, 30*h)
pspace = ParamSpace(pdomain)

η₀ = 0.01
λ(μ)    = μ[1]
k(μ)    = 2π/λ(μ)
ω(μ)    = x -> √(g*k(μ)*tanh(k(μ)*H))
ηᵢₙ(μ)  = x -> η₀*exp(im*k(μ)*x[1])
vᵢₙ(μ)  = x -> (η₀*ω(μ)(x))*(cosh(k(μ)*x[2]) / sinh(k(μ)*H))*exp(im*k(μ)*x[1])
vzᵢₙ(μ) = x -> -im*ω(μ)(x)*η₀*exp(im*k(μ)*x[1])
αₕ(μ)   = x -> -im*ω(μ)(x)/g * (1-βₕ)/βₕ
μ₂ᵢₙ(μ) = x -> μ₁ᵢₙ(x)*k(μ)
μ₂ₒᵤₜ(μ)= x -> μ₁ₒᵤₜ(x)*k(μ)
ηd(μ)   = x -> μ₂ᵢₙ(μ)(x)*ηᵢₙ(μ)(x)
∇ₙϕd(μ) = x -> μ₁ᵢₙ(x)*vzᵢₙ(μ)(x)

ωμ(μ)       = parameterise(ω,μ)
vμᵢₙ(μ)     = parameterise(vᵢₙ,μ)
αμₕ(μ)      = parameterise(αₕ,μ)
μμ₂ᵢₙ(μ)   = parameterise(μ₂ᵢₙ,μ)
μμ₂ₒᵤₜ(μ)  = parameterise(μ₂ₒᵤₜ,μ)
ηdμ(μ)      = parameterise(ηd,μ)
∇ₙϕdμ(μ)   = parameterise(∇ₙϕd,μ)

# Non-uniform mesh map (applied node-by-node, works identically per-rank)
domain = (x₀, LΩ, 0.0, H)
partition_size = (nx_total, ny)
function f_y(x)
  x == H && return H
  i = x / (H/ny)
  return H - H/(2.5^i)
end
map_Ω(x) = VectorValue(x[1], f_y(x[2]))

# Surface-region classifiers: each checks the centroid of an element's coordinates.
# All of them guard x[2] ≈ H so they only fire on the physical top boundary.
function is_beam1(xs)
  x = sum(xs) / length(xs)
  (xb₀ <= x[1] <= xbⱼ) & (x[2] ≈ H)
end
function is_beam2(xs)
  x = sum(xs) / length(xs)
  (xbⱼ <= x[1] <= xb₁) & (x[2] ≈ H)
end
function is_damping1(xs)
  x = sum(xs) / length(xs)
  (x₀  <= x[1] <= xdᵢₙ) & (x[2] ≈ H)
end
function is_damping2(xs)
  x = sum(xs) / length(xs)
  (xdₒᵤₜ <= x[1]) & (x[2] ≈ H)
end

function main(distribute, parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  # A MUMPS direct solve is a safe default for this complex, multi-field system.
  # Replace with an iterative solver if the problem is too large.
  options = "-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps"
  GridapPETSc.with(args=split(options)) do

    # -----------------------------------------------------------------------
    # Distributed model + tag assignment
    # -----------------------------------------------------------------------
    𝒯_Ω = CartesianDiscreteModel(ranks, parts, domain, partition_size, map=map_Ω)

    # For a distributed model, get_cell_coordinates(Γ) is not implemented on
    # distributed triangulations.  Instead we work entirely within local_views:
    # each rank modifies its own FaceLabeling in place, then creates triangulations
    # from those tags via the standard Boundary/Interior constructors.
    #
    # Entity numbering for a 2D CartesianDiscreteModel (max = 9):
    #   10 → beam1 (1-faces on x ∈ [xb₀,xbⱼ], y ≈ H)
    #   11 → beam2 (1-faces on x ∈ [xbⱼ,xb₁], y ≈ H)
    #   12 → damping1 (1-faces on x ∈ [x₀,xdᵢₙ], y ≈ H)
    #   13 → damping2 (1-faces on x ∈ [xdₒᵤₜ,LΩ], y ≈ H)
    # After reclassification, entity 6 retains only the pure free-surface edges.
    map(local_views(𝒯_Ω)) do local_model
      labels = get_face_labeling(local_model)

      # Standard domain tags (same entity numbers as the serial version)
      add_tag_from_tags!(labels, "surface", [3, 4, 6])
      add_tag_from_tags!(labels, "bottom",  [1, 2, 5])
      add_tag_from_tags!(labels, "inlet",   [7])
      add_tag_from_tags!(labels, "outlet",  [8])
      add_tag_from_tags!(labels, "water",   [9])

      # Reclassify surface 1-faces (edges).
      # labels.d_to_dface_to_entity is 1-indexed in Julia:
      #   [1] = 0-faces (vertices), [2] = 1-faces (edges), [3] = 2-faces (cells)
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

      # Composite tags.  After reclassification entity 6 = pure free surface.
      # eta_surface   = beam1 ∪ beam2                  (η dofs live here)
      # kappa_surface = free surface ∪ damping zones   (κ dofs live here)
      add_tag_from_tags!(labels, "eta_surface",   ["beam1", "beam2"])
      add_tag_from_tags!(labels, "kappa_surface", ["surface", "damping1", "damping2"])
    end

    # -----------------------------------------------------------------------
    # Triangulations
    # -----------------------------------------------------------------------
    Ω   = Interior(𝒯_Ω)
    Γin = Boundary(𝒯_Ω, tags="inlet")
    Γb1 = Boundary(𝒯_Ω, tags="beam1")
    Γb2 = Boundary(𝒯_Ω, tags="beam2")
    Γd1 = Boundary(𝒯_Ω, tags="damping1")
    Γd2 = Boundary(𝒯_Ω, tags="damping2")
    Γfs = Boundary(𝒯_Ω, tags="surface")         # pure free surface (entity 6 remainder)
    Γη  = Boundary(𝒯_Ω, tags="eta_surface")     # beam1 ∪ beam2
    Γκ  = Boundary(𝒯_Ω, tags="kappa_surface")   # free surface ∪ damping zones

    # Interior skeletons of each beam for the DG penalty terms
    Λb1 = Skeleton(Γb1)
    Λb2 = Skeleton(Γb2)

    # -----------------------------------------------------------------------
    # Measures and normals
    # -----------------------------------------------------------------------
    degree = 2*order
    dΩ    = Measure(Ω,   degree)
    dΓb1  = Measure(Γb1, degree)
    dΓb2  = Measure(Γb2, degree)
    dΓd1  = Measure(Γd1, degree)
    dΓd2  = Measure(Γd2, degree)
    dΓfs  = Measure(Γfs, degree)
    dΓin  = Measure(Γin, degree)
    dΓκ   = Measure(Γκ,  degree)
    dΛb1  = Measure(Λb1, degree)
    dΛb2  = Measure(Λb2, degree)

    nΛb1 = get_normal_vector(Λb1)
    nΛb2 = get_normal_vector(Λb2)

    # -----------------------------------------------------------------------
    # FE spaces
    # -----------------------------------------------------------------------
    reffe = ReferenceFE(lagrangian, Float64, order)
    V_Ω  = TestFESpace(Ω,  reffe, conformity=:H1, vector_type=Vector{ComplexF64})
    V_Γκ = TestFESpace(Γκ, reffe, conformity=:H1, vector_type=Vector{ComplexF64})
    V_Γη = TestFESpace(Γη, reffe, conformity=:H1, vector_type=Vector{ComplexF64})
    U_Ω  = ParamTrialFESpace(V_Ω)
    U_Γκ = ParamTrialFESpace(V_Γκ)
    U_Γη = ParamTrialFESpace(V_Γη)
    X = MultiFieldFESpace([U_Ω, U_Γκ, U_Γη]; style=BlockMultiFieldStyle())
    Y = MultiFieldFESpace([V_Ω, V_Γκ, V_Γη]; style=BlockMultiFieldStyle())

    # -----------------------------------------------------------------------
    # Weak form
    # Note: the joint term ∫(kᵣ*jump(∇v⋅nΛj)*jump(∇η⋅nΛj))dΛj from the serial
    # version is omitted here because kᵣ = ξ*a₁/Lb = 0 for ξ = 0.
    # -----------------------------------------------------------------------
    ∇ₙ(ϕ) = ∇(ϕ)⋅VectorValue(0.0, 1.0)

    a(μ, (ϕ,κ,η), (w,u,v), dΩ, dΓfs, dΓd1, dΓd2, dΓb1, dΓb2, dΛb1, dΛb2) =
      ∫(  ∇(w)⋅∇(ϕ)  )dΩ +
      ∫(  βₕ*(u + αμₕ(μ)*w)*(g*κ - im*ωμ(μ)*ϕ) + im*ωμ(μ)*w*κ  )dΓfs +
      ∫(  βₕ*(u + αμₕ(μ)*w)*(g*κ - im*ωμ(μ)*ϕ) + im*ωμ(μ)*w*κ
          - μμ₂ᵢₙ(μ)*κ*w + μ₁ᵢₙ*∇ₙ(ϕ)*(u + αμₕ(μ)*w)  )dΓd1 +
      ∫(  βₕ*(u + αμₕ(μ)*w)*(g*κ - im*ωμ(μ)*ϕ) + im*ωμ(μ)*w*κ
          - μμ₂ₒᵤₜ(μ)*κ*w + μ₁ₒᵤₜ*∇ₙ(ϕ)*(u + αμₕ(μ)*w)  )dΓd2 +
      ∫(  v*((-ωμ(μ)^2*d₀ + g)*η - im*ωμ(μ)*ϕ) + a₁*Δ(v)*Δ(η) + im*ωμ(μ)*w*η  )dΓb1 +
      ∫(  v*((-ωμ(μ)^2*d₀ + g)*η - im*ωμ(μ)*ϕ) + a₂*Δ(v)*Δ(η) + im*ωμ(μ)*w*η  )dΓb2 +
      ∫(  a₁*( - jump(∇(v)⋅nΛb1)*mean(Δ(η)) - mean(Δ(v))*jump(∇(η)⋅nΛb1)
               + γ*(jump(∇(v)⋅nΛb1)*jump(∇(η)⋅nΛb1)) )  )dΛb1 +
      ∫(  a₂*( - jump(∇(v)⋅nΛb2)*mean(Δ(η)) - mean(Δ(v))*jump(∇(η)⋅nΛb2)
               + γ*(jump(∇(v)⋅nΛb2)*jump(∇(η)⋅nΛb2)) )  )dΛb2

    l(μ, (w,u,v), dΓin, dΓd1) =
      ∫( w*vμᵢₙ(μ) )dΓin -
      ∫( ηdμ(μ)*w - ∇ₙϕdμ(μ)*(u + αμₕ(μ)*w) )dΓd1

    res(μ, ϕ, w, dΩ, dΓin, dΓfs, dΓd1, dΓd2, dΓb1, dΓb2, dΛb1, dΛb2) =
      a(μ, ϕ, w, dΩ, dΓfs, dΓd1, dΓd2, dΓb1, dΓb2, dΛb1, dΛb2) -
      l(μ, w, dΓin, dΓd1)

    trian_res = (Ω, Γin, Γfs, Γd1, Γd2, Γb1, Γb2, Λb1, Λb2)
    trian_a   = (Ω, Γfs, Γd1, Γd2, Γb1, Γb2, Λb1, Λb2)
    domains = FEDomains(trian_res, trian_a)

    op = LinearParamOperator(res, a, pspace, X, Y, domains)

    # -----------------------------------------------------------------------
    # RB solver
    # -----------------------------------------------------------------------
    solver = PETScLinearSolver()
    tol = 1e-4
    nparams = 50
    nparams_res = 50
    nparams_jac = 50
    _ω = √(g*(2π/(20*h))*tanh((2π/(20*h))*H))

    energy((ϕ,κ,η), (w,u,v)) =
      ∫(∇(ϕ)⋅∇(w))dΩ +
      ∫(βₕ*g*κ*u)dΓκ +
      ∫((_ω^2*d₀ + g)*η*v)dΓb1 + ∫((_ω^2*d₀ + g)*η*v)dΓb2 +
      ∫(a₁*Δ(v)*Δ(η))dΓb1 + ∫(a₂*Δ(v)*Δ(η))dΓb2 +
      ∫(a₁*γ*(jump(∇(v)⋅nΛb1)*jump(∇(η)⋅nΛb1)))dΛb1 +
      ∫(a₂*γ*(jump(∇(v)⋅nΛb2)*jump(∇(η)⋅nΛb2)))dΛb2

    state_reduction = PODReduction(tol, energy; nparams, sketch=:sprn)
    rbsolver = RBSolver(solver, state_reduction; nparams_res, nparams_jac)

    dir = datadir("khaba_freq_distr")
    create_dir(dir)

    @btime solution_snapshots($rbsolver, $op; nparams=1)
  end
end

with_mpi() do distribute
  main(distribute, (2, 2))
end
