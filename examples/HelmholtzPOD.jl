module HelmholtzPOD

include("ExamplesInterface.jl")

using GridapGmsh
using Gridap.Fields
using Gridap.Geometry

const λ = 1.0          # Wavelength (arbitrary unit)
const L = 4.0          # Width of the area
const H = 6.0          # Height of the area
const xc = [0 -1.0]    # Center of the cylinder
const r = 1.0          # Radius of the cylinder
const d_pml = 0.8      # Thickness of the PML
const k = 2*π/λ        # Wave number
ϵ₁(μ) = μ[1]           # Relative electric permittivity for cylinder

const Rpml = 1e-12     # Tolerance for PML reflection
const σ = -3/4*log(Rpml)/d_pml # σ_0
const LH = (L,H)       # Size of the PML inner boundary (a rectangular center at (0,0))

function s_PML(x,σ,k,LH,d_pml)
  u = abs.(Tuple(x)).-LH./2  # get the depth into PML
  return @. ifelse(u > 0,  1+(1im*σ/k)*(u/d_pml)^2, $(1.0+0im))
end

function ds_PML(x,σ,k,LH,d_pml)
  u = abs.(Tuple(x)).-LH./2 # get the depth into PML
  ds = @. ifelse(u > 0, (2im*σ/k)*(1/d_pml)^2*u, $(0.0+0im))
  return ds.*sign.(Tuple(x))
end

struct Λ<:Function
  σ::Float64
  k::Float64
  LH::NTuple{2,Float64}
  d_pml::Float64
end

function (Λf::Λ)(x)
  s_x,s_y = s_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)
  return VectorValue(1/s_x,1/s_y)
end

Fields.∇(Λf::Λ) = x->TensorValue{2,2,ComplexF64}(-(Λf(x)[1])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)[1],0,0,-(Λf(x)[2])^2*ds_PML(x,Λf.σ,Λf.k,Λf.LH,Λf.d_pml)[2])

pdomain = (0.1,1.0)
pspace = ParamSpace(pdomain)

model = GmshDiscreteModel(datadir("models/emscatter.msh"))

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model;tags="Source")

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe,dirichlet_tags="DirichletEdges",vector_type=Vector{ComplexF64})
trial = ParamTrialFESpace(test)

degree = 2*order
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

labels = get_face_labeling(model)
dimension = num_cell_dims(model)
tags = get_face_tag(labels,dimension)
cylinder_tag = get_tag_from_name(labels,"Cylinder")

function ξ(μ)
  function ξtag(tag)
    if tag == cylinder_tag
      return 1/ϵ₁(μ)
    else
      return 1.0
    end
  end
  return ξtag
end

ξμ(μ) = parameterize(ξ,μ)

τ = CellField(tags,Ω)
Λf = Λ(σ,k,LH,d_pml)

a(μ,u,v,dΩ) = ∫(  (∇.*(Λf*v))⊙((ξμ(μ)∘τ)*(Λf.*∇(u))) - (k^2*(v*u))  )dΩ
b(μ,u,v,dΩ,dΓ) = a(μ,u,v,dΩ) - ∫(v)*dΓ

domains = FEDomains((Ω,Γ),(Ω,))
feop = LinearParamOperator(b,a,pspace,trial,test,domains)

fesolver = LUSolver()

energy(du,v) = ∫(∇(v)⊙∇(du))dΩ

tol = 1e-5
state_reduction = Reduction(tol,energy;nparams=80,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=40,nparams_jac=40)

dir = datadir("helmholtz_pod")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
run_test(dir,rbsolver,feop,tols)

end
