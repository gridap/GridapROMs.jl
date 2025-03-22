function ODEs.time_derivative(r::RBSpace)
  fet = time_derivative(get_fe_space(r))
  rb = get_reduced_subspace(r)
  reduced_subspace(fet,rb)
end

function RBSteady.project(r1::RBSpace,x::Projection,r2::RBSpace,combine::Function)
  galerkin_projection(get_reduced_subspace(r1),x,get_reduced_subspace(r2),combine)
end

const TransientSingleFieldEvalRBSpace{S} = EvalRBSpace{SingleFieldRBSpace{S},<:TransientRealization}

function Algebra.allocate_in_domain(r::TransientSingleFieldEvalRBSpace,x::V) where V<:AbstractParamVector
  x̂ = allocate_vector(eltype(V),num_reduced_dofs(r))
  np = num_params(r.realization)
  return global_parameterize(x̂,np)
end

function Algebra.allocate_in_range(r::TransientSingleFieldEvalRBSpace,x̂::V) where V<:AbstractParamVector
  x = allocate_vector(eltype(V),num_fe_dofs(r))
  ntp = length(r.realization)
  return global_parameterize(x,ntp)
end

function RBSteady.project!(x̂::ConsecutiveParamVector,r::TransientSingleFieldEvalRBSpace,x::ConsecutiveParamVector)
  nt = num_times(r.realization)
  @check Int(param_length(x) / param_length(x̂)) == nt
  np = param_length(x̂)
  rsb = get_reduced_subspace(r)
  @inbounds for ip in eachindex(x̂)
    ipt = ip:np:np*nt
    xpt = vec(view(x.data,:,ipt))
    x̂p = x̂[ip]
    project!(x̂p,rsb,xpt)
  end
end

function RBSteady.inv_project!(x::AbstractParamVector,r::TransientSingleFieldEvalRBSpace,x̂::AbstractParamVector)
  nt = num_times(r.realization)
  @check Int(param_length(x) / param_length(x̂)) == nt
  np = param_length(x̂)
  rsb = get_reduced_subspace(r)
  @inbounds for ip in eachindex(x̂)
    ipt = ip:np:np*nt
    xpt = vec(view(x.data,:,ipt))
    x̂p = x̂[ip]
    inv_project!(xpt,rsb,x̂p)
  end
end

const TransientMultiFieldEvalRBSpace = EvalRBSpace{MultiFieldRBSpace,<:TransientRealization}

for f in (:(Algebra.allocate_in_domain),:(Algebra.allocate_in_range))
  @eval begin
    function $f(r::TransientMultiFieldEvalRBSpace,x::Union{BlockVector,BlockParamVector})
      mortar(map(i -> $f(r[i],x[Block(i)]),1:length(r)))
    end
  end
end

for f! in (:(RBSteady.project!),:(RBSteady.inv_project!))
  @eval begin
    function $f!(x::AbstractParamVector,r::TransientMultiFieldEvalRBSpace,x̂::AbstractParamVector)
      map(i -> $f!(x[Block(i)],r[i],x̂[Block(i)]),1:length(r))
    end
  end
end
