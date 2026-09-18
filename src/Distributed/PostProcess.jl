for T in (:DEIMHyperReduction,:SOPTHyperReduction)
  @eval begin
    function RBSteady.check_interpolation(res::DistributedSnapshots,a::HRVecProjection{<:$T},_fecache)
      msg = "fecache mismatch at interpolation points"
      fecache = sreduce(map(get_all_data,local_views(_fecache)))
      dofs = get_interpolation_dofs(get_interpolation(a))
      data = map(local_views(res),local_views(dofs)) do rvals,rdofs
        d = zero(fecache)
        isempty(rdofs.global_rows) && return d
        g2l = global_to_local(rdofs.index_parts)
        b = flatten(rvals)
        @views for (gri,i) in zip(rdofs.global_rows,rdofs.global_cols)
          d[i,:] .= b[g2l[gri],:]
        end
        d
      end |> sreduce
      @check isapprox(fecache,data;rtol=1e-8) msg
      return true
    end

    function RBSteady.check_interpolation(jac::DistributedSnapshots,a::HRMatProjection{<:$T},_fecache)
      msg = "fecache mismatch at interpolation points"
      fecache = sreduce(map(get_all_data,local_views(_fecache)))
      dofs = get_interpolation_dofs(get_interpolation(a))
      data = map(local_views(jac),local_views(dofs)) do jvals,rdofs
        d = zero(fecache)
        rrows,rcols = rdofs
        isempty(rrows.global_rows) && return d
        rnz = sparsify_split_indices(rrows,rcols,get_dof_map(jvals))
        A = flatten(jvals)
        @views for (nzi,i) in zip(rnz,rrows.global_cols)
          d[i,:] .= A[nzi,:]
        end
        d
      end |> sreduce
      @check isapprox(fecache,data;rtol=1e-8) msg
      return true
    end
  end
end

for T in (:TransientDEIMHyperReduction,:TransientSOPTHyperReduction)
  @eval begin
    function RBSteady.check_interpolation(res::DistributedSnapshots,a::HRVecProjection{<:$T},_fecache)
      msg = "fecache mismatch at interpolation points"
      fecache = sreduce(map(get_all_data,local_views(_fecache)))
      interp = get_interpolation(a)
      rows = get_interpolation_dofs(interp)
      indices_time = get_indices_time(interp)
      style = get_domain_style(interp)
      bdata = if style isa KroneckerDomain
        RBTransient.get_at_kron_domain(res,rows,indices_time)
      else
        @check style isa SequentialDomain "Unsupported transient domain style"
        RBTransient.get_at_seq_domain(res,rows,indices_time)
      end
      @check isapprox(fecache,get_all_data(bdata);rtol=1e-8) msg
      return true
    end

    function RBSteady.check_interpolation(jac::DistributedSnapshots,a::HRMatProjection{<:$T},_fecache)
      msg = "fecache mismatch at interpolation points"
      fecache = sreduce(map(get_all_data,local_views(_fecache)))
      interp = get_interpolation(a)
      dofs = get_interpolation_dofs(interp)
      rows,cols = map(first,dofs),map(last,dofs)
      dof_maps = map(get_dof_map,local_views(jac))
      rows_nz = sparsify_split_indices(rows,cols,dof_maps)
      indices_time = get_indices_time(interp)
      style = get_domain_style(interp)
      Adata = if style isa KroneckerDomain
        RBTransient.get_at_kron_domain(jac,rows_nz,indices_time)
      else
        @check style isa SequentialDomain "Unsupported transient domain style"
        RBTransient.get_at_seq_domain(jac,rows_nz,indices_time)
      end
      @check isapprox(fecache,get_all_data(Adata);rtol=1e-8) msg
      return true
    end
  end
end

const DOFMAP_LABEL = "dofmap"
const HRPROJECTION_LABEL = "hrprojection"
const NORM_MATRIX_LABEL = "norm"
const BLOCK_LABEL = "block"
const TRIAN_LABEL = "trian"

function DrWatson.save(dir,s::DistributedSnapshots;label="")
  _psave(dir,SNAPSHOTS_LABEL,s.snaps;label)
end

function DrWatson.save(dir,s::DistributedBlockSnapshots;label="")
  for i in eachindex(blocks(s))
    save(dir,blocks(s)[i];label=_plabel(label,BLOCK_LABEL*"$i"))
  end
end

function RBSteady.load_snapshots(dir,ranks::AbstractArray;label="")
  if _haspart(dir,SNAPSHOTS_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"1"))
    nblocks = 0
    while _haspart(dir,SNAPSHOTS_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"$(nblocks+1)"))
      nblocks += 1
    end
    array = map(1:nblocks) do i
      _pload(dir,SNAPSHOTS_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"$i"))
    end
    param_data = mortar(map(get_param_data,array))
    BlockSnapshots(array,param_data)
  else
    snaps = _pload(dir,SNAPSHOTS_LABEL,ranks;label)
    DistributedSnapshots(snaps)
  end
end

function DrWatson.save(dir,a::DistributedProjection;label="")
  _psave(dir,PROJECTION_LABEL,a.basis;label)
  _psave(dir,DOFMAP_LABEL,a.dof_map;label)
end

function DrWatson.save(dir,a::DistributedNormedProjection;label="")
  save(dir,a.projection;label)
  _psave(dir,NORM_MATRIX_LABEL,a.norm_matrix;label)
end

function DrWatson.save(dir,a::DistributedKroneckerProjection;label="")
  save(dir,a.projection_space;label=_get_label(label,"space"))
  RBSteady.save(dir,a.projection_time;label=_get_label(label,"time"))
end

for T in (:DistributedProjection,:DistributedNormedProjection,:DistributedKroneckerProjection)
  @eval begin
    function DrWatson.save(dir,a::BlockProjection{<:$T};label="")
      for i in eachindex(a)
        save(dir,a[i];label=_plabel(label,BLOCK_LABEL*"$i"))
      end
    end
  end
end

function RBSteady.load_projection(dir,ranks::AbstractArray;label="")
  if _haspart(dir,PROJECTION_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"1"))
    # 1) block projection
    nblocks = 0
    while _haspart(dir,PROJECTION_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"$(nblocks+1)"))
      nblocks += 1
    end
    block_basis = map(1:nblocks) do i
      RBSteady.load_projection(dir,ranks;label=_plabel(label,BLOCK_LABEL*"$i"))
    end
    return BlockProjection(block_basis)
  elseif _haspart(dir,PROJECTION_LABEL,ranks;label=_get_label(label,"space"))
    # 2) kronecker projection: the space part is distributed, the time part is not
    projection_space = RBSteady.load_projection(dir,ranks;label=_get_label(label,"space"))
    projection_time = RBSteady.load_projection(dir;label=_get_label(label,"time"))
    return KroneckerProjection(projection_space,projection_time)
  else
    basis = _pload(dir,PROJECTION_LABEL,ranks;label)
    dof_map = _pload(dir,DOFMAP_LABEL,ranks;label)
    proj = Projection(basis,dof_map)
    if _haspart(dir,NORM_MATRIX_LABEL,ranks;label)
      # 3) normed projection
      X = _pload(dir,NORM_MATRIX_LABEL,ranks;label)
      return NormedProjection(proj,X)
    end
    # 4) generic projection
    return proj
  end
end

function DrWatson.save(dir,a::DistributedHRProjection;label="")
  map(local_views(a),linear_indices(local_views(a))) do a,p
    serialize(_part_filename(dir,HRPROJECTION_LABEL,label,p),a)
  end
end

function DrWatson.save(dir,a::BlockHRProjection{<:HyperReduction,<:Projection,<:DistributedHRProjection};label="")
  for i in eachindex(a)
    save(dir,a[i];label=_plabel(label,BLOCK_LABEL*"$i"))
  end
end

function RBSteady.load_reduced_subspace(dir,f::DistributedSingleFieldFESpace,ranks::AbstractArray;label="")
  basis = RBSteady.load_projection(dir,ranks;label)
  reduced_subspace(f,basis)
end

function RBSteady.load_reduced_subspace(dir,f::DistributedMultiFieldFESpace,ranks::AbstractArray;label="")
  basis = RBSteady.load_projection(dir,ranks;label)
  reduced_subspace(f,basis)
end

function DrWatson.save(dir,contrib::Contribution{V,T};label="") where {V,T<:DistributedTriangulation}
  for (i,v) in enumerate(get_contributions(contrib))
    save(dir,v;label=_plabel(label,"$(TRIAN_LABEL)$i"))
  end
end

function RBSteady.load_contribution(dir,trian::Tuple{Vararg{DistributedTriangulation}},ranks::AbstractArray;label="")
  vals = ntuple(length(trian)) do i
    _load_distributed_hr(dir,ranks;label=_plabel(label,"$(TRIAN_LABEL)$i"))
  end
  RBSteady._setup_contribution(vals,trian)
end

function RBSteady.load_operator(dir,feop::ParamOperator,ranks::AbstractArray;label="")
  test = RBSteady.load_reduced_subspace(dir,get_test(feop),ranks;label=_plabel(label,RBSteady.TEST_LABEL))
  trial = RBSteady.load_reduced_subspace(dir,get_trial(feop),ranks;label=_plabel(label,RBSteady.TRIAL_LABEL))
  trian_res = get_domains_res(feop)
  trian_jac = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res,ranks;label=_plabel(label,RBSteady.RHS_LABEL))
  red_lhs = load_contribution(dir,trian_jac,ranks;label=_plabel(label,RBSteady.LHS_LABEL))
  ReducedOperator(feop,trial,test,red_lhs,red_rhs)
end

# utils

_plabel(name,label...) = foldl(RBSteady._get_label,label;init=name)
_part_name(name,label,part) = _plabel(name,label,"part$part")
_part_filename(dir,name,label,part) = joinpath(dir,_part_name(name,label,part)*".jld")
_haspart(dir,name,ranks;label="") = isfile(_part_filename(dir,name,label,getany(ranks)))

function _psave(dir,name,x::Union{GenericPArray,PVector};label="")
  map(partition(x),partition(axes(x,1))) do xloc,ind
    serialize(_part_filename(dir,name,label,part_id(ind)),(xloc,ind))
  end
end

function _psave(dir,name,x::PSparseMatrix;label="")
  map(partition(x),partition(axes(x,1)),partition(axes(x,2))) do xloc,rind,cind
    serialize(_part_filename(dir,name,label,part_id(rind)),(xloc,rind,cind))
  end
end

function _pload(dir,name,ranks;label="")
  data,inds... = map(ranks) do p
    deserialize(_part_filename(dir,name,label,p))
  end |> tuple_of_arrays
  _pallocate(data,inds...)
end

function _pallocate(d,i...)
  @abstractmethod
end

function _pallocate(
  d::AbstractArray{<:AbstractVector},
  r::AbstractVector{<:AbstractVector}
  )

  PVector(d,r)
end

function _pallocate(
  d::AbstractArray{<:AbstractArray},
  r::AbstractVector{<:AbstractVector}
  )

  GenericPArray(d,r)
end

function _pallocate(
  d::AbstractArray{<:AbstractMatrix},
  r::AbstractVector{<:AbstractVector},
  c::AbstractVector{<:AbstractVector}
  )

  PSparseMatrix(d,r,c)
end

function _load_distributed_hrprojection(dir,ranks;label="")
  basis,style,interps = map(ranks) do p
    a = deserialize(_part_filename(dir,HRPROJECTION_LABEL,label,p))
    get_basis(a),get_style(a),get_interpolation(a)
  end |> tuple_of_arrays
  HRProjection(getany(basis),getany(style),DistributedInterpolation(interps))
end

function _load_distributed_hr(dir,ranks;label="")
  if _haspart(dir,HRPROJECTION_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"1"))
    nblocks = 0
    while _haspart(dir,HRPROJECTION_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"$(nblocks+1)"))
      nblocks += 1
    end
    array = map(1:nblocks) do i
      _load_distributed_hrprojection(dir,ranks;label=_plabel(label,BLOCK_LABEL*"$i"))
    end
    return BlockHRProjection(array)
  end
  _load_distributed_hrprojection(dir,ranks;label)
end