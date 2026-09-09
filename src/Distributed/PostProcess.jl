for T in (:DEIMHyperReduction,:SOPTHyperReduction,:HighDimDEIMHyperReduction,:HighDimSOPTHyperReduction)
  for (A,B) in zip((:PVector,:PSparseMatrix),(:HRVecProjection,:HRMatProjection))
    @eval begin
      function RBSteady.check_interpolation(resjac::$A,a::$B{<:$T},fecache::AbstractArray{<:AbstractArray})
        map(local_views(resjac),local_views(a),local_views(fecache)) do resjac,a,fecache
          check_interpolation(resjac,a,fecache)
        end
      end
    end
  end
end

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
    array = DistributedSnapshots[]
    i = 1
    while _haspart(dir,SNAPSHOTS_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"$i"))
      snaps = _pload(dir,SNAPSHOTS_LABEL,ranks;label=_plabel(label,BLOCK_LABEL*"$i"))
      push!(array,DistributedSnapshots(snaps))
      i += 1
    end
    param_data = mortar(map(get_param_data,array))
    DistributedBlockSnapshots(array,param_data)
  else
    DistributedSnapshots(_pload(dir,SNAPSHOTS_LABEL,ranks;label))
  end
end

function DrWatson.save(dir,a::DistributedPODProjection;label="")
  _psave(dir,PROJECTION_LABEL,a.basis;label)
end

function DrWatson.save(dir,a::DistributedNormedProjection;label="")
  save(dir,a.projection;label)
  _psave(dir,NORM_MATRIX_LABEL,a.norm_matrix;label)
end

function DrWatson.save(dir,a::BlockProjection{<:DistributedProjection};label="")
  for i in eachindex(a)
    save(dir,a[i];label=_plabel(label,BLOCK_LABEL*"$i"))
  end
end

function RBSteady.load_projection(dir,ranks::AbstractArray;label="")
  basis = _pload(dir,PROJECTION_LABEL,ranks;label)
  proj = DistributedPODProjection(basis)
  if _haspart(dir,NORM_MATRIX_LABEL,ranks;label)
    X = _pload(dir,NORM_MATRIX_LABEL,ranks;label)
    return DistributedNormedProjection(proj,X)
  end
  return proj
end

function DrWatson.save(dir,a::DistributedHRProjection;label="")
  map(local_views(a),linear_indices(local_views(a))) do a,p
    serialize(_part_filename(dir,HRPROJECTION_LABEL,label,p),a)
  end
end

function DrWatson.save(dir,a::BlockHRProjection{<:DistributedHRProjection};label="")
  for i in eachindex(a)
    save(dir,a[i];label=_plabel(label,BLOCK_LABEL*"$i"))
  end
end

function RBSteady.load_reduced_subspace(dir,f::DistributedSingleFieldFESpace,ranks::AbstractArray;label="")
  basis = RBSteady.load_projection(dir,ranks;label)
  reduced_subspace(f,basis)
end

function RBSteady.load_reduced_subspace(dir,f::DistributedMultiFieldFESpace,ranks::AbstractArray;label="")
  basis = _load_distributed_block_projection(dir,ranks,num_fields(f);label)
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
  d::AbstractArray{<:AbstractMatrix},
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

function _load_distributed_block_projection(dir,ranks,nfields;label="")
  block_basis = map(1:nfields) do i
    RBSteady.load_projection(dir,ranks;label=_plabel(label,BLOCK_LABEL*"$i"))
  end
  BlockProjection(block_basis)
end

function _load_distributed_hrprojection(dir,ranks;label="")
  basis,style,interps = map(ranks) do p
    a = deserialize(_part_filename(dir,HRPROJECTION_LABEL,label,p))
    get_basis(a),get_style(a),get_interpolation(a)
  end |> tuple_of_arrays
  DistributedHRProjection(getany(basis),getany(style),DistributedInterpolation(interps))
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