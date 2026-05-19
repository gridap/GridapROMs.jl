Base.@propagate_inbounds function RBSteady.contraction(
  factor1::AbstractArray{T,3},
  factor2::AbstractArray{S,3},
  factor3::AbstractArray{U,3},
  combine::TimeCombination
  ) where {T,S,U}

  @check size(factor1,2) == size(factor2,2) == size(factor3,2)
  Nt = size(factor1,2)
  A = reshape(permutedims(factor1,(2,1,3)),Nt,:)
  B = reshape(permutedims(factor2,(2,1,3)),Nt,:)
  C = reshape(permutedims(factor3,(2,1,3)),Nt,:)
  θ = get_coefficients(combine,Nt)
  TSU = promote_type(T,S,U)
  ABC = zeros(TSU,size(A,2),size(B,2),size(C,2))
  tmp = zeros(TSU,Nt,size(C,2))
  @inbounds for iC in axes(C,2)
    for m = 1:Nt
      t = zero(TSU)
      γmax = min(length(θ),m)
      for γ = 1:γmax
        t += θ[γ]*C[m-γ+1,iC]
      end
      tmp[m,iC] = t
    end
  end
  @inbounds for (iA,a) = enumerate(eachcol(A))
    for (iB,b) = enumerate(eachcol(B))
      for iC in axes(C,2)
        s = zero(TSU)
        for m = 1:Nt
          s += a[m]*b[m]*tmp[m,iC]
        end
        RBSteady._entry!(+,ABC,s,iA,iB,iC)
      end
    end
  end
  s1,s2 = size(factor1,1),size(factor1,3)
  s3,s4 = size(factor2,1),size(factor2,3)
  s5,s6 = size(factor3,1),size(factor3,3)
  ABCp = permutedims(reshape(ABC,s1,s2,s3,s4,s5,s6),(1,3,5,2,4,6))
  return ABCp
end

Base.@propagate_inbounds function _contraction(
  factor1::AbstractArray{T,3},
  factor2::AbstractArray{S,3},
  args...
  ) where {T,S} 

  @check size(factor1,2) == size(factor2,2) 
  @check size(factor1,1) == size(factor2,1)
  TS = promote_type(T,S)
  AB = zeros(TS,size(factor1,3),size(factor2,3))
  for iA in axes(factor1,3)
    for iB in axes(factor2,3)
      for i1 in axes(factor1,1)
        for n in axes(factor1,2)
          AB[iA,iB] += factor1[i1,n,iA]*factor2[i1,n,iB]
        end
      end
    end
  end
  AB
end

Base.@propagate_inbounds function _contraction(
  factor1::AbstractArray{T,3},
  factor2::AbstractArray{S,4},
  factor3::AbstractArray{U,3},
  combine::TimeCombination
  ) where {T,S,U}

  @check size(factor1,2) == size(factor2,2) == size(factor3,2)
  @check size(factor1,1) == size(factor2,1)
  @check size(factor2,4) == size(factor3,1)
  Nt = size(factor1,2)
  θ = get_coefficients(combine,Nt)
  TSU = promote_type(T,S,U)
  ABC = zeros(TSU,size(factor1,3),size(factor2,3),size(factor3,3))
  tmp = zeros(TSU,size(factor3,1),Nt,size(factor3,3))
  @inbounds for iC in axes(factor3,3)
    for m = 1:Nt
      γmax = min(length(θ),m)
      for i3 in axes(factor3,1)
        t = zero(TSU)
        for γ = 1:γmax
          t += θ[γ]*factor3[i3,m-γ+1,iC]
        end
        tmp[i3,m,iC] = t
      end
    end
  end
  @inbounds for m = 1:Nt
    for iA in axes(factor1,3)
      for iB in axes(factor2,3)
        for iC in axes(factor3,3)
          for i1 in axes(factor1,1)
            for i3 in axes(factor3,1)
              v = factor1[i1,m,iA]*factor2[i1,m,iB,i3]*tmp[i3,m,iC]
              RBSteady._entry!(+,ABC,v,iA,iB,iC)
            end
          end
        end
      end
    end
  end
  ABC
end