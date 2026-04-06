Base.@propagate_inbounds function RBSteady.contraction(
  factor1::Array{T,3},
  factor2::Array{S,3},
  factor3::Array{U,3},
  combine::TimeCombination
  ) where {T,S,U}

  @check size(factor1,2) == size(factor2,2) == size(factor3,2)
  Nt = size(factor1,2)
  A = reshape(permutedims(factor1,(2,1,3)),size(factor1,2),:)
  B = reshape(permutedims(factor2,(2,1,3)),size(factor2,2),:)
  C = reshape(permutedims(factor3,(2,1,3)),size(factor3,2),:)
  θ = get_coefficients(combine,Nt)
  TSU = promote_type(T,S,U)
  ABC = zeros(TSU,size(A,2),size(B,2),size(C,2))
  for (iA,a) = enumerate(eachcol(A))
    for (iB,b) = enumerate(eachcol(B))
      for (iC,c) = enumerate(eachcol(C))
        for γ = eachindex(θ)
          for n in axes(factor1,2)
            n+γ > Nt && break
            RBSteady._entry!(+,ABC,θ[γ]*a[n+γ-1]*b[n+γ-1]*c[n],iA,iB,iC)
          end
        end
      end
    end
  end
  s1,s2 = size(factor1,1),size(factor1,3)
  s3,s4 = size(factor2,1),size(factor2,3)
  s5,s6 = size(factor3,1),size(factor3,3)
  ABCp = permutedims(reshape(ABC,s1,s2,s3,s4,s5,s6),(1,3,5,2,4,6))
  return ABCp
end
