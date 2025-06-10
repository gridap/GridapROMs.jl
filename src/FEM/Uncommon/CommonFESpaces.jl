function common_space(f::Vector{<:FESpace})
  @abstractmethod
end

function common_space(f::Vector{<:DirectSumFESpace})
  space = common_space(map(get_fe_space,f))
  complementary = complementary_space(space)
  DirectSumFESpace(space,complementary)
end

get_free_indices(f::EmbeddedFESpace) = f.fdof_to_bg_fdofs
get_diri_indices(f::EmbeddedFESpace) = f.ddof_to_bg_ddofs

function common_space(f::Vector{<:EmbeddedFESpace})
  fdof_to_bg_fdofs = common_indices(map(get_free_indices,f))
  ddof_to_bg_ddofs = common_indices(map(get_diri_indices,f))
  bg_cell_dof_ids = common_table(map(get_bg_cell_dof_ids,f))
  EmbeddedFESpace(bg_space,bg_space,fdof_to_bg_fdofs,ddof_to_bg_ddofs,bg_cell_dof_ids)
end
