function save(
  dir,
  contribs::ContributionTuple;
  label=""
  )

  for (i,contrib) in enumerate(contribs)
    save(dir,contrib;label=to_label(label,i))
  end
end

function RBSteady.load_contribution(
  dir,
  trians::Tuple{Vararg{Tuple}};
  label=""
  )

  c = ()
  for (i,trian) in enumerate(trians)
    c = (c...,load_contribution(dir,trian;label=to_label(label,i)))
  end
  return ContributionTuple(c)
end

include("Diagnostics.jl")