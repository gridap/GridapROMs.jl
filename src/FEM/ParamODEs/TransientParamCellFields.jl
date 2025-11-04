function ODEs.TransientCellField(f::SingleFieldParamFEFunction,derivatives::Tuple)
  ODEs.TransientSingleFieldCellField(f,derivatives)
end

function ODEs.TransientCellField(multi_field::MultiFieldParamFEFunction,derivatives::Tuple)
  transient_single_fields = ODEs._to_transient_single_fields(multi_field,derivatives)
  ODEs.TransientMultiFieldCellField(multi_field,derivatives,transient_single_fields)
end
