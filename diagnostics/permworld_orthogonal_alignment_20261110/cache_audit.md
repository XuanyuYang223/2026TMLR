# Existing cache audit

All 40 standardized feature files for unit02–unit05 and all 10 historical standardized files match their completed prior-probe SHA receipts and can be reused. No feature rebuild is required.

All 24 required newer-unit validation NPZ sources and 6 historical validation NPY sources contain `[5376, 3, 256]` FP32 pre-task `final_norm` states. Both encodings use peaks/initial and valleys/trained for the requested target files; one-line descents/recoils add trained complement controls.

All 45 associated physical checkpoint files and five data files were rehashed. Checkpoint/data/protocol/parameter bindings, full anchor order, and existing validation identity slices agree. Validation has256 and test512 examples per registered length10–30.

Current audit hashed 201 unique files (2,589,404,795 bytes) in 23.49 seconds. This is a named-file audit, not continuous or repository-wide immutability evidence.

No model or probe was constructed or fitted; no forward, backward, alignment fit or old-file write was performed. The ZIP historical extractor is unnecessary for these inputs. Activation provenance relies on the saved producer metadata and current SHA, not an independent forward recomputation.

The JSON records every exact path, expected/current content hash, shape, order, checkpoint binding, and action to export.
