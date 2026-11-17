# Explicit-edge token diagnostic

This stage uses a newly fitted **scalar shared probe**, separately from the prefix-final multi-output probe.
All base networks are frozen. The same source AFTER probe and AFTER centering are used before and after; BEFORE is a distribution-shift diagnostic, not proof of absent information.
Native inverse target edge j=pi(i) is gathered once. Transferred scores already have source i indexing.
Primary AUROC averages anchors within length and then uses all 21 lengths with fixed 1/21 weights. Missing lengths propagate undefined; no zero imputation.
Pooled-edge AUROC is supplementary. Canonical source closures are all E=0, inverse-target closures all paired E=1, and singletons E=0; their individual AUROCs are structurally undefined.
The both-internal and token-kind-prior controls are frozen supplemental analyses, not the primary endpoint.

Full machine-readable results: `per_unit.csv`, `cohort_summary.csv`, per-unit `by_length.csv`; score tensors, probes, maps and shuffles are on the external F drive.

| Unit | Encoding | Point | Source own | Direct | Fit inverse | Best wrong map | Initial target | Fixed target | Target own |
|---|---|---|---:|---:|---:|---:|---:|---:|
| archdev001 | canonical_cycle | before | 0.5181 | 0.4569 | 0.5137 | 0.5905 | 0.5825 | 0.6244 | 0.4670 |
| archdev001 | canonical_cycle | after | 0.9919 | 0.5284 | 0.6965 | 0.6572 | 0.6012 | 0.6536 | 0.9178 |
| archdev001 | one_line | before | 0.7182 | 0.5776 | 0.7012 | 0.6510 | 0.7867 | 0.7717 | 0.7293 |
| archdev001 | one_line | after | 0.9534 | 0.5755 | 0.7886 | 0.7108 | 0.7936 | 0.7884 | 0.9547 |
| unit02 | canonical_cycle | before | 0.4168 | 0.4836 | 0.5053 | 0.6235 | 0.6163 | 0.6478 | 0.4362 |
| unit02 | canonical_cycle | after | 0.9820 | 0.5432 | 0.7409 | 0.6512 | 0.6294 | 0.6516 | 0.9099 |
| unit02 | one_line | before | 0.7350 | 0.5081 | 0.7094 | 0.5642 | 0.7867 | 0.7948 | 0.7048 |
| unit02 | one_line | after | 0.9468 | 0.5285 | 0.7541 | 0.6360 | 0.7947 | 0.8195 | 0.9476 |
| unit03 | canonical_cycle | before | 0.5806 | 0.4953 | 0.4867 | 0.5653 | 0.5962 | 0.5857 | 0.5021 |
| unit03 | canonical_cycle | after | 0.9974 | 0.4764 | 0.7772 | 0.7183 | 0.6137 | 0.6287 | 0.9153 |
| unit03 | one_line | before | 0.6892 | 0.5045 | 0.6840 | 0.5609 | 0.7895 | 0.7324 | 0.7269 |
| unit03 | one_line | after | 0.9588 | 0.5401 | 0.7605 | 0.6323 | 0.7998 | 0.7876 | 0.9605 |
| unit04 | canonical_cycle | before | 0.4705 | 0.4419 | 0.5313 | 0.6302 | 0.6052 | 0.6353 | 0.5162 |
| unit04 | canonical_cycle | after | 0.9729 | 0.4559 | 0.7246 | 0.6178 | 0.6206 | 0.6713 | 0.9192 |
| unit04 | one_line | before | 0.7068 | 0.4950 | 0.7167 | 0.6582 | 0.7787 | 0.7704 | 0.6946 |
| unit04 | one_line | after | 0.9670 | 0.5008 | 0.7957 | 0.7258 | 0.7905 | 0.7885 | 0.9795 |
| unit05 | canonical_cycle | before | 0.4516 | 0.4803 | 0.4540 | 0.6404 | 0.6053 | 0.6084 | 0.5390 |
| unit05 | canonical_cycle | after | 0.9715 | 0.5002 | 0.7175 | 0.6639 | 0.6194 | 0.6310 | 0.9431 |
| unit05 | one_line | before | 0.7433 | 0.5454 | 0.7161 | 0.5677 | 0.7965 | 0.7940 | 0.6996 |
| unit05 | one_line | after | 0.9518 | 0.5322 | 0.7855 | 0.6240 | 0.8088 | 0.8187 | 0.9522 |
