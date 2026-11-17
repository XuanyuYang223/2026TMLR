# Exceedance/inverse-deficiency mathematical audit

The identity is exact: for `j = pi(i)`, `E_i(pi) = 1[j>i] = D_j(inverse(pi))`. Fixed points are negative in both labels. Target-owned scores or priors require the `j=pi(i)` gather exactly once; a transferred source-index output must not be gathered again.

Twenty-six tests pass. They cover all 46,233 permutations through n=8 for label identities/count-registry agreement and all 5,913 permutations through n=7 under both real encodings for token locations. An independent streaming parser checks all 80,638 edge reveal cases. Explicit examples cover inverse closure roles, singleton cycles, fixed negatives, and incorrect double reordering.

Indices include the actual BOS/SIZE header. One-line edge after-state is its value token; before-state is the preceding token. In canonical cycles, a nonclosing edge is explicit at its second numeric endpoint; closing edges require the semicolon or CYCLE_END. Singletons use that delimiter and are kept distinct.

These are local explicit-edge reveal points, not the earliest logical entailment among all valid completions. Permutation/canonical-order constraints and n/i class priors can already inform the before-state. A diagnostic must retain this limitation, and report closing and singleton edges separately. No model, forward pass, fitting or probe outcome was used here.
