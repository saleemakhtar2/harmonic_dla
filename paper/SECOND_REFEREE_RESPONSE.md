# Response to the second referee report

We thank the referee for the careful second review and the recommendation of acceptance subject to
minor revision. We made every requested correction.

1. We repaired the broken `rho^4` typesetting in the calibration-cost discussion.
2. We regenerated the calibration-cost figure so the confidence symbol is `eta`, matching the
   manuscript.
3. Corollary 8.3 now uses `n_0` consistently in the path law, schedule, and Hurwitz-zeta tail.
4. Remark 9.12 now says that the sufficient constraints derived in the paper meet at `11/24`; it no
   longer suggests an unproved lower-bound theorem.
5. Theorem 9.7 and Algorithm 1 now impose the explicit condition
   `tau_0 >= max(n_geom, ceil(2^(1/kappa)))`.
6. The legacy death-circle restart remark is phrased through an effective radius `b < d` and the
   limit `b -> d`, avoiding a degenerate continuum Brownian start on an absorbing boundary.
7. The Green-function stopping proof now explains the exhaustion argument, monotone convergence,
   zero death-circle boundary value, and the polar exceptional set needed for equality.
8. Figure 8's caption now states how a runtime-verified diameter exponent `xi > 1/2` shifts the
   feasible region via Corollary 9.9.
9. We audited the conjugation convention in the compact-set identity, the definition of the
   Neumann operators, and the final substitution. The coefficient of `eta_n^c` is consistently
   `conjugate(M_n^c(mu))`.
10. We also made the small grammatical correction noted during final proofreading.

The revised PDF was rebuilt, preflighted, rendered in full, and visually inspected.
