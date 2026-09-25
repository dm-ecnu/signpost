# X4 primary model, fixed before the MuSiQue held-out result (2026-09-25T18:14:05+08:00)

- Primary: entries_share_scaled, t_hat = a * (sum of stored list entries over dirty views / over all views), a = 0.916 fitted on the five frozen e4 corpora (25 cells).
- Selected by leave-one-corpus-out MAE (0.046; byte share 0.070, dirty count 0.113); MuSiQue not used for selection or fitting.
- Decision rule: incremental iff t_hat < 1; no threshold tuning.
- Known limitation before held-out test: shares are <= 1, so the rule cannot predict t/t_full > 1 (Legal 10%, measured 1.044, predicted 0.909).
- fit5.json sha256: 0d70695e3a4d7bc618ccdd7339861a45fa97a281a4e9ed28b579229370a1ecca
- Disclosure: measured MuSiQue t/t_full for the 0.1/0.5/1% batches (0.683/0.831/0.889) had appeared in the X2 progress log before this file was written; MuSiQue features had not been computed and were not seen.
