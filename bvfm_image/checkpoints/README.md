# Selected checkpoint

`bvfm_image_step40000.pt` is the selected inference checkpoint.

- Format: `flowtok_bvfm_shared_variational_v2`
- Step: 40,000
- Contains: shared FlowTok field, full-pair posterior, text/image priors, and
  AR caption decoder
- Does not contain optimizer state

It was selected over the old `best_joint.pt` at step 1,000 because the latter
was chosen only by a proxy velocity-drift gate. Step 40,000 is the checkpoint
that was validated end-to-end for both T2I and I2T.
