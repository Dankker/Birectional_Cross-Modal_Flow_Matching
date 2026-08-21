# Model assets

Training expects:

```text
assets/FlowTok-XL.pth
assets/FlowTiTok_512.bin
assets/decoder_init.pt
```

- `FlowTok-XL.pth`: released FlowTok-XL vector field initialization.
- `FlowTiTok_512.bin`: released 512px image tokenizer.
- `decoder_init.pt`: AR text-decoder warm start. Its task-conditioned vector
  field is never loaded.

These binary files are ignored by Git. Place them at the paths above or
override their locations with `FLOWTOK_CKPT`, `FLOWTITOK_CKPT`, and
`DECODER_INIT`.
