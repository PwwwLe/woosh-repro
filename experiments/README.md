# Experiment Logs

Use this directory for reproduction records that should not change the official baseline code.

Suggested layout:

```text
experiments/
├── logs/              # command output, timing logs, environment captures
└── runs/              # generated audio/video and per-run metadata
```

Record at least:

- git commit
- conda environment name
- install command
- checkpoint release/version
- exact command
- prompt, seed, model name, device
- output path
- known failure or limitation
