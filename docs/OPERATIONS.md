# Safe Operation and Recovery

## Before every run

1. Stop any synchronizer, reader, backup tool, or process that could change the
   same image tree.
2. Copy `config.example.toml` to an ignored local `config.toml`. Keep the input
   and output roots disjoint in `mirror` mode.
3. Confirm both Real-HAT checkpoint files are present and have passed their
   manifest SHA-256 check.
4. Run `inspect_workload.bat` before starting GPU work. It is read-only and
   reports the planned normal/sharper routes, output collisions, and safety
   limits.

Do not use a generated JXL as a later super-resolution input. A non-JXL source
remains authoritative for planning; JXL is an output format only.

## Mirror first

The public template uses:

```toml
[output]
mode = "mirror"
existing_jxl_policy = "error"
```

This is the required first workflow for a new source tree or a changed model,
threshold, tile setting, or JPEG XL configuration. Inspect both the output
library and the source library before considering source replacement.

## Explicit replace workflow

`replace` is intentionally gated. It may remove a non-JXL source only after
these steps succeed in order:

1. Write the candidate to a same-directory temporary file.
2. Decode it with `djxl` and verify planned dimensions and candidate hash.
3. Atomically place the final JXL and verify that final file again.
4. Delete the source only after the final verification succeeds.

Enable it only in a local configuration after a successful mirror run:

```toml
[output]
mode = "replace"
existing_jxl_policy = "replace"
allow_lossy_replace = true
allow_metadata_loss = true
```

`jxl.distance > 0` is lossy. The pipeline rejects unsupported alpha or bit-depth
loss rather than silently flattening or reducing precision unless the matching
local acceptance settings are deliberately changed. Keep backups; ordinary file
deletion may not be recoverable through a desktop recycle bin.

## Metrics

Pass `--metrics-dir` to the worker only when you need timing records. Metrics
are observability data, never transaction authority. Put the directory outside
input and output roots, redact it before sharing, and do not commit it.

## Errors and recovery

For a nonzero exit:

1. Confirm that no watchdog or worker process is still active.
2. Preserve the state journal, worklist, source file, final JXL, and any `.part`
   candidate exactly as they are.
3. Rerun with the same configuration and checkpoints.

The recovery code rechecks source identity, configuration/model signature,
candidate content, and final JXL before deciding whether a transaction can
continue. If it cannot prove the safe action, it retains both files for manual
inspection. Never delete recovery material merely to make a later run start.

## Completion

Run `inspect_workload.bat` once more after a successful job. An empty planned
workload, no outstanding transaction state, and no unexpected same-stem output
conflict are the completion condition. `processed=0` is valid only when it
matches the read-only inspection result.
