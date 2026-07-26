# Result archive format

Results are compressed `.npz` files with schema version 1. They contain only numeric arrays
and Unicode scalars. Loading always uses `allow_pickle=False`.

Core fields include particle positions, particle radius, seed, backend, restart mode, growth and
calibration counters, calibration centers/sizes/bounds/failure allocations/death ratios, and a
JSON metadata object.

Writes are atomic: data are fsynced to a temporary file in the destination directory and then
installed with `os.replace`.
