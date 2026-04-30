# Task 3: ExecutableSpec / Leaf Reader Contract

## Role

This document defines the durable `ExecutableSpec` and leaf-reader contract.
The historical filename contains `runtime_gate_and_workload_cutover`, but the
long-term boundary is the representation, not the pass name.

Overall design:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Goal

`ExecutableSpec`
is the canonical leaf projection and runtime-module build contract derived
from validated `TTProgram`.

It answers:

- which entry and kernels belong to the executable
- which projected CB / semaphore / core / accessor / runtime-arg records are
  available to leaf consumers
- which formal buffer identities are bound to runtime and compile-time ABI
  records
- which backend admission reasons apply
- how `BlackholeModule` or codegen/export materializes the runtime module

It does not answer:

- target planning
- compute legality
- resource allocation
- semantic recovery from source text, work IDs, builtin sequences, names, or
  argument positions

## Inputs

Allowed inputs:

- validated `TTProgram`
- canonical `MaterializeBlackholeExecutable` projection
- leaf-local schema validation
- backend admission checks

Forbidden inputs:

- `blackhole.copy_semantics`
- `blackhole.segment_kind` after final projection
- lowering facts or helper bags
- payload fallbacks
- implicit buffer-role recovery
- runtime/codegen reconstruction of planner decisions

If leaf inputs are insufficient, fix `TTProgram` or projection.
Do not add a leaf-time matcher.

## Executable Truth

### Identity

Required:

- schema version
- source representation identity
- entry identity
- member function identity

The source identity may describe provenance from `tl.tt_program`.
It must not become a planner payload.

### Kernel / Segment Records

Required:

- segment identity and kind
- core type
- launch/core plan
- compile-time arg specs
- runtime and common runtime args
- per-work arg specs
- accessors
- semaphore bindings
- typed compute operation records

Leaf readers must require these fields.
Missing maps or arrays are errors, not empty defaults.

### Buffer Identity

Formal buffer identity must be explicit and exact.

Leaf readers must not infer buffer roles from:

- argument position
- name suffixes
- runtime arg kind
- work-linear IDs
- source text

### Backend Admission

Admission reasons must be typed and queryable:

- unsupported layout
- missing accessor proof
- unsupported synchronization / event lifetime
- resource pressure
- runtime support not admitted
- simulator capability boundary when applicable

Backend admission cannot remove the need for schema-complete
`ExecutableSpec` projection.

### Runtime Module Serialization

If a runtime module advertises binary serialization, it must provide real
non-empty bytes and a matching loader.

For Blackhole this means:

- `SaveToBytes` writes a versioned module payload
- `ffi.Module.load_from_bytes.blackhole` restores it
- file-level `WriteToFile` stays fail-closed until a real file format exists

## Validation Contract

Leaf validation must reject:

- missing required maps/arrays
- default core fallback
- unknown compute op records
- missing formal buffer identity
- runtime args without explicit schema
- unsupported direct-runtime admission cases
- serialization contracts that cannot be loaded

## Exit Invariant

After `ExecutableSpec`,
runtime/codegen/export can either build the executable directly or fail
closed with a typed reason.

No leaf consumer may rebuild planner semantics.
