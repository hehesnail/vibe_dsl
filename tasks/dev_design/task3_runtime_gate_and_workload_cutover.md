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
- segment body TIR already selected by `TTProgram` / executable projection
- core type
- launch/core plan
- compile-time arg specs
- runtime and common runtime args
- per-work arg specs
- accessors
- semaphore bindings
- remote core descriptors for synchronization endpoints
- typed compute operation records
- structured physical CB queue events

Leaf readers must require these fields.
Missing maps or arrays are errors, not empty defaults.

Segment body ownership is part of the kernel record.  Final leaf readers may
materialize a segment `PrimFunc` from that explicit body, but they must not
read `blackhole.segment_kind`, scan neighboring builtins, or infer segment
membership from the final function body.  The marker is allowed only as
pass-local lowering mechanics before projection.

Per-work arg specs have one owner for nontrivial dynamic values:
`value_source=value_expr` plus the serialized TIR expression.  Leaf readers
may evaluate that expression against the logical work context, typed compute
records, and any explicitly referenced input buffers, but they must not add
workload-shaped `value_source` enums such as compute extent, row/page, table,
or selection-specific sources.  They also must not classify a binding by
runtime-arg name prefixes such as `per_work_value[_N]`; once a runtime-arg
expression resolves to a per-work spec, `value_source`, `value_expr`, and
`AccessRegion` evidence are the protocol.

Remote synchronization endpoints have the same rule.  A pair of
`logical_core_noc_x/y` runtime args may bind the ABI values consumed by a
leaf builtin, but the endpoint object must already be projected as a
`remote_core_descriptors` segment/kernel record.  Leaf readers may validate
that the runtime args agree with that object; they must not reconstruct the
descriptor from the arg pair when the explicit record is missing.

Physical CB queue events follow the same boundary.  `KernelSpec.queue_events`
is the serialized execution trace projected from validated `TTProgram`
kernels and `TTCBPlan` bindings.  Leaf admission may replay those events to
validate FIFO capacity and visibility, but it must not scan generated source
or reconstruct queue events from segment-body TIR after the executable
projection is already materialized.

### Buffer Identity

Formal buffer identity must be explicit and exact.

Leaf readers must not infer buffer roles from:

- argument position
- name suffixes
- runtime arg kind
- work-linear IDs
- source text

### Buffer Address Contract

`ExecutableSpec`
must carry the runtime-visible buffer address contract projected from
validated `TTProgram` placement.

Required buffer distribution fields:

- buffer identity
- mesh identity / index
- distribution kind
- layout and memory space
- page size
- host visibility
- logical index mapping

For interleaved DRAM runtime buffers,
the executable contract must state:

- `distribution_kind = interleaved`
- `layout = interleaved`
- `memory_space = DRAM`
- positive `page_size_bytes`
- `logical_index_mapping = interleaved_linear_page`

There is no `page_indexed` layout or `page_indexed_accessor_cta`
accessor kind in the executable/runtime contract.  Page-addressed transport
remains ordinary interleaved DRAM plus explicit page-size and logical-index
mapping fields.

For admitted source-backed sharded L1 resident views,
the executable contract must state:

- `distribution_kind = sharded`
- `memory_space = L1`
- `sharding_strategy = height | width | block`
- `shard_orientation = row_major | col_major`
- positive `shard_grid_shape`
- positive per-core `shard_shape`
- `source_buffer`
- `source_region_kind = per_work_tile`
- positive `source_region_shape`
- `logical_index_mapping = work_packet_row_major`
- `core_local_address_mapping = l1_shard_linear`
- attached core-group identity and index

For pure-local sharded buffers that are not materialized from an external
source buffer, the executable contract must still state the sharding,
memory-space, mapping, and core-local fields above, but it must not invent
`source_buffer`, `source_region_kind`, or `source_region_shape`.
Source-region fields are required exactly when a source binding exists.

Leaf readers must validate these fields directly.
Direct runtime admission must consume them before execution.
It may reject unsupported distribution kinds, but it must not recover source
regions, page metadata, or core-local mapping from names, source text, or
argument order.

### CB / Leaf Identity Contract

Projected CB records must carry the physical CB ID and any associated
requirement indices needed to link back to TTProgram allocation and compute
operand bindings.  They must not carry `requirement_names`, name-derived
lookup tables, or other debug-only aliases as leaf-readable protocol.

Leaf codegen may resolve a requirement index through
`ExecutableSpec.cb_configs[*].requirement_indices -> cb_id`, or fail closed
when that mapping is missing or ambiguous.  It must not recover a CB from a
buffer suffix, data-format channel guess, CB config name, or runtime
observation.

Queue-event replay consumes only physical `cb_id` and positive page counts
from `KernelSpec.queue_events`.  Requirement-index remapping is completed at
the `TTProgram -> ExecutableSpec` projection boundary, not in the runtime
module.

The executable sharded fields intentionally mirror TT-Metal's split between
memory-layout strategy and `ShardSpec` orientation:
`height/width/block` are layout strategies, while `row_major/col_major`
describe core traversal.
`source_buffer` and source-region fields are TileLang materialization ABI
fields for DRAM-to-resident-L1 views, not TT-Metal `ShardSpec` fields.

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
