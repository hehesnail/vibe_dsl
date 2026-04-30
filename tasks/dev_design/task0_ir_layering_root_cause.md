# Task 0: Root Cause And Rewrite Direction

## Role

This document records the root cause behind the Blackhole backend rewrite.
It is a design diagnosis, not a progress log.

Overall design:
`final_blackhole_backend_redesign.md`.
Current status:
`tasks/progress.md`.

## Root Cause

The core problem was not a lack of late matchers.
It was that cross-stage program truth escaped the explicit IR chain and was
carried by attrs, bags, helper wrappers, payloads, naming conventions, and
runtime fallbacks.

The durable fix is one explicit owner-truth chain:

```text
Normalized Tile TIR
  -> SpatialPlan
  -> TTProgram
  -> ExecutableSpec
```

Every downstream decision must be derived from the current explicit layer or
from the next lowered typed representation.

## Failure Pattern

Historical Blackhole code drifted into three coupled problems:

- tile compute semantics were destroyed by scalar lowering before exact
  TT-Metal leaf legality was established
- `SpatialPlan` was too thin to own virtual spatial/dataflow semantics
- downstream planning, codegen, and runtime recovered semantics from side
  channels

That combination produced:

- late scalar-loop matchers
- contract maps
- transition attrs
- wrapper/facts bags
- leaf name/default fallbacks
- runtime/codegen semantic recovery

These surfaces are implementation debt unless rewritten into the correct
typed layer.

## Correct Boundary

### Normalized Tile TIR

Owns algorithmic compute and access semantics:

- tile op semantics
- `BufferLoad` / `BufferStore`
- address expressions
- loop/domain/predicate structure
- explicit TT-Metal leaf tile-compute statements

It does not own TT placement, CB allocation, runtime args, launch order, or
backend admission.

### SpatialPlan

Owns target-independent virtual spatial/dataflow semantics:

- execution units
- dataflow / carry / reduction / broadcast / join edges
- virtual layout/distribution
- phase and materialization boundaries
- logical live values
- validated hints

It does not own TT builtins, CB IDs, semaphores, runtime args, or physical
placement.

### TTProgram

Owns TT-specific target realization:

- hardware model facts used by planning
- mesh / device / core placement
- buffer distribution
- kernel, block, compute, transport, sync, ABI, runtime-arg plans
- resource demand and pressure reports

It does not own target-independent semantic recovery or leaf/backend fallback.

### ExecutableSpec

Owns leaf projection and backend admission:

- projected kernel / segment / CB / semaphore / runtime records
- formal buffer identities
- backend admission results
- runtime module build inputs

It does not own planning or semantic recovery.

## Design Rules

- Cross-stage truth must be typed and explicit.
- Pass-local analysis may be temporary, invalidatable, and recomputable only.
- If current IR cannot prove a fact, extend the IR/DSL or reject.
- Do not infer semantics from names, source text, argument positions, or
  runtime observations.
- A validator must reject missing or inconsistent owner-truth fields before
  source/runtime emission.
- Current implementation residue is never design legitimacy.

## Support-Surface Interpretation

Remaining gaps should be classified by layer:

- missing normalization belongs to `Normalized Tile TIR`
- missing virtual dataflow belongs to `SpatialPlan`
- missing TT physical realization belongs to `TTProgram`
- missing leaf projection or backend admission belongs to `ExecutableSpec`

Do not solve a lower-layer gap by reintroducing a side channel above or below
it.

## Completion Standard

The rewrite direction is satisfied only while:

- the active chain remains centered on the four explicit layers
- legacy bags/payloads/contracts are not reintroduced as public protocol
- typed validators fail closed
- source/runtime consumers do not reconstruct planner semantics
- docs and `tasks/progress.md` stay separated by role
