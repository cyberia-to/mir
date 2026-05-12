---
tags: cyber, core
crystal-type: entity
crystal-domain: cyber
alias: mir, render engine, world renderer, R-1.0 implementation
---
# mir

Russian мир: world, peace, community.

mir is the render engine for [[cyber]]. it reads two inputs and produces one output:

```
tru   →  positions (φ*, eigenvectors, spectral coords)  ─┐
                                                           ├→  mir  →  R-1.0 world
glia  →  inference outputs (neural features)            ─┘
```

every [[neuron]] running mir on the same graph state arrives at the same world. not approximately — topologically identical. the math island is at the same BVH path for everyone.

mir knows nothing about graphs, particles, or cyberlinks. it receives coordinates and features and makes them visible.

## what mir does

| epoch work (≤ 1 Hz) | frame work (display refresh) |
|---|---|
| LOBPCG eigensolver on normalized Laplacian ℒ | GPU BVH frustum cull → VisibleSet |
| Procrustes alignment to anchor-1024 | tier dispatch (T0–T3 draw calls) |
| heat-kernel BVH at four τ scales | focus luminosity animation |
| NRF training (Phase 2+) | edge flow UV offset update |
| results double-buffered into EpochState | IOSurface composite (zero copy) |

between epochs, positions and cluster IDs are frozen. only focus luminosity and edge flow animate within a frame epoch.

## rendering tiers

| tier | condition | what renders |
|---|---|---|
| T0 content | screen diameter > 200 px | particle opens — content environment |
| T1 surface | 40–200 px | impostor + text label + edge preview |
| T2 shape | 8–40 px | analytic impostor (indirect draw, tile-shaded) |
| T3 splat | 1–8 px | 3D Gaussian splat (Kerbl 2023) |
| T∞ field | < 1 px | neural radiance field — cost per pixel, not per particle |

T∞ is the tier that makes R-1.0 scale to unbounded graph size. every frame costs pixels × samples, independent of particle count.

## hardware (honeycrisp backend)

| component | role |
|---|---|
| [[aruminium]] | Metal GPU: BVH cull compute, impostor / splat / edge draw, composite |
| [[rane]] | ANE: NRF head inference (Phase 2+) |
| [[acpu]] | AMX: LOBPCG eigensolver, Procrustes alignment, BVH build, diffusion steps |
| [[unimem]] | IOSurface: zero-copy frame handoff across CPU / GPU / ANE |

target: M3 Pro at 4K 120 Hz (8.3 ms frame budget), 1 M-particle graph.

## what mir is not

mir is not the graph engine. it does not compute φ*, eigenvectors, or cyberank — those come from [[tru]]. it does not run inference — that comes from [[glia]]. it receives results and renders them.

## specs

- `reference/render.md` — R-1.0 canonical protocol spec (from [[cybergraph]])
- `specs/render.md` — R-1.0-cyb: cyb implementation of R-1.0
- `reference/plan.md` — Phase 1 implementation plan and architectural decisions

## implementation order (Phase 1)

| step | module | deliverable |
|---|---|---|
| 1 | `graph/` | snapshot loader, CSR adjacency, particle index |
| 2 | `epoch/eigensolver` | LOBPCG + Procrustes aligned positions |
| 3 | `epoch/bvh` | heat-kernel BVH, 4 τ scales |
| 4 | `frame/cull` | GPU BVH traversal → VisibleSet + TierLevel |
| 5 | `frame/tiers/t2` | analytic impostor (indirect draw, tile-shaded) |
| 6 | `frame/tiers/t3` | Gaussian splat (Kerbl 2023) |
| 7 | `bevy/` | GraphWorldPlugin, WorldState::Graph, Cmd+5 |
| 8 | `frame/tiers/t1` | labels via sugarloaf, world-space text |
| 9 | `frame/edges` | bundled tubes + flow UV animation |
| 10 | `frame/tiers/t0` | content entry: camera transition + sandbox |
| 11 | conformance | P-RENDER-TOPO, P-RENDER-POS, P-RENDER-FPS |

Phase 2: T∞ hash-grid MLP (no CT-1.1).
Phase 3: T∞ full NRF — CT-1.1 + Clifford block + volume ray-march.

## in the stack

```
tru (compile model)  →  .model + φ* + eigenvectors
glia (run model)     →  neural features
mir (render)         →  R-1.0 world
```

see [[stack]]
