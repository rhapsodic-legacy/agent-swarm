# The Priority Market — One Substrate for Every Signal

*A walkthrough of the architectural decision at the heart of this project: why every input to the swarm — operator clicks, LLM chat, environmental hazards, adaptive learning — flows through a single auction.*

---

## The problem I kept running into

Multi-agent systems accumulate input modalities. You start with one: an operator paints a priority zone. Then you add an LLM mission planner that can issue directives. Then you add chat: natural language that produces coordinates. Then you realize wind should steer drones away from hazardous regions. Then you want the system to *learn* which signals actually produce finds, so it can weight trusted sources higher over time.

The naive path is to wire each modality independently into the drone's decision loop — a pile of `if-else` branches that each handle their own signal type. Each one looks reasonable in isolation. Together they become unmaintainable. Worse, they're brittle: fixing a bug in the avoid-zone path doesn't automatically fix the equivalent bug in the wind-hazard path, because they're separate code.

I wanted one thing the drones consume — a stream of weighted points they bid on — and every input source becomes a producer of that stream. The drones never learn about new modalities. They just bid.

## The shape of the solution

```
Producers (compose freely; adding one is a one-line change)
──────────────────────────────────────────────────────────
  Bayesian PoC grid       →  PriorityAsset(source="poc_field")
  Operator high-zone      →  PriorityAsset(source="operator_high_zone")
  LLM / chat intel pin    →  PriorityAsset(source="intel_pin")
  Survivor find           →  PriorityAsset(source="survivor_find")
  Operator avoid zone     ─┐
  Active wind gust region ─┴→  is_in_avoid_zone(x, z) callback
                                │
                                ▼
                     ┌─────────────────────┐
                     │  clear_market()     │
                     │  global auction     │
                     └─────────────────────┘
                                │
                                ▼
                 Per-drone assignment this tick
```

Every positive signal becomes a `PriorityAsset` — a tuple of `(x, z, radius, value, source, expires_tick)`. Every negative signal becomes an entry in a single `is_in_avoid_zone(x, z) -> bool` callback. A once-per-tick auction ([`clear_market()`](../backend/src/agents/priority_market.py)) resolves the whole thing.

Drones bid through one function:

```python
bid = value × source_scale[source]
      / (1 + distance_penalty × dist_to_asset + base_return_penalty × dist_to_base)
      / (1 + switching_cost[current_task])
```

Three invariants fall out of this for free:

1. **Battery feasibility is a hard zero.** A drone whose round-trip exceeds its remaining range after the safety margin bids zero. It literally can't compete for something it can't reach.
2. **Distance is a soft penalty.** Far drones bid less than close drones for the same asset. Far-side swarm members don't run across the map for a marginal target.
3. **Switching cost is stickiness.** A drone investigating a survivor find has a high switching cost; its bids for new assets are dampened, so it doesn't peel off.

## Why this specific form

I considered three alternatives before committing to the market:

- **Per-modality priority lists.** Each modality ranks its own targets; drones consume a union. Easy to implement, impossible to weight against each other in any principled way. How much is a "high-priority operator zone" worth relative to a "3-hour-old intel pin"? Without a common scale, you're guessing.
- **A monolithic priority field (heatmap).** Every signal rasterizes into one 2D grid of priority values; drones gradient-descend it. This was my first instinct. It breaks down once signals have non-local semantics: an avoid zone shouldn't just lower priority in its area, it should invalidate any target that passes through it. An intel pin with a TTL is more naturally a point object than a smeared field.
- **One auction over point objects with explicit source tags.** What shipped. Each asset carries its source label, a global `source_value_scale` dict maps labels to multipliers, and the auction enforces capacity/feasibility invariants uniformly.

The auction won because it gave me one honest answer to the comparability problem (source scales) and let point objects stay point objects.

## Adaptive weights: making the substrate learn

Hardcoded source scales are brittle. An operator might over- or under-trust their own zones relative to LLM pins. The value an intel pin should have depends on whether similar intel pins have historically produced finds.

So the source scales aren't fixed. They're the *baseline* from [`PriorityWeights`](../backend/src/agents/priority_market.py); they're then run through an [`AdaptiveWeights`](../backend/src/agents/adaptive_weights.py) layer that:

- **Credits sources for finds.** When a survivor is discovered, any asset whose area covered that survivor at the time gets a scale bump on its source.
- **Debits sources for unused expiry.** An intel pin that expires without producing a find takes a scale hit on `intel_pin`.
- **Records operator overrides.** A human click that redirects a drone bumps the switching cost on that drone's prior task type — the system learns what task the operator doesn't trust.
- **Gentle decay.** Every tick, learned adjustments pull toward 1.0 (neutral) at a small rate, so transient noise can't lock in a bad weight.

A single **trust slider** (0.0–3.0) in the UI multiplies only operator-origin source scales at runtime. Dial it up and the swarm pursues every operator hint; dial it down and operator signals become suggestions that compete with PoC and LLM input fairly.

This is where the market shape pays off a second time: adaptive weights touch *source scales*, not drone logic. The learning layer is a pure wrapper around `PriorityWeights`.

## Environmental signals flow through the same pipe

The clearest test of "is this really a substrate, or is it just abstraction theater?" is whether an unrelated signal type composes without special-casing drones.

Wind was that test.

The `WeatherSystem` ([`weather.py`](../backend/src/simulation/weather.py)) seeds six `GustRegion`s per mission; each cycles on/off on its own 60–150s sine period. When a region's current strength crosses a threshold, it becomes hazardous. The system exposes one predicate:

```python
weather.is_hazardous_at(x, z) -> bool
```

That's the entire integration surface. In the coordinator's market step, the `is_in_avoid_zone` callback composes operator avoid zones with wind hazards:

```python
def is_in_avoid(x, z):
    if any(zone.contains(x, z) for zone in avoid_zones):
        return True
    if wind_hazard_fn is not None and wind_hazard_fn(x, z):
        return True
    return False
```

No drone code changed. No new asset type. The existing auction just filters a few more candidates. When a gust region activates mid-mission, drones with targets inside it have those targets scrubbed on the next tick; their subsequent re-bid picks a safe target naturally.

This is what "substrate" means in practice. A new input modality is one predicate and one line of composition.

## The rest of the stack, briefly

The market is the load-bearing idea, but it sits inside a more conventional three-tier architecture:

**Tier 3 — Reactive (20 Hz, every tick, classical).** Physics, pathfinding (A* + potential-field repulsion), search-pattern generation (lawnmower / frontier / priority), market bidding. Zero API latency on the critical path. This layer runs standalone if every LLM key is missing.

**Tier 2 — Tactical (~5 s per drone, async, Mistral).** Per-drone decisions like "investigate that signal fire" or "reposition for better coverage." Rate-limited, cached by state hash, and always allowed to time out — the classical layer below it carries on.

**Tier 1 — Strategic (event-driven, async, Claude).** Fleet-level briefings, zone priority shifts, natural-language chat. Emits the same `PriorityAsset` stream that operators do. Treated as another producer.

The async layers are important. The sim loop is synchronous and deterministic; nothing the LLMs do can block or non-determinize it. Every LLM output is a *command* fed to the next tick, just like a human mouse click.

## Concrete consequences of the design

A few places where this architecture paid off mid-development:

**Bug hunting becomes one-site debugging.** When a priority signal isn't producing the expected drone behavior, you don't hunt through modality-specific branches. You inspect the `PriorityAsset` stream that tick (they're dataclasses — trivial to log) and the market's output. If the asset is absent, the producer is wrong. If the asset is present but nobody won it, the bid math is wrong. Two things to check, not nine.

**Testing is compositional.** The market has its own test suite ([`test_priority_market.py`](../backend/tests/test_priority_market.py)). Each producer is tested against a stubbed market. The wind-hazard integration test ([`test_wind_market.py`](../backend/tests/test_wind_market.py)) feeds a synthetic `wind_hazard_fn` into `coordinator.update` and asserts the expected avoidance — no live `WeatherSystem` needed.

**Adding a modality is cheap.** When I wired wind gusts, the bulk of the work was in the `WeatherSystem` itself: seeding regions, computing activation, exposing the predicate. The agent layer got two lines: one to accept the callback on `coordinator.update`, one to compose it into `is_in_avoid`. The drones got nothing.

**The invariant resists erosion.** Every time I was tempted to special-case a signal ("what if *this* one needed to behave differently?"), the question became "can this be expressed as a source scale, a capacity cap, or an avoid predicate?" Almost every time the answer was yes. When the answer was no (e.g., intel pin TTL — a point asset with expiration), the data model absorbed it (`expires_tick` field on `PriorityAsset`), and the auction pruned expired assets before bidding.

## What I'd do next

A few directions I stopped short of:

- **Multi-drone assets with per-drone roles.** A high-value asset currently soaks multiple drones at different altitudes via saturation. But they all do "search." Real SAR would have one drone at low altitude scanning, another providing communications relay, a third on overwatch. The market could express this with per-asset role slots instead of a single capacity number.
- **Learned switching costs.** Currently learned from operator overrides only. Could also learn from market outcome: if drones that switched off task X to task Y produced finds, switching cost on X should fall.
- **Probabilistic acceptance instead of hard capacity.** Capacity caps are a blunt instrument. A softmax over top-K bids with temperature would let low-capacity assets still occasionally pull a distant drone if no better match exists.
- **Partner-swarm federation.** Two swarms running on adjacent terrain could exchange `PriorityAsset` streams directly. Their operator zones and intel pins would compose across the market boundary with no new protocol.

Each of these is a localized change. That's the payoff: the invariant held, so the system is modifiable in the places that matter instead of defensively everywhere.

---

*Written after shipping phases 5–7. Source: [github.com/rhapsodic-legacy/agent-swarm](https://github.com/rhapsodic-legacy/agent-swarm). The relevant code starts at [`backend/src/agents/priority_market.py`](../backend/src/agents/priority_market.py) (200 lines) and [`backend/src/agents/coordinator.py`](../backend/src/agents/coordinator.py)'s `_run_priority_market`. If you read those and the wind-hazard composition in `coordinator.update`, you have the whole story in ~10 minutes.*
