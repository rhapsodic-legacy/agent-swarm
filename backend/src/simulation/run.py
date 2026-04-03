"""Headless simulation runner — no frontend, no WebSocket.

Runs the simulation for a fixed number of ticks and prints stats.
Useful for testing and benchmarking.

Usage:
    python -m src.simulation.run --headless
    python -m src.simulation.run --headless --ticks 1000 --drones 30
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.simulation.engine import create_world, get_coverage_pct, tick
from src.simulation.types import Command, SimConfig, Vec3


def run_headless(config: SimConfig, max_ticks: int = 1000) -> None:
    """Run simulation without visualization."""
    print("=== Drone Swarm Simulation (Headless) ===")
    print(f"Terrain: {config.terrain_size}x{config.terrain_size}, seed={config.terrain_seed}")
    print(f"Drones: {config.drone_count}, Survivors: {config.survivor_count}")
    print(f"Max ticks: {max_ticks} ({max_ticks / config.tick_rate:.0f}s simulated)")
    print()

    t0 = time.monotonic()
    world = create_world(config)
    init_time = time.monotonic() - t0
    print(f"World created in {init_time:.2f}s")

    rng = np.random.default_rng(config.terrain_seed)
    dt = 1.0 / config.tick_rate

    # Give drones spread-out search targets
    n = len(world.drones)
    w = world.terrain.width
    h = world.terrain.height
    margin = 20
    cols = max(1, int(n**0.5))
    rows = max(1, (n + cols - 1) // cols)
    dx = (w - 2 * margin) / max(1, cols)
    dz = (h - 2 * margin) / max(1, rows)

    commands: list[Command] = []
    for i in range(n):
        col = i % cols
        row = i // cols
        tx = margin + dx * (col + 0.5)
        tz = margin + dz * (row + 0.5)
        commands.append(Command(type="move_to", drone_id=i, target=Vec3(tx, 0, tz)))

    sim_start = time.monotonic()
    for t in range(max_ticks):
        world = tick(world, dt, commands if t == 0 else None, rng=rng, config=config)

        # Log every 5 simulated seconds
        if world.tick % (int(config.tick_rate) * 5) == 0:
            coverage = get_coverage_pct(world.fog_grid)
            found = sum(1 for s in world.survivors if s.discovered)
            active = sum(1 for d in world.drones if d.status.name == "ACTIVE")
            avg_bat = sum(d.battery for d in world.drones) / len(world.drones)
            print(
                f"  t={world.elapsed:6.1f}s | "
                f"Coverage: {coverage:5.1f}% | "
                f"Found: {found}/{len(world.survivors)} | "
                f"Active: {active}/{len(world.drones)} | "
                f"Avg Battery: {avg_bat:.0f}%"
            )

    sim_time = time.monotonic() - sim_start
    tps = max_ticks / sim_time

    print()
    print("=== Results ===")
    print(f"Simulated time: {world.elapsed:.1f}s")
    print(f"Wall-clock time: {sim_time:.2f}s")
    print(f"Tick rate: {tps:.0f} ticks/sec (target: {config.tick_rate})")
    print(f"Real-time capable: {'YES' if tps >= config.tick_rate else 'NO'}")
    print(f"Final coverage: {get_coverage_pct(world.fog_grid):.1f}%")

    found = sum(1 for s in world.survivors if s.discovered)
    print(f"Survivors found: {found}/{len(world.survivors)}")

    active = sum(1 for d in world.drones if d.status.name == "ACTIVE")
    failed = sum(1 for d in world.drones if d.status.name == "FAILED")
    print(f"Drones: {active} active, {failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drone Swarm Headless Simulator")
    parser.add_argument("--headless", action="store_true", help="Run headless (required)")
    parser.add_argument("--ticks", type=int, default=1000, help="Number of ticks to simulate")
    parser.add_argument("--drones", type=int, default=20, help="Number of drones")
    parser.add_argument("--survivors", type=int, default=15, help="Number of survivors")
    parser.add_argument("--terrain-size", type=int, default=256, help="Terrain grid size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = SimConfig(
        terrain_size=args.terrain_size,
        terrain_seed=args.seed,
        drone_count=args.drones,
        survivor_count=args.survivors,
    )

    run_headless(config, max_ticks=args.ticks)


if __name__ == "__main__":
    main()
