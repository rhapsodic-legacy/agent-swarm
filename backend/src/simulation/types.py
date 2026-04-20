"""Core types for the drone swarm simulation.

All state objects are frozen dataclasses for immutability between ticks.
Coordinates: X = east, Y = up (altitude), Z = north. Terrain is in the XZ plane.
Units: meters for distance, seconds for time, percentage (0-100) for battery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import NamedTuple


class Vec3(NamedTuple):
    """3D vector / position. X=east, Y=up, Z=north."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vec3) -> Vec3:  # type: ignore[override]
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self.__mul__(scalar)

    def length(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def length_xz(self) -> float:
        """Horizontal distance (ignoring altitude)."""
        return (self.x**2 + self.z**2) ** 0.5

    def normalized(self) -> Vec3:
        ln = self.length()
        if ln < 1e-8:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / ln, self.y / ln, self.z / ln)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z


class Biome(Enum):
    WATER = 0
    BEACH = 1
    FOREST = 2
    URBAN = 3
    MOUNTAIN = 4
    SNOW = 5


class DroneStatus(Enum):
    ACTIVE = auto()
    RETURNING = auto()
    RECHARGING = auto()
    FAILED = auto()


class EventType(Enum):
    SURVIVOR_FOUND = auto()
    DRONE_FAILED = auto()
    DRONE_BATTERY_CRITICAL = auto()
    DRONE_COMMS_LOST = auto()
    DRONE_RETURNED = auto()
    ZONE_FULLY_EXPLORED = auto()
    EVIDENCE_FOUND = auto()


class EvidenceKind(StrEnum):
    """Types of search evidence a drone can discover.

    Phase 3 ships three kinds that exercise the core posterior-update
    geometries (directional cone, drift ring, high-confidence circle).
    Additional kinds (body, beacon_ping, vocal_signal, witness_report)
    follow the same contract and will plug in later.
    """

    FOOTPRINT = "footprint"
    DEBRIS = "debris"
    SIGNAL_FIRE = "signal_fire"


# Fog-of-war cell states
FOG_UNEXPLORED = 0
FOG_EXPLORED = 1
FOG_STALE = 2


@dataclass(frozen=True)
class Drone:
    """Immutable snapshot of a single drone's state."""

    id: int
    position: Vec3
    velocity: Vec3 = Vec3(0.0, 0.0, 0.0)
    heading: float = 0.0  # radians, 0 = north (+Z)
    battery: float = 100.0  # percentage 0-100
    sensor_active: bool = True
    comms_active: bool = True
    status: DroneStatus = DroneStatus.ACTIVE
    current_task: str = "idle"
    target: Vec3 | None = None
    # Speed settings (m/s)
    max_speed: float = 15.0
    # Sensor range (meters)
    sensor_range: float = 40.0
    # Communication range (meters)
    comms_range: float = 100.0


@dataclass(frozen=True)
class Survivor:
    """A person to be found during search-and-rescue."""

    id: int
    position: Vec3
    discovered: bool = False
    discovered_by: int | None = None  # drone id
    discovered_at_tick: int | None = None
    mobile: bool = False  # whether this survivor wanders
    speed: float = 0.5  # movement speed in m/s


@dataclass(frozen=True)
class Evidence:
    """A clue planted at world-gen time, discoverable by drones.

    Evidence is the bridge between the mission's ground truth (where
    survivors actually are) and the PoC prior (where the planner thinks
    they are). Each clue, when discovered, triggers a Bayesian posterior
    update that re-focuses the belief map.

    Attributes:
        id: Unique identifier within the mission.
        position: World-space location (Y is terrain elevation at plant time).
        kind: One of the EvidenceKind values (footprint / debris / signal_fire).
        confidence: 0-1 weight applied when the evidence updates the PoC.
            High for fresh signal_fire (~0.95), moderate for footprints
            (~0.6), lower for weathered debris (~0.4).
        heading: Direction (radians, 0 = +Z/north, clockwise) the evidence
            points toward. Used by directional kinds (footprint) to shape
            the update as a cone downstream of the clue. None = isotropic.
        age_hours: Estimated time since the clue was deposited. Affects
            update geometry — older footprints spread the cone wider.
        discovered, discovered_by, discovered_at_tick: Detection bookkeeping
            (mirrors the Survivor pattern).
    """

    id: int
    position: Vec3
    kind: str  # EvidenceKind value
    confidence: float = 0.7
    heading: float | None = None
    age_hours: float | None = None
    discovered: bool = False
    discovered_by: int | None = None
    discovered_at_tick: int | None = None


@dataclass(frozen=True)
class SimEvent:
    """An event that occurred during a simulation tick."""

    type: EventType
    tick: int
    drone_id: int | None = None
    survivor_id: int | None = None
    data: dict | None = None


@dataclass(frozen=True)
class Terrain:
    """Procedurally generated terrain data.

    heightmap: 2D float array, shape (height, width), values 0.0 to max_elevation.
    biome_map: 2D int array, shape (height, width), values are Biome enum ints.
    survivors: tuple of Survivor objects placed on the terrain.
    """

    width: int
    height: int
    max_elevation: float
    # These are numpy arrays but typed loosely to avoid import at type level
    heightmap: object  # np.ndarray, shape (height, width), float64
    biome_map: object  # np.ndarray, shape (height, width), int32
    survivors: tuple[Survivor, ...] = ()
    seed: int = 42


@dataclass(frozen=True)
class Command:
    """A command issued by a human or agent."""

    type: str  # "move_to", "search_area", "return_to_base", "hold_position", "override", "set_priority"
    drone_id: int | None = None
    target: Vec3 | None = None
    zone_id: str | None = None
    priority: str | None = None
    data: dict | None = None


@dataclass(frozen=True)
class WorldState:
    """Complete simulation state at a given tick. Immutable."""

    tick: int
    elapsed: float  # seconds since sim start
    terrain: Terrain
    drones: tuple[Drone, ...]
    survivors: tuple[Survivor, ...]
    # Fog-of-war grid: 0=unexplored, 1=explored, 2=stale
    # numpy array shape (terrain.height, terrain.width), int8
    fog_grid: object  # np.ndarray
    # Communication links: list of (drone_id, drone_id) tuples for drones in range
    comms_links: tuple[tuple[int, int], ...] = ()
    # Events generated this tick
    events: tuple[SimEvent, ...] = ()
    # Base position for drone resupply
    base_position: Vec3 = Vec3(0.0, 0.0, 0.0)
    # Simulation config
    tick_rate: float = 10.0  # Hz — lower CPU usage
    # Bayesian search map — Probability of Containment per cell.
    # Optional; None means search-theory-based routing is disabled.
    # See src/simulation/search_map.py for the SearchMap class.
    search_map: object = None  # SearchMap | None
    # Evidence — clues planted by the mission that drones can find and use
    # to update the PoC posterior. Empty in missions without an evidence
    # trail (Phase 3 wires lost_hiker + aircraft_crash; others stay empty).
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class SimConfig:
    """Configuration for initializing a simulation."""

    terrain_size: int = 256
    terrain_seed: int = 42
    max_elevation: float = 200.0
    drone_count: int = 20
    tick_rate: float = 10.0
    survivor_count: int = 15
    # Drone physics
    drone_max_speed: float = 15.0  # m/s
    drone_sensor_range: float = 40.0  # meters
    drone_comms_range: float = 100.0  # meters
    drone_battery_drain_rate: float = 0.05  # % per second at cruise speed
    drone_battery_critical: float = 15.0  # % threshold
    # Failure probabilities (per tick)
    sensor_failure_prob: float = 0.00005
    comms_failure_prob: float = 0.00003
    # Fog-of-war
    fog_stale_ticks: int = 200  # ticks before explored cell becomes stale
    # Flight altitude
    drone_cruise_altitude: float = 50.0  # meters above terrain
    # Detection realism — occlusion and visibility toggles
    detection_requires_los: bool = True  # terrain line-of-sight check
    canopy_occlusion: float = 0.7  # 0-1, how much forest canopy blocks detection
    urban_occlusion: float = 0.5  # 0-1, how much buildings block detection
    weather_visibility: float = 1.0  # 0-1, 1.0=clear, 0.3=heavy fog/rain
    night_penalty: float = 0.4  # detection multiplier at night (0-1)
    transponder_range: float = 200.0  # range at which transponder-equipped survivors are always detected
    transponder_ratio: float = 0.0  # fraction of survivors with transponders (0 = none)


@dataclass(frozen=True)
class ChunkedWorldConfig:
    """Configuration for the chunked world system.

    Defines the overall world dimensions, chunk granularity, seed, and survivor
    density.  Used alongside ``SimConfig`` (which holds drone / physics params).
    """

    world_size: int = 10240  # total world size in meters (10km x 10km)
    chunk_size: int = 1024  # size of each chunk in meters (1km x 1km)
    seed: int = 42
    survivor_density: float = 0.00008  # survivors per m² (~80 per chunk)
