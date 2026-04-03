"""Core types for the drone swarm simulation.

All state objects are frozen dataclasses for immutability between ticks.
Coordinates: X = east, Y = up (altitude), Z = north. Terrain is in the XZ plane.
Units: meters for distance, seconds for time, percentage (0-100) for battery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
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

    type: str  # "move_to", "search_area", "return_to_base", "override", "set_priority"
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
    tick_rate: float = 20.0  # Hz


@dataclass(frozen=True)
class SimConfig:
    """Configuration for initializing a simulation."""

    terrain_size: int = 256
    terrain_seed: int = 42
    max_elevation: float = 200.0
    drone_count: int = 20
    tick_rate: float = 20.0
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
