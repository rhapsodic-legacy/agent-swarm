"""Day/night cycle for the drone swarm simulation.

Controls sun position, lighting colours, and sensor effectiveness.
The cycle starts at dawn (time_of_day = 0.25) so the simulation begins
at sunrise.
"""

from __future__ import annotations

import math


class DayCycle:
    """Manages a continuous day/night cycle with lighting and gameplay effects.

    Time of day is a float in [0, 1):
        0.00 = midnight
        0.25 = dawn
        0.50 = noon
        0.75 = dusk

    One full cycle takes ``day_length`` simulated seconds (default 300 s).
    """

    # Phase boundary thresholds (fractions of the day)
    _DAWN_START: float = 0.20
    _DAWN_END: float = 0.30
    _DUSK_START: float = 0.70
    _DUSK_END: float = 0.80

    def __init__(self, day_length: float = 300.0) -> None:
        self._day_length = day_length
        self._elapsed: float = 0.0
        self._time_of_day: float = 0.25  # start at dawn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, elapsed: float) -> None:
        """Update cycle state from the simulation's total elapsed seconds."""
        self._elapsed = elapsed
        # Offset so elapsed=0 maps to dawn (0.25)
        self._time_of_day = ((elapsed / self._day_length) + 0.25) % 1.0

    def get_time_of_day(self) -> float:
        """Return current time of day in [0, 1). 0.25=dawn, 0.5=noon, 0.75=dusk."""
        return self._time_of_day

    def get_sun_intensity(self) -> float:
        """Sun intensity: 0.0 at night, peaks at 1.0 at noon.

        Computed as max(0, sin(time_of_day * 2 * pi)).
        """
        return max(0.0, math.sin(self._time_of_day * 2.0 * math.pi))

    def get_sun_color(self) -> tuple[float, float, float]:
        """RGB colour of sunlight (each channel 0-1).

        - Noon:      warm white  (1.0, 0.98, 0.92)
        - Dawn/Dusk: orange      (1.0, 0.55, 0.20)
        - Night:     cool blue   (0.15, 0.15, 0.35)
        """
        tod = self._time_of_day
        intensity = self.get_sun_intensity()

        # Define reference colours
        noon_r, noon_g, noon_b = 1.0, 0.98, 0.92
        warm_r, warm_g, warm_b = 1.0, 0.55, 0.20
        night_r, night_g, night_b = 0.15, 0.15, 0.35

        if intensity <= 0.0:
            # Full night
            return (night_r, night_g, night_b)

        # Blend factor: 0 at horizon (dawn/dusk), 1 at noon
        # Use intensity itself as a proxy — it's 0 at horizon, 1 at zenith
        blend = intensity

        # Determine if we're in the dawn/dusk zone for extra warmth
        in_transition = self._is_in_transition(tod)

        if in_transition:
            # Lerp from warm → noon as blend increases
            t = min(blend * 2.0, 1.0)  # faster ramp through the orange zone
            r = _lerp(warm_r, noon_r, t)
            g = _lerp(warm_g, noon_g, t)
            b = _lerp(warm_b, noon_b, t)
        else:
            # Lerp from night → noon
            r = _lerp(night_r, noon_r, blend)
            g = _lerp(night_g, noon_g, blend)
            b = _lerp(night_b, noon_b, blend)

        return (round(r, 4), round(g, 4), round(b, 4))

    def get_ambient_intensity(self) -> float:
        """Ambient light level: 0.15 at deepest night, 0.6 at noon."""
        return 0.15 + 0.45 * self.get_sun_intensity()

    def get_sensor_effectiveness(self) -> float:
        """Sensor effectiveness: 0.3 at night, 1.0 at full daylight."""
        return 0.3 + 0.7 * self.get_sun_intensity()

    def serialize(self) -> dict:
        """Serialize for WebSocket broadcast."""
        r, g, b = self.get_sun_color()
        return {
            "time_of_day": round(self._time_of_day, 4),
            "sun_intensity": round(self.get_sun_intensity(), 4),
            "sun_color": [round(r, 4), round(g, 4), round(b, 4)],
            "sensor_effectiveness": round(self.get_sensor_effectiveness(), 4),
            "phase": self._get_phase(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_phase(self) -> str:
        """Return a human-readable phase name."""
        tod = self._time_of_day
        if self._DAWN_START <= tod < self._DAWN_END:
            return "dawn"
        if self._DAWN_END <= tod < self._DUSK_START:
            return "day"
        if self._DUSK_START <= tod < self._DUSK_END:
            return "dusk"
        return "night"

    def _is_in_transition(self, tod: float) -> bool:
        """True when the sun is near the horizon (dawn or dusk bands)."""
        return (self._DAWN_START <= tod < self._DAWN_END) or (
            self._DUSK_START <= tod < self._DUSK_END
        )


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between *a* and *b* by *t* in [0, 1]."""
    return a + (b - a) * t
