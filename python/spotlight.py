from __future__ import annotations

import math
from typing import Any

import graphics_math as gm

from light import Light


class SpotLight(Light):
    """Spot light: a point light restricted to a cone around a direction.

    The position is translated by coordinate mappings; the direction ignores
    translation and is normalized. The cone is given by two half-angles, in
    degrees: full intensity inside `inner`, fading smoothly to zero at
    `cutoff`. The shader receives the cosines of both angles."""

    def __init__ (self, x: float, y: float, z: float,
                  dx: float, dy: float, dz: float,
                  cutoff: float = 30.0, inner: float | None = None,
                  space: str = "world", **values: Any) -> None:
      Light.__init__(self, space, **values)
      self.set_position(x, y, z)
      self.set_direction(dx, dy, dz)
      self.set_cutoff(cutoff, inner)

    def set_position (self, x: float, y: float, z: float) -> None:
      self.set("light_position", gm.vec4(x,y,z,1.0))

    def set_direction (self, x: float, y: float, z: float) -> None:
      self.set("light_direction", gm.vec3(x,y,z))

    def set_cutoff (self, cutoff: float, inner: float | None = None) -> None:
      """Sets the cone half-angles (degrees); `inner` defaults to 80% of
      `cutoff` and must be smaller than it (the shader's smoothstep needs
      two distinct edges)."""
      if inner is None:
        inner = 0.8 * cutoff
      if not 0.0 <= inner < cutoff < 90.0:
        raise ValueError("SpotLight needs 0 <= inner < cutoff < 90 degrees")
      self.cutoff = cutoff
      self.inner = inner
      self.set("light_cos_cutoff", math.cos(math.radians(self.cutoff)))
      self.set("light_cos_inner", math.cos(math.radians(self.inner)))

    def get_cutoff (self) -> float:
      return self.cutoff

    def map (self, matrix: gm.Mat4) -> dict[str, Any]:
      position = self.get("light_position")
      d = self.get("light_direction")
      direction = (matrix @ gm.vec4(d[0], d[1], d[2], 0.0))[:3]
      return {"light_position": matrix @ position,
              "light_direction": gm.normalize(direction)}
