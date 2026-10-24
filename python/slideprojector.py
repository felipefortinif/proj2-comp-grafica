from __future__ import annotations

from typing import Any

import graphics_math as gm

from spotlight import SpotLight

class SlideProjector(SpotLight):
  """Spot que projeta um slide: alem da posicao/direcao, entrega ao shader
  slide_matrix, que leva um ponto do espaco de iluminacao (camera) para o
  clip do projetor. O frustum do projetor tem abertura 2*cutoff, de modo
  que o slide ocupa o mesmo cone da spot."""

  def __init__ (self, x: float, y: float, z: float,
                dx: float, dy: float, dz: float,
                cutoff: float = 30.0, inner: float | None = None,
                aspect: float = 1.0, near: float = 0.05, far: float = 20.0,
                space: str = "world", **values: Any) -> None:
    SpotLight.__init__(self, x, y, z, dx, dy, dz, cutoff, inner, space, **values)
    self.aspect, self.near, self.far = aspect, near, far

  def map (self, matrix: gm.Mat4) -> dict[str, Any]:
    out = SpotLight.map(self, matrix)
    p = self.get("light_position")[:3]
    d = self.get("light_direction")
    # "up" do projetor nao pode ser paralelo a direcao
    up = gm.vec3(1, 0, 0) if abs(d[0]) < 0.9 else gm.vec3(0, 0, 1)
    view = gm.look_at(p, p + d, up)   # espaco local -> olho do projetor
    proj = gm.perspective(gm.radians(2.0 * self.get_cutoff()), self.aspect, self.near, self.far)
    out["slide_matrix"] = proj @ view @ gm.inverse(matrix)   # camera -> local -> projetor -> clip
    return out
