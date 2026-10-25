from __future__ import annotations

from typing import TYPE_CHECKING

import graphics_math as gm

if TYPE_CHECKING:
  from shader import Shader

class ClipPlane:
  def __init__ (self, shader: Shader, a: float, b: float, c: float, d: float) -> None:
    self.shader = shader
    self.set_plane(a, b, c, d)
    self.set_color(0.5, 0.5, 0.5)

  def set_plane (self, a: float, b: float, c: float, d: float) -> None:
    self.plane = gm.vec4(a, b, c, d)

  def get_plane (self) -> gm.Vec4:
    return self.plane

  def set_color (self, r: float, g: float, b: float) -> None:
    self.shader.set_value("clip_plane_color", gm.vec4(r, g, b, 1.0))

  def update (self, view: gm.Mat4) -> None:
    self.shader.set_value("clip_plane", gm.transpose(gm.inverse(view)) @ self.plane)
