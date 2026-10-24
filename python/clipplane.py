from __future__ import annotations

from typing import TYPE_CHECKING

import graphics_math as gm

if TYPE_CHECKING:
  from shader import Shader

class ClipPlane:
  """Plano de corte ax+by+cz+d=0 dado em coordenadas de mundo; mantem o
  lado em que ax+by+cz+d >= 0. Emula gl_ClipDistance (sem equivalente em
  WGSL) via discard por fragmento. Como o shader ilumina no espaco da
  camera, update(view) leva o plano para esse espaco a cada quadro: planos
  se transformam pela inversa transposta da matriz que leva os pontos.
  "clip_plane" e "clip_plane_color" sao campos do bloco "global" (ver
  Shader.set_value)."""

  def __init__ (self, shader: Shader, a: float, b: float, c: float, d: float) -> None:
    self.shader = shader
    self.set_plane(a, b, c, d)
    self.set_color(0.5, 0.5, 0.5)

  def set_plane (self, a: float, b: float, c: float, d: float) -> None:
    """Sets the plane equation ax+by+cz+d=0, in world space."""
    self.plane = gm.vec4(a, b, c, d)

  def get_plane (self) -> gm.Vec4:
    return self.plane

  def set_color (self, r: float, g: float, b: float) -> None:
    """Cor da secao de corte (o interior do objeto cortado)."""
    self.shader.set_value("clip_plane_color", gm.vec4(r, g, b, 1.0))

  def update (self, view: gm.Mat4) -> None:
    """Escreve o plano no espaco da camera; chamar a cada quadro, antes de renderizar."""
    self.shader.set_value("clip_plane", gm.transpose(gm.inverse(view)) @ self.plane)
