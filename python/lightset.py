from __future__ import annotations

from typing import TYPE_CHECKING

from light import Light

if TYPE_CHECKING:
  from state import State
  from uniformbuffer import UniformBlock

class LightSet(Light):
  """Agrupa varias luzes num unico bloco "global" (o Shader aceita uma so
  luz). Cada luz escreve seus valores com um prefixo proprio: o prefixo
  troca o "light_" inicial do nome (light_position -> light2_position), de
  modo que o shader declara um conjunto de campos por luz. Valores que nao
  comecam com "light_" (ex.: slide_matrix) passam com o nome original."""

  def __init__ (self, lights: dict[str, Light]) -> None:
    Light.__init__(self, "world")
    self.lights = lights   # prefixo -> luz, ex.: {"light_": spot, "light2_": lamp}

  def map (self, matrix):
    return {}

  def write_fields (self, block: UniformBlock, st: State, lighting_space: str) -> None:
    for prefix, light in self.lights.items():
      matrix = light.get_mapping_matrix(st, lighting_space)
      values = light.get_values()
      values.update(light.map(matrix))
      for name, value in values.items():
        field = prefix + name[len("light_"):] if name.startswith("light_") else name
        if block.has_field(field):
          block.set(field, value)
