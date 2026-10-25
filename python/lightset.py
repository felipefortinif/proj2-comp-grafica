from __future__ import annotations

from typing import TYPE_CHECKING

from light import Light

if TYPE_CHECKING:
  from state import State
  from uniformbuffer import UniformBlock

class LightSet(Light):
  def __init__ (self, lights: dict[str, Light]) -> None:
    Light.__init__(self, "world")
    self.lights = lights

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
