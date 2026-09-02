from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class BlockState:
    name: str
    properties: tuple[tuple[str,str]]
    
    @classmethod
    def create(
        cls,
        name: str,
        properties: dict[str, str] | None = None,
    ) -> "BlockState":
        if properties is None:
            properties = {}

        return cls(
            name=name,
            properties=tuple(sorted(properties.items()))
        )

    def get(self, key: str) -> str | None:
        for property_name, value in self.properties:
            if property_name == key:
                return value

        return None