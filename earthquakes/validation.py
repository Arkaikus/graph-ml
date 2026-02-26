"""
Pydantic models for config and data validation.
QuakesConfig: latitude/longitude bounds (used from env + CLI overrides).
"""

from pydantic import BaseModel, Field, field_validator, model_validator


def _parse_tuple(value: str | list | tuple) -> tuple[float, float]:
    """Parse 'min,max' string or 2-element list/tuple to (float, float)."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("Must have exactly 2 elements (min, max)")
        return (float(value[0]), float(value[1]))
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("Must be 'min,max' with two numbers")
        return (float(parts[0]), float(parts[1]))
    raise TypeError("Expected str, list, or tuple of 2 numbers")


class CoordinateBounds(BaseModel):
    """Latitude or longitude bounds as (min, max)."""

    min_val: float = Field(..., alias="min")
    max_val: float = Field(..., alias="max")

    model_config = {"populate_by_name": True}

    @property
    def tuple(self) -> tuple[float, float]:
        return (self.min_val, self.max_val)

    @classmethod
    def from_string(cls, s: str) -> "CoordinateBounds":
        a, b = _parse_tuple(s)
        return cls(min_val=min(a, b), max_val=max(a, b))


class QuakesConfig(BaseModel):
    """Validated run config: latitude and longitude bounds only."""

    latitude: tuple[float, float] = Field(
        ...,
        description="(min_latitude, max_latitude)",
    )
    longitude: tuple[float, float] = Field(
        ...,
        description="(min_longitude, max_longitude)",
    )

    @field_validator("latitude", "longitude", mode="before")
    @classmethod
    def coerce_bounds(cls, v):
        if v is None or (isinstance(v, str) and not v.strip()):
            return (0.0, 0.0)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            a, b = float(v[0]), float(v[1])
            return (min(a, b), max(a, b))
        if isinstance(v, str):
            return _parse_tuple(v)
        return v


class USGSQueryParams(BaseModel):
    """Validated USGS FDSN query parameters."""

    min_latitude: float = Field(..., ge=-90, le=90)
    max_latitude: float = Field(..., ge=-90, le=90)
    min_longitude: float = Field(..., ge=-180, le=180)
    max_longitude: float = Field(..., ge=-180, le=180)
    format: str = Field(default="csv")
    starttime: str = Field(default="1975-01-01")
    orderby: str = Field(default="time-asc")
    eventtype: str = Field(default="earthquake")

    @model_validator(mode="after")
    def bounds_ordered(self):
        if self.max_latitude < self.min_latitude:
            raise ValueError("max_latitude must be >= min_latitude")
        if self.max_longitude < self.min_longitude:
            raise ValueError("max_longitude must be >= min_longitude")
        return self


class GridConfig(BaseModel):
    """Validated grid bounds and cell size (for Grid construction)."""

    min_latitude: float = Field(..., ge=-90, le=90)
    max_latitude: float = Field(..., ge=-90, le=90)
    min_longitude: float = Field(..., ge=-180, le=180)
    max_longitude: float = Field(..., ge=-180, le=180)
    distance_km: float = Field(..., gt=0, description="Grid cell size in km")

    @model_validator(mode="after")
    def bounds_ordered(self):
        if self.max_latitude < self.min_latitude:
            raise ValueError("max_latitude must be >= min_latitude")
        if self.max_longitude < self.min_longitude:
            raise ValueError("max_longitude must be >= min_longitude")
        return self

    @property
    def latitude(self) -> tuple[float, float]:
        return (self.min_latitude, self.max_latitude)

    @property
    def longitude(self) -> tuple[float, float]:
        return (self.min_longitude, self.max_longitude)
