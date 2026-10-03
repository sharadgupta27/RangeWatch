"""Minimal Mapbox Vector Tile decoder for point layers (GBIF density tiles).

GBIF's map API serves observation density as MVT point features (one per tile pixel, with a
`total` record count). Only what that needs is decoded — point geometries and numeric
attributes — so no protobuf schema or third-party MVT dependency is required.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from struct import unpack

_MOVE_TO = 1


@dataclass(frozen=True, slots=True)
class TilePoint:
    x: int  # tile pixel coordinates in [0, extent)
    y: int
    properties: dict[str, float | int | str | bool]


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if b < 0x80:
            return result, i


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _fields(buf: bytes) -> Iterator[tuple[int, int | bytes]]:
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field_no, wire = key >> 3, key & 7
        value: int | bytes
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 2:
            n, i = _varint(buf, i)
            value, i = buf[i : i + n], i + n
        elif wire == 1:
            value, i = buf[i : i + 8], i + 8
        elif wire == 5:
            value, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield field_no, value


def _packed(buf: bytes) -> list[int]:
    out, i = [], 0
    while i < len(buf):
        v, i = _varint(buf, i)
        out.append(v)
    return out


def _value(buf: bytes) -> float | int | str | bool:
    for field_no, v in _fields(buf):
        if field_no == 1:
            return bytes(v).decode("utf-8")  # type: ignore[arg-type]
        if field_no == 2:
            return unpack("<f", v)[0]  # type: ignore[arg-type]
        if field_no == 3:
            return unpack("<d", v)[0]  # type: ignore[arg-type]
        if field_no in (4, 5):
            n = int(v)  # type: ignore[arg-type]
            return n - (1 << 64) if field_no == 4 and n >= 1 << 63 else n
        if field_no == 6:
            return _zigzag(int(v))  # type: ignore[arg-type]
        if field_no == 7:
            return bool(v)
    raise ValueError("Empty MVT value")


def decode_points(data: bytes, layer: str | None = None) -> tuple[int, list[TilePoint]]:
    """(extent, points) of the first matching layer; points outside the tile are dropped."""
    for field_no, raw in _fields(data):
        if field_no != 3:
            continue
        name, extent = "", 4096
        keys: list[str] = []
        values: list[float | int | str | bool] = []
        features: list[bytes] = []
        for f, v in _fields(raw):  # type: ignore[arg-type]
            if f == 1:
                name = bytes(v).decode("utf-8")  # type: ignore[arg-type]
            elif f == 2:
                features.append(v)  # type: ignore[arg-type]
            elif f == 3:
                keys.append(bytes(v).decode("utf-8"))  # type: ignore[arg-type]
            elif f == 4:
                values.append(_value(v))  # type: ignore[arg-type]
            elif f == 5:
                extent = int(v)  # type: ignore[arg-type]
        if layer is not None and name != layer:
            continue
        points: list[TilePoint] = []
        for feat in features:
            tags: list[int] = []
            geom: list[int] = []
            for f, v in _fields(feat):
                if f == 2:
                    tags = _packed(v)  # type: ignore[arg-type]
                elif f == 4:
                    geom = _packed(v)  # type: ignore[arg-type]
            props = {keys[tags[k]]: values[tags[k + 1]] for k in range(0, len(tags) - 1, 2)}
            x = y = 0
            i = 0
            while i < len(geom):
                cmd, count = geom[i] & 7, geom[i] >> 3
                i += 1
                if cmd != _MOVE_TO:  # not a point feature
                    break
                for _ in range(count):
                    x += _zigzag(geom[i])
                    y += _zigzag(geom[i + 1])
                    i += 2
                    if 0 <= x < extent and 0 <= y < extent:
                        points.append(TilePoint(x, y, props))
        return extent, points
    return 4096, []
