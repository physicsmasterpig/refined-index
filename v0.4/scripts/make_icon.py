cd v0.4 && ./bin/iref_cache_build.sh --census m208-m412cd v0.4 && ./bin/iref_cache_build.sh --census m208-m412cd v0.4 && ./bin/iref_cache_build.sh --census m208-m412#!/usr/bin/env python3
"""
Generate ManifoldIndex app icon — SVG-based for crisp rendering.

Design: Poincaré disk with {3,∞} ideal triangle tiling.
All geodesics use SVG circular-arc commands → perfectly smooth at any size.

Usage:
    python scripts/make_icon.py
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"
ICNS_PATH = OUT_DIR / "ManifoldIndex.icns"

# SVG canvas: work in a 512×512 coordinate system, centred at (256, 256)
W = 512
CX, CY = W / 2, W / 2
DISK_R = 210  # radius of the Poincaré disk in SVG units


# ── Hyperbolic geometry ──────────────────────────────────────────

def _geodesic_circle(p1: complex, p2: complex):
    """Return (centre, radius) of the Euclidean circle carrying the geodesic.

    If the geodesic passes through the origin (straight line), return None.
    """
    x1, y1 = p1.real, p1.imag
    x2, y2 = p2.real, p2.imag
    cross = x1 * y2 - x2 * y1
    if abs(cross) < 1e-10:
        return None  # straight line through origin

    # Circle through p1, p2, and inv(p1) w.r.t. the unit circle
    inv1 = 1.0 / np.conj(p1)
    ax, ay = x1, y1
    bx, by = x2, y2
    cx, cy = inv1.real, inv1.imag

    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        return None

    ux = ((ax**2+ay**2)*(by-cy) + (bx**2+by**2)*(cy-ay) + (cx**2+cy**2)*(ay-by)) / D
    uy = ((ax**2+ay**2)*(cx-bx) + (bx**2+by**2)*(ax-cx) + (cx**2+cy**2)*(bx-ax)) / D
    R = math.hypot(ax - ux, ay - uy)
    return (complex(ux, uy), R)


def _svg_arc(p1: complex, p2: complex, disk_r: float,
             cx: float, cy: float) -> str | None:
    """Return an SVG <path d="..."> for a geodesic arc clipped to the disk.

    Coordinates are in SVG space (origin top-left).
    """
    # Scale from unit-disk coords to SVG coords
    sx1 = cx + p1.real * disk_r
    sy1 = cy - p1.imag * disk_r  # SVG y is flipped
    sx2 = cx + p2.real * disk_r
    sy2 = cy - p2.imag * disk_r

    circ = _geodesic_circle(p1, p2)
    if circ is None:
        # Straight line through centre
        return f'M {sx1:.2f} {sy1:.2f} L {sx2:.2f} {sy2:.2f}'

    centre, R = circ
    svg_R = R * disk_r  # scale to SVG units
    # SVG arc: determine sweep direction
    # Cross product to decide large-arc and sweep flags
    dx1 = p1 - centre
    dx2 = p2 - centre
    cross = dx1.real * dx2.imag - dx1.imag * dx2.real
    sweep = 1 if cross < 0 else 0  # flipped because SVG y is inverted
    # Always use the small arc (large_arc = 0)
    return f'M {sx1:.2f} {sy1:.2f} A {svg_R:.2f} {svg_R:.2f} 0 0 {sweep} {sx2:.2f} {sy2:.2f}'


def _reflect(z: complex, p1: complex, p2: complex) -> complex:
    """Reflect z in the geodesic through p1, p2."""
    circ = _geodesic_circle(p1, p2)
    if circ is None:
        # Reflection in a line through origin
        # Direction of the line
        d = p2 - p1
        d /= abs(d)
        return d * np.conj(z / d)

    centre, R = circ
    w = z - centre
    if abs(w) < 1e-15:
        return z
    return R**2 / np.conj(w) + centre


def _build_tiling(depth: int = 4):
    """Build {3,∞} ideal triangle tiling of the Poincaré disk."""
    # Start with one ideal triangle: vertices on the unit circle
    v0 = np.exp(1j * math.pi / 2)
    v1 = np.exp(1j * (math.pi / 2 + 2 * math.pi / 3))
    v2 = np.exp(1j * (math.pi / 2 + 4 * math.pi / 3))
    triangles = [(v0, v1, v2)]
    seen = set()
    seen.add(_tri_key(v0, v1, v2))

    for _ in range(depth):
        new_tris = []
        for tri in triangles:
            for i in range(3):
                j, k = (i+1) % 3, (i+2) % 3
                refl = _reflect(tri[k], tri[i], tri[j])
                if abs(refl) > 0.998:
                    continue
                key = _tri_key(tri[i], tri[j], refl)
                if key not in seen:
                    seen.add(key)
                    new_tris.append((tri[i], tri[j], refl))
        triangles.extend(new_tris)
    return triangles


def _tri_key(a, b, c):
    centre = (a + b + c) / 3
    return (round(centre.real, 3), round(centre.imag, 3))


def _collect_edges(triangles):
    """De-duplicate edges from triangle list."""
    edges = {}
    for tri in triangles:
        for i in range(3):
            j = (i + 1) % 3
            p1, p2 = tri[i], tri[j]
            key = tuple(sorted([
                (round(p1.real, 3), round(p1.imag, 3)),
                (round(p2.real, 3), round(p2.imag, 3)),
            ]))
            if key not in edges:
                # Shrink ideal vertices inward slightly
                q1 = p1 * 0.995 if abs(p1) > 0.99 else p1
                q2 = p2 * 0.995 if abs(p2) > 0.99 else p2
                edges[key] = (q1, q2)
    return list(edges.values())


# ── SVG generation ───────────────────────────────────────────────

def _build_svg() -> str:
    triangles = _build_tiling(depth=4)
    edges = _collect_edges(triangles)

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'viewBox="0 0 {W} {W}" width="{W}" height="{W}">')

    # ── Defs: gradients and filters ──────────────────────────────
    parts.append('<defs>')

    # Background radial gradient
    parts.append('''
      <radialGradient id="bg" cx="50%" cy="50%" r="70%">
        <stop offset="0%"   stop-color="#161d45"/>
        <stop offset="100%" stop-color="#0a0e28"/>
      </radialGradient>
    ''')

    # Disk radial gradient
    parts.append('''
      <radialGradient id="diskGrad" cx="50%" cy="50%" r="50%">
        <stop offset="0%"   stop-color="#1a2255"/>
        <stop offset="85%"  stop-color="#0e1338"/>
        <stop offset="100%" stop-color="#0b0f30"/>
      </radialGradient>
    ''')

    # Glow filter
    parts.append('''
      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="6" result="blur"/>
        <feComposite in="SourceGraphic" in2="blur" operator="over"/>
      </filter>
    ''')

    # Outer ring glow
    parts.append('''
      <filter id="ringGlow" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="8" result="blur"/>
        <feMerge>
          <feMergeNode in="blur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    ''')

    # Text shadow
    parts.append('''
      <filter id="textShadow" x="-10%" y="-10%" width="120%" height="120%">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge>
          <feMergeNode in="blur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    ''')

    # Clip to disk
    parts.append(f'  <clipPath id="diskClip">'
                 f'<circle cx="{CX}" cy="{CY}" r="{DISK_R}"/>'
                 f'</clipPath>')

    parts.append('</defs>')

    # ── Background ───────────────────────────────────────────────
    parts.append(f'<rect width="{W}" height="{W}" fill="url(#bg)"/>')

    # ── Disk fill ────────────────────────────────────────────────
    parts.append(f'<circle cx="{CX}" cy="{CY}" r="{DISK_R}" '
                 f'fill="url(#diskGrad)"/>')

    # ── Geodesic edges (clipped to disk) ─────────────────────────
    parts.append(f'<g clip-path="url(#diskClip)" filter="url(#glow)">')

    for q1, q2 in edges:
        mid = (q1 + q2) / 2
        r_mid = abs(mid)
        # Opacity and width fade with distance from centre
        opacity = max(0.06, 0.75 * (1 - r_mid ** 2.0))
        lw = max(0.3, 2.2 * (1 - r_mid ** 1.5))

        arc_d = _svg_arc(q1, q2, DISK_R, CX, CY)
        if arc_d is None:
            continue

        # Blue with brightness varying by depth
        brightness = 0.5 + 0.4 * (1 - r_mid)
        r_c = int(70 + 100 * (1 - r_mid))
        g_c = int(140 + 80 * (1 - r_mid))
        b_c = 255
        parts.append(
            f'  <path d="{arc_d}" fill="none" '
            f'stroke="rgb({r_c},{g_c},{b_c})" '
            f'stroke-width="{lw:.2f}" '
            f'stroke-opacity="{opacity:.3f}" '
            f'stroke-linecap="round"/>'
        )

    parts.append('</g>')

    # ── Disk boundary ring ───────────────────────────────────────
    parts.append(f'<circle cx="{CX}" cy="{CY}" r="{DISK_R}" '
                 f'fill="none" stroke="#5080e0" stroke-width="2.5" '
                 f'stroke-opacity="0.85" filter="url(#ringGlow)"/>')

    # ── "M" + subscript "I" ──────────────────────────────────────
    parts.append(f'''
      <text x="{CX}" y="{CY + 8}" text-anchor="middle"
            dominant-baseline="central"
            font-family="-apple-system, Helvetica Neue, Arial, sans-serif"
            font-weight="700" font-size="128"
            fill="#dce2ff" fill-opacity="0.90"
            filter="url(#textShadow)">M</text>
    ''')
    parts.append(f'''
      <text x="{CX + 72}" y="{CY + 45}" text-anchor="middle"
            dominant-baseline="central"
            font-family="-apple-system, Helvetica Neue, Arial, sans-serif"
            font-weight="700" font-size="62"
            fill="#70a8ff" fill-opacity="0.75"
            filter="url(#textShadow)">I</text>
    ''')

    parts.append('</svg>')
    return '\n'.join(parts)


# ── Build pipeline ───────────────────────────────────────────────

def generate_icon():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Write SVG
    svg_path = OUT_DIR / "ManifoldIndex.svg"
    svg_content = _build_svg()
    svg_path.write_text(svg_content)
    print(f"✓ SVG saved: {svg_path}")

    # 2. Render to 1024×1024 PNG via cairosvg
    import cairosvg
    master_png = OUT_DIR / "ManifoldIndex_1024.png"
    cairosvg.svg2png(
        bytestring=svg_content.encode(),
        write_to=str(master_png),
        output_width=1024,
        output_height=1024,
    )
    print(f"✓ PNG saved: {master_png}")

    # 3. Add macOS-style rounded corners
    from PIL import Image, ImageDraw
    img = Image.open(master_png).convert("RGBA")
    mask = Image.new("L", (1024, 1024), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, 1024, 1024], radius=225, fill=255)
    img.putalpha(mask)
    img.save(master_png)

    # 4. Build .icns
    with tempfile.TemporaryDirectory() as tmpdir:
        iconset = Path(tmpdir) / "ManifoldIndex.iconset"
        iconset.mkdir()
        for s in [16, 32, 64, 128, 256, 512, 1024]:
            resized = img.resize((s, s), Image.LANCZOS)
            resized.save(iconset / f"icon_{s}x{s}.png")
            if s >= 32:
                resized.save(iconset / f"icon_{s//2}x{s//2}@2x.png")

        result = subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(ICNS_PATH)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"iconutil stderr: {result.stderr}")
            return master_png

    print(f"✓ Icon saved: {ICNS_PATH}")
    return ICNS_PATH


if __name__ == "__main__":
    generate_icon()
