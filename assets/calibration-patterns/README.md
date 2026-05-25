# Calibration Patterns

Reference art for the Cobblestone calibration system.

## Aesthetic decision (2026-05-24)

Ted picked the muted naturalistic palette over saturated/vivid encodings. Saturated colors read as "cartoony" and break the illusion that the projection belongs in the room. The muted palette reads as a real cobblestone street.

## Files

- **`reference_deep_joint.jpg`** — original reference image (Creative Concrete CH600-RA101, deep-joint cobblestone). The brief.
- **`cobble_v4_realistic.png`** — production idle pattern. Neutral grey palette anchored on the reference (~RGB 135,130,128), running-bond rows, deep dark mortar joints (~RGB 38), weathered surface texture (vertical light gradient, dark pitting specks, mineral-flake highlights).
- **`cobble_v5_realistic_calibration_subtle.png`** — same look as v4, but each stone is uniquely color-coded at **18% tint strength**. Audience reads it as natural stone mineralization; camera + decoder reads it as a calibration target. Candidate for **continuous always-on calibration** — never needs a visible calibration "mode."
- **`generator.py`** — the Python script that produces all variants. Run with `python generator.py <out_dir>`.

## Calibration strategy

If 18% tint is enough decoding signal under real venue lighting:
- Run v5 always-on as the ambient pattern.
- System continuously calibrates from the live feed.
- No visible "calibration flash" ever needed.
- This is the dream operator experience.

If 18% isn't enough:
- Run v4 as the ambient pattern.
- Flash a stronger-tint variant briefly (1-2 sec) for initial calibration and after drift detection.
- Still ten-minute setup, just one quick visible step.

**To be determined Tuesday** when JMGO arrives and we capture the projected pattern with the D455 under real conditions.

## Pattern parameters (current)

- Resolution: 1920×1080 (will rescale to projector native)
- Layout: running bond (offset rows like a brick wall)
- Row height: ~110 px (~17 rows on a 1080p frame)
- Stone width: ~140 px (~13 stones per row)
- Total stones per frame: ~220
- Mortar gap: 12 px
- Corner radius: 18–30% of stone width (tumbled feel)
- Edge jitter: 8–18% (weathered/chipped)
- Texture per stone: vertical highlight gradient + dark pits + mineral specks
- Palette: 16 neutral-grey BGR values, all in cool-grey family with slight taupe variation

## Future variants to test

- Smaller granite-cube setts (~60 px) for higher decode density at the cost of more visual noise
- Larger flagstones (~250 px) for spaces where you want bigger calibration targets and less visual repetition
- Belgian-block elongated stones (more rectangular) for a different aesthetic
- 1664-NYC photo reference if we can find one in the TimeWalk research archive
- Animation: slow pulse / palette shift on idle, faster pulse during active calibration capture
