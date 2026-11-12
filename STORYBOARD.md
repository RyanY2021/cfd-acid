# CFD // ACID — storyboard & style bible

A 110-second motion-design piece introducing Computational Fluid Dynamics,
dressed in the "graphic realism" acid style of Bungie's *Marathon*.
1920×1080, 30 fps, 120 BPM (1 beat = 0.5 s = 15 frames, 1 bar = 2 s).
Every flow on screen is a real simulation (LBM, pseudo-spectral NS, analytic potential flow).
The aerial surf shot, the CFD-02 vehicle, the perfused hand and the crashing computer are modelled and rendered in Blender (Cycles; the smoke is a Mantaflow gas simulation).

## Style bible (distilled from the three references)

| Ref | What we take |
|---|---|
| Alpha Intro Cinematic | Black field; electric blue / acid green / red / acid yellow; segmented ring HUDs; wireframe + contour-line objects; big pixel-block glitches; repeated-word banners (`PASSPHRASE_PASSPHRASE`) with one word filled; vertical labels; micro tech text (`QR EXC FF+`); full-bleed colour frames with hard cuts |
| Reveal Cinematic | Stencil condensed logotype in lime with `+` registration marks and a micro data line; vertical acid credits |
| NuCaloric message | Off-white + hot magenta Swiss layout; giant notched numerals; crop marks, crosshair circles, barcodes, monospace serials; vertical side rail |

**Palette** — BLACK `#040406` · BLUE `#141CFF` · GREEN `#28FF3C` · LIME `#BEFF00` · YELLOW `#EBFF14` · RED `#FF1E28` · ORANGE `#FF3C14` · MAGENTA `#EC005F` · PAPER `#F5F5F2` · WHITE.
Flat colour, posterised fields, ordered dithering, no soft gradients.

**Type** — Big Shoulders Stencil (logotype) · Archivo Expanded (banners) · Inter (Swiss captions) · Tektur (numerals) · JetBrains Mono / Martian Mono (micro labels) · Silkscreen / Doto (pixel readouts).

## Timeline

| Time | Section | Content |
|---|---|---|
| 0–6 | 0 INTRO | Top-down aerial surf (Blender) already resampled into the film palette up to the land boundary; sliding `LOADING` type over the compound; survey HUD tracks breakers and tags the KELVIN HYDRODYNAMICS LABORATORY; the divider sweeps across and the whole frame becomes a field |
| 6–14 | A BOOT | Micro labels type on, segmented ring assembles, `INITIATING // FLOW_SOLVER`, buffer counter, pixel-glitch blowout |
| 14–22 | B TITLE | Giant stencil `CFD`; a lattice-Boltzmann wind starts and sheds vortices off the letters |
| 22–30 | C EVERYTHING FLOWS | Beat-cut full-bleed cards AIR / WATER / BLOOD / … with densities; molecules → continuum |
| 30–42 | D THE LAW | Navier–Stokes assembled term by term; `UNSOLVED` + Millennium Prize banner; "so we compute" |
| 42–50 | E DISCRETIZE | Blue frame; Joukowski airfoil + body-fitted O-grid drawing itself; cell counter |
| 50–58 | F SOLVE | Residual curves falling, iteration counter, ring to 100 %, `CONVERGED` |
| 58–82 | G FLOW REVEAL | Airfoil streamlines → von Kármán street → Reynolds sweep 2×2 → Kelvin–Helmholtz (magenta Swiss) → turbulence point cloud |
| 82–97 | H APPLICATIONS | 01 Aerospace · 02 Automotive (Blender CFD-02 product shot → car in its own LBM wind tunnel) · 03 Cardiovascular, graded after the Arachne faction intro (black, crimson latex drapes, pale porcelain, 2.39:1 letterbox, grain, halation, Cinzel serif): a pale mannequin hand hangs on strings, blood is pumped in beat by beat, the strings snap as it comes alive, push into the radial artery → stenosis flow as crimson topographic bands · 04 Climate |
| 97–110 | I OUTRO | "SIMULATE THE ███" — the voice jams on "the", the last word is junk code → `CFD` card (its tagline corrupted too) match-dissolves into the CRT (the Blender camera starts with the screen filling the frame 1:1) → slow pull back: it's on an old computer in near darkness, which loses hold, throws NaN errors, sparks and smokes → SIMULATION EXHAUSTED (on screen only, not voiced) |
