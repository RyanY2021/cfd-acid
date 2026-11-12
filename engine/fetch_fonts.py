"""Fetch static TTFs from the Google Fonts CSS API (OFL fonts)."""
import re, urllib.request, pathlib, sys

OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "fonts"
OUT.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/4.0"  # old UA -> CSS API serves truetype

FAMILIES = {
    "BigShouldersStencil": "Big+Shoulders+Stencil:wght@700;900",
    "BigShoulders": "Big+Shoulders:wght@400;700;900",
    "Inter": "Inter:wght@400;500;700;900",
    "InterTight": "Inter+Tight:wght@500;700;900",
    "JetBrainsMono": "JetBrains+Mono:wght@400;700;800",
    "MartianMono": "Martian+Mono:wght@400;700",
    "Silkscreen": "Silkscreen:wght@400;700",
    "Doto": "Doto:wght@700;900",
    "Tektur": "Tektur:wght@500;700;900",
    "SpaceGrotesk": "Space+Grotesk:wght@500;700",
    "Anton": "Anton",
    "ArchivoExp": "Archivo:wdth,wght@125,800;125,900",
    "ChakraPetch": "Chakra+Petch:wght@500;700",
    "Cinzel": "Cinzel:wght@500;700;900",
    "CormorantSC": "Cormorant+SC:wght@500;700",
}

for name, q in FAMILIES.items():
    url = f"https://fonts.googleapis.com/css2?family={q}"
    try:
        css = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA})).read().decode()
    except Exception as e:
        print("FAIL", name, e); continue
    blocks = re.findall(r"@font-face\s*{(.*?)}", css, re.S)
    for b in blocks:
        w = re.search(r"font-weight:\s*(\d+)", b).group(1)
        st = re.search(r"font-stretch:\s*([\d.]+)%", b)
        src = re.search(r"url\((.*?)\)", b).group(1)
        ext = src.rsplit(".", 1)[-1][:5]
        fn = OUT / f"{name}-{w}{'-w'+st.group(1) if st else ''}.{ext}"
        if not fn.exists():
            fn.write_bytes(urllib.request.urlopen(src).read())
        print(fn.name, fn.stat().st_size)
