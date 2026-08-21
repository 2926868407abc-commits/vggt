from pathlib import Path
from PIL import Image, ImageDraw

root = Path("/mnt/data/wangqq/vggt/data/textures/dtd/images")
classes = ["stained", "cracked", "marbled", "blotchy", "fibrous", "woven", "matted", "banded", "stratified"]

paths = []
for c in classes:
    paths += sorted((root / c).glob("*.jpg"))[:6]

cols = 6
rows = (len(paths) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 160, rows * 180), "white")
draw = ImageDraw.Draw(sheet)

for i, p in enumerate(paths):
    im = Image.open(p).convert("RGB").resize((144, 144))
    x = (i % cols) * 160
    y = (i // cols) * 180
    sheet.paste(im, (x, y))
    draw.text((x, y + 148), f"{p.parent.name}/{p.name}", fill=(0, 0, 0))

out = Path("/mnt/data/wangqq/vggt/assets/natural_textures/dtd_texture_candidates.png")
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out)
print(out)
