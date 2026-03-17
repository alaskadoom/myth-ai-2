"""
MYTH AI 2.0 — Textile Colorway & Repeat Studio
Upload image → Extract colours → Edit layers → Seamless repeat → Export
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.cluster import KMeans
import io, math, colorsys, copy, base64

st.set_page_config(
    page_title="MYTH AI 2.0",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# CSS — Dark, clean, MYTH AI inspired
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --bg:      #0D0D14;
  --panel:   #13131E;
  --card:    #1A1A28;
  --border:  #262638;
  --accent:  #4F6EF7;
  --accent2: #7B5EA7;
  --gold:    #C9A84C;
  --text:    #E8E4DC;
  --muted:   #6B6B88;
  --green:   #3DDC84;
  --red:     #FF5E5E;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg) !important;
  color: var(--text);
}
#MainMenu, footer, header { visibility: hidden; }

/* Top bar */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 24px; background: var(--panel);
  border-bottom: 1px solid var(--border);
  margin-bottom: 0;
}
.logo {
  font-family: 'Syne', sans-serif; font-size: 1.3rem;
  font-weight: 800; color: #fff; letter-spacing: -0.5px;
}
.logo span { color: var(--accent); }
.step-pills { display: flex; gap: 6px; }
.pill {
  padding: 5px 14px; border-radius: 20px; font-size: .75rem;
  font-weight: 600; cursor: pointer; transition: all .15s;
  border: 1px solid var(--border); background: var(--card);
  color: var(--muted); letter-spacing: .3px;
}
.pill.active {
  background: var(--accent); border-color: var(--accent);
  color: #fff;
}

/* Canvas area */
.canvas-wrap {
  background: #080810;
  border-right: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  min-height: 80vh; position: relative;
}

/* Right panel */
.panel-title {
  font-family: 'Syne', sans-serif; font-size: .72rem;
  font-weight: 700; letter-spacing: 1.8px; text-transform: uppercase;
  color: var(--muted); padding: 14px 16px 10px;
  border-bottom: 1px solid var(--border);
}

/* Color swatch grid */
.swatch-grid { display: flex; flex-wrap: wrap; gap: 8px; padding: 14px 16px; }
.swatch-item {
  width: 52px; text-align: center; cursor: pointer;
}
.swatch-box {
  width: 52px; height: 52px; border-radius: 8px;
  border: 2px solid transparent; transition: all .15s;
  cursor: pointer; position: relative;
}
.swatch-box:hover { border-color: #fff; transform: scale(1.05); }
.swatch-box.active { border-color: var(--accent); }
.swatch-hex {
  font-size: .58rem; color: var(--muted);
  margin-top: 4px; font-family: monospace;
}

/* Layer rows */
.layer-item {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 16px; border-bottom: 1px solid var(--border);
  transition: background .1s; cursor: pointer;
}
.layer-item:hover { background: var(--card); }
.layer-item.active { background: rgba(79,110,247,.08); }
.layer-thumb {
  width: 36px; height: 36px; border-radius: 6px;
  border: 1px solid var(--border); flex-shrink: 0;
  overflow: hidden; background: #080810;
}
.layer-dot {
  width: 20px; height: 20px; border-radius: 50%;
  border: 2px solid rgba(255,255,255,.15); flex-shrink: 0;
}
.layer-name { font-size: .8rem; color: var(--text); flex: 1; }
.layer-hex { font-size: .68rem; color: var(--muted); font-family: monospace; }
.eye-btn {
  font-size: .75rem; color: var(--muted); cursor: pointer;
  width: 22px; text-align: center; flex-shrink: 0;
}

/* Variant cards */
.variant-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 8px; cursor: pointer;
  transition: all .15s; text-align: center;
}
.variant-card:hover { border-color: var(--accent); }
.variant-card.active { border-color: var(--accent); background: rgba(79,110,247,.08); }
.variant-label { font-size: .7rem; color: var(--muted); margin-top: 6px; }
.color-row { display: flex; gap: 3px; margin-top: 5px; justify-content: center; }
.color-chip {
  height: 10px; border-radius: 2px; flex: 1;
}

/* Buttons */
.stButton > button {
  background: var(--accent) !important; color: #fff !important;
  font-weight: 600 !important; font-family: 'DM Sans', sans-serif !important;
  border: none !important; border-radius: 8px !important;
  padding: 10px 20px !important; width: 100% !important;
  font-size: .88rem !important; letter-spacing: .2px !important;
  transition: all .15s !important;
}
.stButton > button:hover {
  background: #3a5de0 !important;
  transform: translateY(-1px) !important;
}

/* Secondary button */
.btn-secondary > button {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}
.btn-secondary > button:hover {
  background: var(--border) !important; transform: none !important;
}

/* Repeat mode tabs */
.repeat-tabs { display: flex; gap: 4px; padding: 10px 16px; }
.rtab {
  flex: 1; padding: 7px 4px; text-align: center; border-radius: 6px;
  font-size: .72rem; font-weight: 600; cursor: pointer;
  border: 1px solid var(--border); background: var(--card);
  color: var(--muted); transition: all .15s;
}
.rtab.active {
  background: var(--accent); border-color: var(--accent); color: #fff;
}

/* Slider styling */
[data-testid="stSlider"] > div > div { background: var(--border) !important; }

/* Upload area */
[data-testid="stFileUploader"] {
  border: 2px dashed var(--border) !important;
  border-radius: 12px !important; background: var(--card) !important;
  padding: 20px !important;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; }

/* Info boxes */
.info-box {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
}
.info-label { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }
.info-value { font-family: 'Syne', sans-serif; font-size: 1.2rem; color: var(--text); font-weight: 700; margin-top: 3px; }

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; border-radius: 7px !important; font-size: .85rem !important;
}

/* Hide streamlit elements */
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: var(--panel) !important;
  border-bottom: 1px solid var(--border) !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--muted) !important;
  font-family: 'DM Sans', sans-serif !important; font-size: .82rem !important;
  font-weight: 500 !important; padding: 10px 20px !important;
  border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text) !important;
  border-bottom: 2px solid var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] { background: var(--bg) !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# BACKEND — All heavy lifting here, UI stays clean
# ═══════════════════════════════════════════════════════════════

def hex_to_rgb(hex_c):
    h = hex_c.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

def pil_to_b64(img, fmt="PNG"):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def pil_to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()

def luminance(r, g, b):
    return 0.2126*(r/255) + 0.7152*(g/255) + 0.0722*(b/255)


@st.cache_data(show_spinner=False, max_entries=20)
def extract_colors(img_bytes, n_colors=7):
    """
    K-means color extraction from image.
    Returns list of (R,G,B) tuples, sorted by frequency.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_small = img.resize((200, 200), Image.LANCZOS)
    pixels = np.array(img_small).reshape(-1, 3).astype(np.float32)
    
    km = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=100)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_
    
    # Sort by frequency (most used colour first)
    counts = np.bincount(labels)
    order  = np.argsort(-counts)
    sorted_colors = [tuple(int(c) for c in centers[i]) for i in order]
    return sorted_colors


def recolor_image(img_bytes, original_colors, new_colors, n_colors):
    """
    Replace each pixel's nearest original color with new_colors mapping.
    This is how MYTH AI recolors layers.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = img.size
    arr  = np.array(img, dtype=np.float32)  # H,W,3

    # Build pixel-to-cluster map
    centers = np.array(original_colors[:n_colors], dtype=np.float32)  # n,3
    pixels  = arr.reshape(-1, 3)
    diffs   = pixels[:, None, :] - centers[None, :, :]  # M,n,3
    dists   = np.sum(diffs**2, axis=2)                  # M,n
    assign  = np.argmin(dists, axis=1)                  # M

    # Build new image using new_colors
    new_centers = np.array([hex_to_rgb(h) for h in new_colors[:n_colors]], dtype=np.float32)
    recolored   = new_centers[assign].reshape(H, W, 3).astype(np.uint8)
    return Image.fromarray(recolored, "RGB")


def get_layer_mask(img_bytes, original_colors, layer_idx, n_colors):
    """
    Get mask for a single color layer (white where this colour appears).
    """
    img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr    = np.array(img, dtype=np.float32)
    H, W   = arr.shape[:2]
    centers= np.array(original_colors[:n_colors], dtype=np.float32)
    pixels = arr.reshape(-1, 3)
    diffs  = pixels[:, None, :] - centers[None, :, :]
    dists  = np.sum(diffs**2, axis=2)
    assign = np.argmin(dists, axis=1).reshape(H, W)
    mask   = (assign == layer_idx).astype(np.uint8) * 255
    return Image.fromarray(mask, "L")


def make_layer_thumbnail(img_bytes, original_colors, layer_idx, n_colors, color_hex, size=36):
    """Generate a small thumbnail showing just one colour layer."""
    mask   = get_layer_mask(img_bytes, original_colors, layer_idx, n_colors)
    mask   = mask.resize((size, size), Image.LANCZOS)
    r, g, b = hex_to_rgb(color_hex)
    coloured = Image.new("RGB", (size, size), (r, g, b))
    bg       = Image.new("RGB", (size, size), (13, 13, 20))
    bg.paste(coloured, mask=mask)
    return bg


def generate_colorways(original_colors, n_variants=4):
    """
    Auto-generate N colorway variants by shifting hue.
    Returns list of lists of hex strings.
    """
    variants = []
    hue_shifts = np.linspace(30, 330, n_variants)
    for hs in hue_shifts:
        variant = []
        for r, g, b in original_colors:
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            h2 = (h + hs/360) % 1.0
            r2, g2, b2 = colorsys.hsv_to_rgb(h2, s, v)
            variant.append(rgb_to_hex(r2*255, g2*255, b2*255))
        variants.append(variant)
    return variants


# ── Seamless & Repeat ──────────────────────────────────────────

def make_seamless(img):
    """Sinusoidal feather blend for perfect seamless tile."""
    W, H   = img.size
    arr    = np.array(img.convert("RGBA"), dtype=np.float32)
    bw     = max(20, W // 10)
    arr_s  = np.roll(np.roll(arr, H//2, axis=0), W//2, axis=1)
    mask   = np.zeros((H, W), dtype=np.float32)
    for i in range(bw):
        a = math.sin((i / bw) * math.pi / 2)
        hy, hx = H//2, W//2
        if 0 <= hy-bw+i < H: mask[hy-bw+i, :] = np.maximum(mask[hy-bw+i, :], a)
        if 0 <= hy+bw-i-1 < H: mask[hy+bw-i-1, :] = np.maximum(mask[hy+bw-i-1, :], a)
        if 0 <= hx-bw+i < W: mask[:, hx-bw+i] = np.maximum(mask[:, hx-bw+i], a)
        if 0 <= hx+bw-i-1 < W: mask[:, hx+bw-i-1] = np.maximum(mask[:, hx+bw-i-1], a)
    mimg   = Image.fromarray((mask*255).astype(np.uint8), "L")
    mimg   = mimg.filter(ImageFilter.GaussianBlur(bw//4))
    ms     = np.array(mimg, dtype=np.float32) / 255.0
    m4     = np.stack([ms]*4, axis=2)
    b      = np.clip(arr*(1-m4) + arr_s*m4, 0, 255).astype(np.uint8)
    return Image.fromarray(b, "RGBA")

def repeat_block(tile, rows, cols):
    t = make_seamless(tile.convert("RGBA"))
    W, H = t.size
    c = Image.new("RGB", (W*cols, H*rows), (255,255,255))
    for r in range(rows):
        for col in range(cols):
            c.paste(t.convert("RGB"), (col*W, r*H))
    return c

def repeat_mirror(tile, rows, cols):
    W, H = tile.size
    th = tile.transpose(Image.FLIP_LEFT_RIGHT)
    tv = tile.transpose(Image.FLIP_TOP_BOTTOM)
    thv= th.transpose(Image.FLIP_TOP_BOTTOM)
    st = Image.new("RGB", (W*2, H*2))
    st.paste(tile.convert("RGB"), (0,0))
    st.paste(th.convert("RGB"),   (W,0))
    st.paste(tv.convert("RGB"),   (0,H))
    st.paste(thv.convert("RGB"),  (W,H))
    SW, SH = st.size
    nc = math.ceil(cols/2); nr = math.ceil(rows/2)
    c  = Image.new("RGB", (SW*nc, SH*nr))
    for r in range(nr):
        for col in range(nc):
            c.paste(st, (col*SW, r*SH))
    return c.crop((0, 0, W*cols, H*rows))

def repeat_half_drop(tile, rows, cols):
    t = make_seamless(tile.convert("RGBA"))
    W, H = t.size
    c = Image.new("RGB", (W*cols, H*rows+H//2), (255,255,255))
    for col in range(cols):
        offset = H//2 if col%2 else 0
        for r in range(rows+1):
            y = r*H - H//2 + offset
            if -H < y < H*rows+H:
                c.paste(t.convert("RGB"), (col*W, y))
    return c.crop((0, 0, W*cols, H*rows))

def repeat_brick(tile, rows, cols):
    t = make_seamless(tile.convert("RGBA"))
    W, H = t.size
    c = Image.new("RGB", (W*cols+W//2, H*rows), (255,255,255))
    for r in range(rows):
        offset = W//2 if r%2 else 0
        for col in range(cols+1):
            x = col*W - W//2 + offset
            if -W < x < W*cols+W:
                c.paste(t.convert("RGB"), (x, r*H))
    return c.crop((0, 0, W*cols, H*rows))

def apply_repeat(img, mode, rows=3, cols=4):
    if mode == "Mirror":    return repeat_mirror(img, rows, cols)
    if mode == "Half Drop": return repeat_half_drop(img, rows, cols)
    if mode == "Brick":     return repeat_brick(img, rows, cols)
    return repeat_block(img, rows, cols)


# ── Sample designs (base64 generated programmatically) ────────

def make_sample_design(n=0):
    """Generate a built-in sample floral/geometric design."""
    samples = [
        # Sample 0: Pink floral
        {"bg":(220,50,100), "colors":[(255,200,180),(180,80,40),(255,255,200),(100,60,20),(255,150,170)]},
        # Sample 1: Blue geometric
        {"bg":(30,60,120),  "colors":[(200,220,255),(255,200,50),(100,150,200),(255,255,255),(50,100,180)]},
        # Sample 2: Green paisley
        {"bg":(30,100,60),  "colors":[(180,220,150),(255,200,100),(100,180,80),(255,255,200),(60,140,40)]},
        # Sample 3: Gold luxury
        {"bg":(20,15,10),   "colors":[(200,160,60),(255,220,100),(150,110,30),(255,255,220),(80,60,20)]},
    ]
    s    = samples[n % len(samples)]
    sz   = 400
    img  = Image.new("RGB", (sz, sz), s["bg"])
    draw = ImageDraw.Draw(img)
    cols_list = s["colors"]
    import random as rnd
    rnd.seed(n * 7 + 42)
    # Large florals
    step = sz // 4
    for cx in range(step//2, sz, step):
        for cy in range(step//2, sz, step):
            ox, oy = rnd.randint(-20, 20), rnd.randint(-20, 20)
            cx2, cy2 = cx+ox, cy+oy
            rp = sz // 14
            for angle in range(0, 360, 60):
                rad = math.radians(angle)
                px = cx2 + int(rp*1.7*math.cos(rad))
                py = cy2 + int(rp*1.7*math.sin(rad))
                draw.ellipse([px-rp, py-rp//2, px+rp, py+rp//2], fill=cols_list[0])
            draw.ellipse([cx2-rp//2, cy2-rp//2, cx2+rp//2, cy2+rp//2], fill=cols_list[1])
            draw.ellipse([cx2-rp//4, cy2-rp//4, cx2+rp//4, cy2+rp//4], fill=cols_list[4])
    # Medium florals
    step2 = sz // 6
    for cx in range(step2//3, sz, step2):
        for cy in range(step2//2, sz, step2):
            rp2 = sz // 24
            for angle in range(0, 360, 72):
                rad = math.radians(angle)
                px = cx + int(rp2*1.5*math.cos(rad))
                py = cy + int(rp2*1.5*math.sin(rad))
                draw.ellipse([px-rp2//2, py-rp2//2, px+rp2//2, py+rp2//2], fill=cols_list[2])
            draw.ellipse([cx-rp2//3, cy-rp2//3, cx+rp2//3, cy+rp2//3], fill=cols_list[3])
    # Vines
    rnd.seed(n * 13)
    for _ in range(5):
        x, y = rnd.randint(0, sz), rnd.randint(0, sz)
        pts = [(x, y)]
        for _ in range(20):
            x = max(0, min(sz-1, x + rnd.randint(-25, 25)))
            y = max(0, min(sz-1, y + rnd.randint(-25, 25)))
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=cols_list[1], width=max(1, sz//120))
    # Dots
    for _ in range(60):
        x, y = rnd.randint(5, sz-5), rnd.randint(5, sz-5)
        r = rnd.randint(2, 5)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=cols_list[3])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "img_bytes"      : None,
        "orig_colors"    : None,   # list of (R,G,B)
        "n_colors"       : 7,
        "layer_colors"   : None,   # list of hex strings (editable)
        "layer_visible"  : None,   # list of bools
        "active_layer"   : 0,
        "layers_created" : False,
        "step"           : 1,      # 1=upload, 2=colors, 3=layers, 4=repeat
        "repeat_mode"    : "Mirror",
        "repeat_rows"    : 3,
        "repeat_cols"    : 4,
        "colorway_idx"   : -1,
        "colorways"      : None,
        "sample_idx"     : 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
S = st.session_state


# ═══════════════════════════════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════════════════════════════
step_names = ["1 · Upload", "2 · Colours", "3 · Layers", "4 · Repeat & Export"]
step_html = ""
for i, name in enumerate(step_names, 1):
    cls = "pill active" if S.step == i else "pill"
    step_html += f'<div class="{cls}">{name}</div>'

st.markdown(f"""
<div class="topbar">
  <div class="logo">MYTH<span> AI</span> 2.0</div>
  <div class="step-pills">{step_html}</div>
  <div style="font-size:.75rem;color:var(--muted);">Textile Design Studio</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════
if S.step == 1:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style='text-align:center;padding:20px 0 28px;'>
          <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                      color:#fff;margin-bottom:8px;'>Start Designing</div>
          <div style='color:var(--muted);font-size:.9rem;'>
            Upload your textile design image or pick a sample to get started
          </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload Design Image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed"
        )

        if uploaded:
            img_bytes = uploaded.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # Resize to max 600px for performance
            max_dim = 600
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = (int(img.width*ratio), int(img.height*ratio))
                img = img.resize(new_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            S.img_bytes = buf.getvalue()
            S.layers_created = False
            S.colorways = None

        st.markdown("""
        <div style='text-align:center;color:var(--muted);font-size:.8rem;
                    padding:14px 0 6px;'>— or choose a sample design —</div>
        """, unsafe_allow_html=True)

        sample_cols = st.columns(4)
        sample_labels = ["🌸 Floral", "💎 Geometric", "🍃 Paisley", "✨ Luxury"]
        for i, (col, label) in enumerate(zip(sample_cols, sample_labels)):
            with col:
                samp_bytes = make_sample_design(i)
                samp_img   = Image.open(io.BytesIO(samp_bytes))
                st.image(samp_img, use_container_width=True)
                if st.button(label, key=f"samp_{i}", use_container_width=True):
                    S.img_bytes     = samp_bytes
                    S.sample_idx    = i
                    S.layers_created= False
                    S.colorways     = None

        st.markdown("<br>", unsafe_allow_html=True)

        if S.img_bytes:
            if st.button("✅  Continue to Colour Extraction  →", use_container_width=True):
                S.step = 2
                st.rerun()


# ═══════════════════════════════════════════════════════════════
# STEP 2 — COLOUR EXTRACTION
# ═══════════════════════════════════════════════════════════════
elif S.step == 2:
    if not S.img_bytes:
        S.step = 1
        st.rerun()

    left, right = st.columns([3, 2], gap="small")

    with left:
        st.markdown("""
        <div style='background:#080810;border-right:1px solid var(--border);
                    padding:20px;min-height:85vh;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;'>
        """, unsafe_allow_html=True)
        st.image(S.img_bytes, use_container_width=True, caption="Your Design")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="panel-title">COLOUR EXTRACTION</div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='padding:14px 16px 0;'>", unsafe_allow_html=True)
        n_col = st.slider("Max Colours", 3, 22, S.n_colors, key="nc_slider")
        if n_col != S.n_colors:
            S.n_colors      = n_col
            S.layers_created= False
        st.markdown("</div>", unsafe_allow_html=True)

        # Extract colours
        with st.spinner("Extracting colours..."):
            colors_rgb = extract_colors(S.img_bytes, S.n_colors)
        S.orig_colors = colors_rgb

        # Init layer_colors from extracted if not edited yet
        if not S.layers_created:
            S.layer_colors  = [rgb_to_hex(*c) for c in colors_rgb[:S.n_colors]]
            S.layer_visible = [True] * S.n_colors

        # Draw colour grid exactly like MYTH AI
        n = S.n_colors
        cols_count = 3
        st.markdown(f"""
        <div class="panel-title">COLORS ({n}/{n})</div>
        """, unsafe_allow_html=True)

        swatch_html = '<div class="swatch-grid">'
        for i, hex_c in enumerate(S.layer_colors[:n]):
            swatch_html += f"""
            <div class="swatch-item">
              <div class="swatch-box" style="background:{hex_c};"></div>
              <div class="swatch-hex">{hex_c.upper()}</div>
            </div>"""
        swatch_html += "</div>"
        st.markdown(swatch_html, unsafe_allow_html=True)

        st.markdown("<div style='padding:10px 16px;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:.75rem;color:var(--muted);margin-bottom:10px;'>
          Model: <span style='color:var(--text);'>Color Unmix (K-Means)</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='color:var(--muted);font-size:.78rem;margin-bottom:14px;'>
          Colours extracted from your design. Adjust the slider to control
          how many colours to separate into layers.
        </div>""", unsafe_allow_html=True)

        if st.button("🎨  Create Layers", use_container_width=True):
            S.layers_created = True
            S.step = 3
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
            if st.button("← Back", key="back2", use_container_width=True):
                S.step = 1
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# STEP 3 — LAYERS (core editing step)
# ═══════════════════════════════════════════════════════════════
elif S.step == 3:
    if not S.img_bytes or not S.orig_colors:
        S.step = 2
        st.rerun()

    n = S.n_colors

    # ── Compute recoloured preview ────────────────
    @st.cache_data(show_spinner=False, max_entries=30)
    def get_preview(img_bytes, orig_tup, new_colors_tup, n):
        return recolor_image(img_bytes, list(orig_tup), list(new_colors_tup), n)

    orig_tup   = tuple(tuple(c) for c in S.orig_colors[:n])
    new_c_tup  = tuple(S.layer_colors[:n])
    preview_img = get_preview(S.img_bytes, orig_tup, new_c_tup, n)

    # ── Layout: Canvas | Layer Panel ─────────────
    canvas_col, panel_col = st.columns([3, 2], gap="small")

    with canvas_col:
        st.markdown("""
        <div style='background:#080810;padding:20px;min-height:85vh;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;'>
        """, unsafe_allow_html=True)

        # Show preview or individual layer
        if S.active_layer == -1 or S.active_layer >= n:
            # Full composite
            # Apply visibility mask
            vis_preview = preview_img.copy()
            # Greyed out hidden layers
            has_hidden = any(not v for v in S.layer_visible[:n])
            if has_hidden:
                arr = np.array(preview_img.convert("RGB"), dtype=np.float32)
                orig_arr = np.array(Image.open(io.BytesIO(S.img_bytes)).resize(preview_img.size), dtype=np.float32)
                centers  = np.array(S.orig_colors[:n], dtype=np.float32)
                pix      = arr.reshape(-1,3)
                opix     = np.array(Image.open(io.BytesIO(S.img_bytes)).convert("RGB").resize(preview_img.size)).reshape(-1,3).astype(np.float32)
                diffs    = opix[:,None,:] - centers[None,:,:]
                assign   = np.argmin(np.sum(diffs**2, axis=2), axis=1)
                result   = arr.reshape(-1,3).copy()
                for i in range(n):
                    if not S.layer_visible[i]:
                        result[assign==i] = [20,20,32]
                vis_preview = Image.fromarray(result.reshape(arr.shape).astype(np.uint8), "RGB")
            st.image(vis_preview, use_container_width=True)
        else:
            # Show isolated layer
            mask = get_layer_mask(S.img_bytes, S.orig_colors[:n], S.active_layer, n)
            mask = mask.resize(preview_img.size, Image.LANCZOS)
            r,g,b = hex_to_rgb(S.layer_colors[S.active_layer])
            layer_img = Image.new("RGB", preview_img.size, (20,20,32))
            coloured  = Image.new("RGB", preview_img.size, (r,g,b))
            layer_img.paste(coloured, mask=mask)
            st.image(layer_img, use_container_width=True,
                     caption=f"Layer {S.active_layer+1} · {S.layer_colors[S.active_layer]}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Bottom: view toggle
        v1, v2 = st.columns(2)
        with v1:
            if st.button("👁  View All Layers", key="view_all", use_container_width=True):
                S.active_layer = -1
                st.rerun()
        with v2:
            if st.button("🔁  Go to Repeat →", key="to_repeat", use_container_width=True):
                S.step = 4
                st.rerun()

    with panel_col:
        # ── Tabs: Layers | Colorways ──────────────
        tab_layers, tab_colorways = st.tabs(["  Layers  ", "  Colorways  "])

        with tab_layers:
            st.markdown(f"""
            <div class="panel-title">LAYERS: {n}</div>
            """, unsafe_allow_html=True)

            for i in range(n):
                hex_c = S.layer_colors[i]
                r2, g2, b2 = hex_to_rgb(hex_c)
                is_active = (S.active_layer == i)
                vis_icon  = "👁" if S.layer_visible[i] else "🙈"

                col_a, col_b, col_c = st.columns([1, 4, 1])
                with col_a:
                    # Visibility toggle
                    if st.button(vis_icon, key=f"vis_{i}", help="Toggle visibility"):
                        S.layer_visible[i] = not S.layer_visible[i]
                        st.rerun()
                with col_b:
                    active_style = "background:rgba(79,110,247,.1);border-radius:6px;" if is_active else ""
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;gap:8px;
                                padding:6px 8px;{active_style}cursor:pointer;'>
                      <div style='width:22px;height:22px;border-radius:50%;
                                  background:{hex_c};border:2px solid rgba(255,255,255,.2);
                                  flex-shrink:0;'></div>
                      <div>
                        <div style='font-size:.8rem;color:{"#fff" if is_active else "var(--text)"};
                                    font-weight:{"600" if is_active else "400"};'>
                          Layer {i+1}
                        </div>
                        <div style='font-size:.65rem;color:var(--muted);font-family:monospace;'>
                          {hex_c.upper()}
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Select", key=f"sel_{i}", use_container_width=True,
                                  help=f"Select layer {i+1}"):
                        S.active_layer = i
                        st.rerun()
                with col_c:
                    st.markdown(f"""
                    <div style='width:28px;height:28px;border-radius:6px;
                                background:{hex_c};border:1px solid rgba(255,255,255,.2);
                                margin-top:6px;'></div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='border-bottom:1px solid var(--border);'></div>",
                            unsafe_allow_html=True)

            # ── Colour editor for active layer ────
            if 0 <= S.active_layer < n:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background:var(--card);border:1px solid var(--border);
                            border-radius:10px;padding:14px;margin:0 4px;'>
                  <div style='font-family:Syne,sans-serif;font-size:.75rem;
                              color:var(--muted);text-transform:uppercase;
                              letter-spacing:1.5px;margin-bottom:12px;'>
                    Edit Layer {S.active_layer+1}
                  </div>
                """, unsafe_allow_html=True)

                cur_hex = S.layer_colors[S.active_layer]

                # HEX input
                new_hex = st.text_input(
                    "HEX Code",
                    value=cur_hex,
                    key=f"hex_edit_{S.active_layer}",
                    placeholder="#RRGGBB"
                )
                if new_hex != cur_hex and new_hex.startswith("#") and len(new_hex) == 7:
                    try:
                        hex_to_rgb(new_hex)
                        S.layer_colors[S.active_layer] = new_hex.upper()
                        st.rerun()
                    except:
                        pass

                # Quick colour presets
                st.markdown("""
                <div style='font-size:.72rem;color:var(--muted);margin:10px 0 6px;'>
                  Quick Colours
                </div>""", unsafe_allow_html=True)

                quick_colors = [
                    "#C0392B","#E74C3C","#FF6B9D","#FF9A3C","#F1C40F","#C8A96E",
                    "#27AE60","#1ABC9C","#3498DB","#2E4057","#8E44AD","#2C3E50",
                    "#FFFFFF","#F5F0E1","#BDC3C7","#7F8C8D","#1C1C1C","#800000",
                ]
                qc_html = '<div style="display:flex;flex-wrap:wrap;gap:5px;">'
                for qc in quick_colors:
                    qc_html += f"""
                    <div style='width:26px;height:26px;border-radius:4px;
                                background:{qc};cursor:pointer;
                                border:2px solid {"var(--accent)" if qc.upper()==cur_hex.upper() else "transparent"};'
                         title='{qc}'></div>"""
                qc_html += "</div>"
                st.markdown(qc_html, unsafe_allow_html=True)

                # Preset click via selectbox
                sel = st.selectbox(
                    "Or select colour",
                    ["— pick —"] + quick_colors,
                    key=f"qsel_{S.active_layer}",
                    label_visibility="collapsed"
                )
                if sel != "— pick —":
                    S.layer_colors[S.active_layer] = sel.upper()
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

        with tab_colorways:
            st.markdown('<div class="panel-title">AUTO COLORWAYS</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style='padding:10px 16px;color:var(--muted);font-size:.78rem;'>
              Auto-generate colour variants of your design. Each variant
              shifts the hue while keeping your layer structure.
            </div>""", unsafe_allow_html=True)

            n_variants = st.slider("Variants", 2, 8, 4, key="n_var_s")

            if st.button("⚡ Generate Colorways", use_container_width=True, key="gen_cw"):
                with st.spinner("Generating colorways..."):
                    S.colorways = generate_colorways(S.orig_colors[:n], n_variants)

            if S.colorways:
                for vi, cway in enumerate(S.colorways):
                    hex_strip = "".join(
                        f'<div style="flex:1;height:12px;background:{c};"></div>'
                        for c in cway[:n]
                    )
                    active_cw = (S.colorway_idx == vi)
                    border = "var(--accent)" if active_cw else "var(--border)"
                    bg     = "rgba(79,110,247,.08)" if active_cw else "var(--card)"

                    st.markdown(f"""
                    <div style='background:{bg};border:1px solid {border};
                                border-radius:8px;padding:8px;margin-bottom:8px;'>
                      <div style='font-size:.72rem;color:var(--muted);margin-bottom:5px;'>
                        Variant {vi+1}
                      </div>
                      <div style='display:flex;gap:3px;border-radius:4px;overflow:hidden;'>
                        {hex_strip}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"Apply Variant {vi+1}", key=f"apply_cw_{vi}",
                                  use_container_width=True):
                        S.layer_colors  = list(cway[:n])
                        S.colorway_idx  = vi
                        S.active_layer  = -1
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
            if st.button("← Back to Colours", key="back3", use_container_width=True):
                S.step = 2
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# STEP 4 — SEAMLESS REPEAT & EXPORT
# ═══════════════════════════════════════════════════════════════
elif S.step == 4:
    if not S.img_bytes or not S.orig_colors:
        S.step = 2
        st.rerun()

    n = S.n_colors

    # Get current recoloured tile
    @st.cache_data(show_spinner=False, max_entries=20)
    def get_tile(img_bytes, orig_tup, new_colors_tup, n):
        return recolor_image(img_bytes, list(orig_tup), list(new_colors_tup), n)

    orig_tup  = tuple(tuple(c) for c in S.orig_colors[:n])
    new_c_tup = tuple(S.layer_colors[:n])
    tile_img  = get_tile(S.img_bytes, orig_tup, new_c_tup, n)

    # ── Layout ───────────────────────────────────
    canvas_col, panel_col = st.columns([3, 2], gap="small")

    with panel_col:
        st.markdown('<div class="panel-title">PATTERN SETTINGS</div>', unsafe_allow_html=True)
        st.markdown("<div style='padding:14px 16px;'>", unsafe_allow_html=True)

        # Repeat mode selector
        st.markdown("""<div style='font-size:.72rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:1px;
                        margin-bottom:8px;'>Repeat Mode</div>""",
                    unsafe_allow_html=True)

        modes = ["Mirror", "Half Drop", "Brick", "Block"]
        mode_icons = {"Mirror":"⊞","Half Drop":"⊟","Brick":"▦","Block":"▤"}
        mode_descs = {
            "Mirror"    : "Mirrors tile — zero seam lines, perfect for organic patterns",
            "Half Drop" : "Odd columns drop 50% — best for florals & botanicals",
            "Brick"     : "Odd rows shift 50% — great for geometric & checks",
            "Block"     : "Standard grid repeat — all tiles identical",
        }

        for mode in modes:
            is_active = (S.repeat_mode == mode)
            bg    = "rgba(79,110,247,.15)" if is_active else "var(--card)"
            border= "var(--accent)" if is_active else "var(--border)"
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {border};
                        border-radius:8px;padding:10px 12px;margin-bottom:6px;'>
              <div style='font-size:.82rem;color:{"#fff" if is_active else "var(--text)"};
                          font-weight:{"600" if is_active else "400"};'>
                {mode_icons[mode]}  {mode}
              </div>
              <div style='font-size:.7rem;color:var(--muted);margin-top:2px;'>
                {mode_descs[mode]}
              </div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"{'✓ Selected' if is_active else 'Select'} {mode}",
                          key=f"rm_{mode}", use_container_width=True):
                S.repeat_mode = mode
                st.rerun()

        st.markdown("---", unsafe_allow_html=True)

        # Grid size
        st.markdown("""<div style='font-size:.72rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:1px;
                        margin-bottom:8px;'>Grid Size</div>""",
                    unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            rows = st.number_input("Rows", 1, 8, S.repeat_rows, key="rep_rows")
            S.repeat_rows = rows
        with r2:
            cols = st.number_input("Cols", 1, 8, S.repeat_cols, key="rep_cols")
            S.repeat_cols = cols

        st.markdown("---")

        # Export
        st.markdown("""<div style='font-size:.72rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:1px;
                        margin-bottom:12px;'>Export</div>""",
                    unsafe_allow_html=True)

        with st.spinner("Building repeat..."):
            repeat_img = apply_repeat(tile_img, S.repeat_mode, S.repeat_rows, S.repeat_cols)

        tile_dl   = pil_to_bytes(tile_img)
        repeat_dl = pil_to_bytes(repeat_img)

        st.download_button(
            "⬇️  Download Tile (PNG)",
            tile_dl,
            "MYTH_AI_tile.png", "image/png",
            use_container_width=True, key="dl_tile"
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.download_button(
            "⬇️  Download Repeat (PNG)",
            repeat_dl,
            f"MYTH_AI_{S.repeat_mode.replace(' ','_')}_repeat.png", "image/png",
            use_container_width=True, key="dl_repeat"
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Colour palette export info
        pal_html = "<div style='display:flex;gap:4px;margin-bottom:6px;border-radius:6px;overflow:hidden;'>"
        for hex_c in S.layer_colors[:n]:
            pal_html += f"<div style='flex:1;height:24px;background:{hex_c};'></div>"
        pal_html += "</div>"
        st.markdown(pal_html, unsafe_allow_html=True)

        hex_codes = " | ".join(S.layer_colors[:n])
        st.text_area("Colour HEX Codes", hex_codes, height=80, key="hex_export",
                      label_visibility="collapsed")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='padding:0 16px 14px;'>", unsafe_allow_html=True)
        st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
        if st.button("← Back to Layers", key="back4", use_container_width=True):
            S.step = 3
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with canvas_col:
        view_tab1, view_tab2 = st.tabs(["  🖼  Tile  ", "  🔁  Repeat Preview  "])

        with view_tab1:
            st.markdown("""
            <div style='background:#080810;padding:24px;text-align:center;'>
            """, unsafe_allow_html=True)
            st.image(tile_img, use_container_width=True,
                     caption=f"Design Tile · {tile_img.size[0]}×{tile_img.size[1]}px")
            st.markdown("</div>", unsafe_allow_html=True)

        with view_tab2:
            st.markdown("""
            <div style='background:#080810;padding:24px;text-align:center;'>
            """, unsafe_allow_html=True)
            with st.spinner(f"Rendering {S.repeat_mode} repeat ({rows}×{cols})..."):
                prev_repeat = apply_repeat(tile_img, S.repeat_mode, rows, cols)
            st.image(prev_repeat, use_container_width=True,
                     caption=f"{S.repeat_mode} Repeat · {rows}×{cols} grid")
            st.markdown("</div>", unsafe_allow_html=True)

            # All 4 modes side by side
            st.markdown("""
            <div style='font-family:Syne,sans-serif;font-size:.72rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:1.5px;
                        padding:14px 0 10px;'>All Repeat Modes</div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            for col_el, mode in zip([m1,m2,m3,m4], ["Mirror","Half Drop","Brick","Block"]):
                with col_el:
                    with st.spinner(f"{mode}..."):
                        sm = apply_repeat(tile_img, mode, 2, 2)
                    border = "2px solid var(--accent)" if mode == S.repeat_mode else "1px solid var(--border)"
                    st.markdown(f"""
                    <div style='border:{border};border-radius:8px;
                                overflow:hidden;margin-bottom:4px;'>
                    """, unsafe_allow_html=True)
                    st.image(sm, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.caption(mode)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#3A3A50;font-size:.72rem;
            padding:20px 0 8px;border-top:1px solid #1A1A28;margin-top:24px;'>
  MYTH AI 2.0 · Textile Colorway & Repeat Studio ·
  Upload → Extract → Edit → Repeat → Export
</div>
""", unsafe_allow_html=True)
