"""
╔═══════════════════════════════════════════════════════════════╗
║   MYTH AI 2.0 — Professional Textile Design Studio           ║
║   22-Layer System · Colorway Editor · 4 Repeat Modes         ║
║   Seamless Engine · Demand Intelligence · Export Ready        ║
╚═══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageEnhance, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random, math, io, colorsys
from itertools import combinations
import copy

st.set_page_config(
    page_title="MYTH AI 2.0 — Textile Design Studio",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=Space+Grotesk:wght@300;400;500;600&display=swap');
:root {
  --bg:       #0B0B14;
  --surface:  #13131F;
  --card:     #1A1A2E;
  --border:   #252540;
  --gold:     #C8A96E;
  --gold2:    #E8C98E;
  --text:     #E2D9C8;
  --muted:    #6B6B8A;
  --green:    #3DD68C;
  --amber:    #F5A623;
  --red:      #E05252;
  --blue:     #5B8DEE;
}
html,body,[class*="css"]{
  font-family:'Space Grotesk',sans-serif;
  background:var(--bg);
  color:var(--text);
}
#MainMenu,footer,header{visibility:hidden}

/* ── Hero ─────────────────────────────────────── */
.hero{
  background:linear-gradient(135deg,#0B0B14 0%,#151528 40%,#0B1420 100%);
  border:1px solid var(--border);
  border-radius:14px;
  padding:28px 36px 24px;
  margin-bottom:22px;
  position:relative;
  overflow:hidden;
}
.hero::after{
  content:'';
  position:absolute;
  bottom:-80px;right:-80px;
  width:280px;height:280px;
  background:radial-gradient(circle,rgba(200,169,110,.08) 0%,transparent 70%);
  border-radius:50%;
}
.hero-eyebrow{
  font-size:.72rem;letter-spacing:2.5px;text-transform:uppercase;
  color:var(--gold);margin-bottom:10px;
}
.hero-title{
  font-family:'Cormorant Garamond',serif;
  font-size:2.6rem;font-weight:700;color:var(--gold2);
  line-height:1.1;margin:0 0 8px;
}
.hero-desc{font-size:.88rem;color:var(--muted);line-height:1.6;max-width:600px;}
.hero-badges{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;}
.badge{
  background:rgba(200,169,110,.1);border:1px solid rgba(200,169,110,.25);
  color:var(--gold);padding:3px 12px;border-radius:20px;
  font-size:.7rem;letter-spacing:1px;text-transform:uppercase;
}

/* ── Cards ────────────────────────────────────── */
.card{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:18px 20px;margin-bottom:14px;
}
.card-title{
  font-family:'Cormorant Garamond',serif;
  font-size:1rem;color:var(--gold);margin-bottom:14px;
  display:flex;align-items:center;gap:8px;letter-spacing:.3px;
}
.card-title-sm{
  font-size:.8rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:1.2px;margin-bottom:10px;
}

/* ── Layer rows ───────────────────────────────── */
.layer-row{
  display:flex;align-items:center;gap:6px;
  background:#0B0B14;border:1px solid var(--border);
  border-radius:8px;padding:7px 10px;margin-bottom:6px;
  transition:border-color .15s;
}
.layer-row:hover{border-color:var(--gold);}
.layer-dot{
  width:14px;height:14px;border-radius:50%;
  border:2px solid rgba(255,255,255,.2);flex-shrink:0;
}
.layer-name{font-size:.78rem;color:var(--text);flex:1;}
.layer-cat{
  font-size:.65rem;color:var(--muted);
  background:rgba(255,255,255,.05);
  padding:2px 6px;border-radius:4px;
}

/* ── Demand badges ────────────────────────────── */
.demand-high{background:#0d2b1d;border:1px solid var(--green);color:var(--green);
  padding:5px 16px;border-radius:20px;font-weight:600;display:inline-block;}
.demand-medium{background:#2b200d;border:1px solid var(--amber);color:var(--amber);
  padding:5px 16px;border-radius:20px;font-weight:600;display:inline-block;}
.demand-low{background:#2b0d0d;border:1px solid var(--red);color:var(--red);
  padding:5px 16px;border-radius:20px;font-weight:600;display:inline-block;}

/* ── Repeat mode buttons ──────────────────────── */
.repeat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px;}
.repeat-btn{
  background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:10px 8px;text-align:center;cursor:pointer;
  transition:all .15s;
}
.repeat-btn.active{background:rgba(200,169,110,.12);border-color:var(--gold);color:var(--gold);}
.repeat-btn-label{font-size:.75rem;margin-top:4px;color:var(--muted);}

/* ── Stats ────────────────────────────────────── */
.stat{
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:14px;text-align:center;
}
.stat-val{font-family:'Cormorant Garamond',serif;font-size:1.6rem;color:var(--gold);}
.stat-lbl{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;}

/* ── Confidence bars ──────────────────────────── */
.cbar-wrap{margin:6px 0;}
.cbar-lbl{font-size:.78rem;color:var(--muted);margin-bottom:3px;
  display:flex;justify-content:space-between;}
.cbar-bg{background:#13131F;border-radius:3px;height:8px;}
.cbar-fill{height:8px;border-radius:3px;}

/* ── Buttons ──────────────────────────────────── */
.stButton>button{
  background:linear-gradient(135deg,#C8A96E,#A07030)!important;
  color:#0B0B14!important;font-weight:600!important;
  font-family:'Space Grotesk',sans-serif!important;
  border:none!important;border-radius:8px!important;
  padding:11px 24px!important;width:100%!important;
  transition:all .2s!important;
}
.stButton>button:hover{transform:translateY(-1px)!important;
  box-shadow:0 6px 20px rgba(200,169,110,.3)!important;}

/* ── Sidebar ──────────────────────────────────── */
[data-testid="stSidebar"]{background:var(--surface)!important;
  border-right:1px solid var(--border);}
[data-testid="stSidebar"] label{color:var(--text)!important;font-size:.82rem!important;}
[data-testid="stSidebar"] .stSlider>div>div{background:var(--border)!important;}

/* ── Select / Input ───────────────────────────── */
[data-testid="stSelectbox"]>div>div{
  background:#0B0B14!important;border:1px solid var(--border)!important;
  color:var(--text)!important;border-radius:7px!important;
}
[data-testid="stTextInput"]>div>div>input{
  background:#0B0B14!important;border:1px solid var(--border)!important;
  color:var(--text)!important;border-radius:7px!important;
}
/* ── Footer ───────────────────────────────────── */
.footer{text-align:center;color:var(--muted);font-size:.75rem;
  padding:24px 0 8px;border-top:1px solid var(--border);margin-top:32px;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & COLOR LIBRARY
# ═══════════════════════════════════════════════════════════════
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Full color library with HEX codes
COLOR_LIBRARY = {
    # Reds & Pinks
    "Crimson"    :"#DC143C","Deep Red"    :"#8B0000","Maroon"     :"#800000",
    "Rose"       :"#FF007F","Hot Pink"    :"#FF69B4","Blush"      :"#FFB6C1",
    # Oranges & Yellows
    "Saffron"    :"#FF6B35","Amber"       :"#FFBF00","Gold"       :"#FFD700",
    "Copper"     :"#B87333","Rust"        :"#B7410E","Peach"      :"#FFDAB9",
    # Greens
    "Emerald"    :"#27AE60","Forest"      :"#228B22","Sage"       :"#8FBC8F",
    "Mint"       :"#98FF98","Olive"       :"#808000","Teal"       :"#008080",
    # Blues
    "Navy"       :"#001F5B","Royal Blue"  :"#2E4057","Cobalt"     :"#0047AB",
    "Sky Blue"   :"#87CEEB","Aqua"        :"#00CED1","Powder Blue":"#B0E0E6",
    # Purples
    "Purple"     :"#800080","Violet"      :"#7F00FF","Mauve"      :"#E0B0FF",
    "Indigo"     :"#4B0082","Lavender"    :"#E6E6FA","Plum"       :"#DDA0DD",
    # Neutrals
    "Ivory"      :"#FFFFF0","Cream"       :"#FFFDD0","Beige"      :"#F5F0E1",
    "Champagne"  :"#F7E7CE","Silver"      :"#C0C0C0","Charcoal"   :"#36454F",
    "Off White"  :"#FAF0E6","Warm White"  :"#FFF8F0","Black"      :"#1C1C1C",
    # Metallics
    "Gold Foil"  :"#C9A84C","Rose Gold"   :"#B76E79","Antique Gold":"#856D4D",
    "Bronze"     :"#CD7F32","Platinum"    :"#E5E4E2","Gunmetal"   :"#2A3439",
}

# Season → curated color clusters
SEASON_COLORS = {
    "Wedding"  :["Crimson","Gold","Maroon","Ivory","Rose","Champagne","Gold Foil","Blush"],
    "Festive"  :["Saffron","Gold","Emerald","Purple","Amber","Copper","Crimson","Rust"],
    "Casual"   :["Sky Blue","Mint","Peach","Cream","Sage","Lavender","Powder Blue","Olive"],
    "Summer"   :["Aqua","Hot Pink","Mint","Ivory","Sky Blue","Peach","Rose","Saffron"],
    "Winter"   :["Navy","Maroon","Charcoal","Silver","Cobalt","Purple","Plum","Gunmetal"],
    "Monsoon"  :["Teal","Royal Blue","Emerald","Silver","Beige","Aqua","Forest","Sage"],
    "Bridal"   :["Crimson","Gold Foil","Maroon","Champagne","Rose Gold","Ivory","Copper","Blush"],
    "Luxury"   :["Gold Foil","Black","Platinum","Antique Gold","Charcoal","Rose Gold","Ivory","Gunmetal"],
}

# 22-LAYER DEFINITIONS
# Each: id, name, category, default_color, default_opacity, default_blend, visible, pattern_fn_key
LAYER_DEFS = [
    # Background layers (1-4)
    {"id":0,  "name":"Background Base",      "cat":"Background", "color":"Ivory",     "opacity":1.0,  "blend":"normal",   "visible":True,  "fn":"bg_solid"},
    {"id":1,  "name":"Fabric Texture",       "cat":"Background", "color":"Beige",     "opacity":0.6,  "blend":"multiply", "visible":True,  "fn":"bg_texture"},
    {"id":2,  "name":"Ground Wash",          "cat":"Background", "color":"Champagne", "opacity":0.35, "blend":"screen",   "visible":False, "fn":"gradient_wash"},
    # Primary Pattern layers (5-9)
    {"id":3,  "name":"Large Floral",         "cat":"Pattern",    "color":"Crimson",   "opacity":0.95, "blend":"normal",   "visible":True,  "fn":"floral_large"},
    {"id":4,  "name":"Medium Floral",        "cat":"Pattern",    "color":"Rose",      "opacity":0.80, "blend":"normal",   "visible":True,  "fn":"floral_medium"},
    {"id":5,  "name":"Floral Fill",          "cat":"Pattern",    "color":"Blush",     "opacity":0.55, "blend":"normal",   "visible":True,  "fn":"floral_small"},
    {"id":6,  "name":"Leaf & Vine",          "cat":"Pattern",    "color":"Emerald",   "opacity":0.75, "blend":"normal",   "visible":True,  "fn":"vines"},
    {"id":7,  "name":"Paisley Primary",      "cat":"Pattern",    "color":"Gold",      "opacity":0.85, "blend":"normal",   "visible":False, "fn":"paisley_large"},
    {"id":8,  "name":"Paisley Secondary",    "cat":"Pattern",    "color":"Amber",     "opacity":0.65, "blend":"normal",   "visible":False, "fn":"paisley_small"},
    # Geometric layers (10-13)
    {"id":9,  "name":"Geo Grid Primary",     "cat":"Geometric",  "color":"Gold Foil", "opacity":0.50, "blend":"normal",   "visible":False, "fn":"geo_grid"},
    {"id":10, "name":"Diamond Lattice",      "cat":"Geometric",  "color":"Copper",    "opacity":0.60, "blend":"normal",   "visible":False, "fn":"diamonds"},
    {"id":11, "name":"Ikat Bands",           "cat":"Geometric",  "color":"Navy",      "opacity":0.70, "blend":"normal",   "visible":False, "fn":"ikat_bands"},
    {"id":12, "name":"Stripe Overlay",       "cat":"Geometric",  "color":"Charcoal",  "opacity":0.20, "blend":"multiply", "visible":False, "fn":"stripes"},
    # Detail layers (14-18)
    {"id":13, "name":"Dot Scatter Large",    "cat":"Detail",     "color":"Gold",      "opacity":0.70, "blend":"normal",   "visible":True,  "fn":"dots_large"},
    {"id":14, "name":"Dot Scatter Small",    "cat":"Detail",     "color":"Copper",    "opacity":0.55, "blend":"normal",   "visible":False, "fn":"dots_small"},
    {"id":15, "name":"Star Elements",        "cat":"Detail",     "color":"Gold Foil", "opacity":0.65, "blend":"screen",   "visible":True,  "fn":"stars"},
    {"id":16, "name":"Botanical Sprigs",     "cat":"Detail",     "color":"Sage",      "opacity":0.60, "blend":"normal",   "visible":False, "fn":"sprigs"},
    {"id":17, "name":"Stipple Texture",      "cat":"Detail",     "color":"Copper",    "opacity":0.30, "blend":"multiply", "visible":False, "fn":"stipple"},
    # Accent / Frame layers (19-22)
    {"id":18, "name":"Outer Border",         "cat":"Accent",     "color":"Gold Foil", "opacity":0.90, "blend":"normal",   "visible":True,  "fn":"border_outer"},
    {"id":19, "name":"Inner Border Line",    "cat":"Accent",     "color":"Amber",     "opacity":0.70, "blend":"normal",   "visible":True,  "fn":"border_inner"},
    {"id":20, "name":"Corner Ornaments",     "cat":"Accent",     "color":"Gold",      "opacity":0.80, "blend":"normal",   "visible":True,  "fn":"corner_ornaments"},
    {"id":21, "name":"Shimmer Highlight",    "cat":"Accent",     "color":"Platinum",  "opacity":0.40, "blend":"screen",   "visible":True,  "fn":"shimmer"},
]

BLEND_MODES = ["normal","multiply","screen","overlay","soft_light","hard_light"]

PATTERN_PRESETS = {
    "Bridal Silk"    :{"visible_layers":[0,1,3,4,5,6,13,15,18,19,20,21],"season":"Bridal",  "fabric":"Silk"},
    "Festive Georgette":{"visible_layers":[0,1,3,4,5,6,13,15,18,19,20,21],"season":"Festive","fabric":"Georgette"},
    "Luxury Velvet"  :{"visible_layers":[0,1,7,8,9,10,13,15,18,19,20,21],"season":"Luxury", "fabric":"Velvet"},
    "Casual Cotton"  :{"visible_layers":[0,1,3,5,12,13,14,18,21],         "season":"Casual", "fabric":"Cotton"},
    "Ikat Traditional":{"visible_layers":[0,1,11,9,13,15,18,19,20],       "season":"Festive","fabric":"Cotton"},
    "Geometric Modern":{"visible_layers":[0,1,9,10,12,13,15,18,19,21],    "season":"Casual", "fabric":"Polyester"},
    "Monsoon Linen"  :{"visible_layers":[0,1,6,9,12,13,17,18,21],         "season":"Monsoon","fabric":"Linen"},
    "Summer Chiffon" :{"visible_layers":[0,2,3,5,13,14,15,21],            "season":"Summer", "fabric":"Chiffon"},
}

FABRICS = ["Silk","Georgette","Cotton","Polyester","Velvet","Chiffon","Linen","Rayon","Crepe","Net"]
MARKETS = ["Premium","Mid Range","Budget","Export"]

FABRIC_TEXTURE_MAP = {
    "Silk":"silk","Georgette":"silk","Cotton":"linen","Polyester":"smooth",
    "Velvet":"rough","Chiffon":"silk","Linen":"linen","Rayon":"smooth",
    "Crepe":"rough","Net":"smooth",
}
MARKET_PRICE = {
    "Premium" :{"Silk":2800,"Velvet":2200,"Georgette":900,"Chiffon":700,
                "Cotton":400,"Polyester":350,"Linen":1200,"Net":600,"Rayon":500,"Crepe":650},
    "Mid Range":{"Silk":1500,"Velvet":1000,"Georgette":550,"Chiffon":400,
                 "Cotton":250,"Polyester":200,"Linen":700,"Net":350,"Rayon":280,"Crepe":380},
    "Budget"  :{"Silk":800,"Velvet":500,"Georgette":280,"Chiffon":200,
                "Cotton":120,"Polyester":90,"Linen":380,"Net":180,"Rayon":150,"Crepe":200},
    "Export"  :{"Silk":3500,"Velvet":2800,"Georgette":1100,"Chiffon":900,
                "Cotton":550,"Polyester":400,"Linen":1500,"Net":750,"Rayon":600,"Crepe":800},
}

# ═══════════════════════════════════════════════════════════════
# COLOUR UTILITIES
# ═══════════════════════════════════════════════════════════════
def hex_to_rgb(hex_c):
    hex_c = hex_c.lstrip("#")
    return tuple(int(hex_c[i:i+2],16) for i in (0,2,4))

def rgb_to_hex(r,g,b):
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"

def get_hex(color_name):
    return COLOR_LIBRARY.get(color_name, "#888888")

def apply_hue_shift(hex_c, hue_shift=0, sat_mult=1.0, val_mult=1.0):
    """Shift hue/saturation/value of a colour."""
    r,g,b = [x/255.0 for x in hex_to_rgb(hex_c)]
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    h = (h + hue_shift/360.0) % 1.0
    s = min(1.0, s * sat_mult)
    v = min(1.0, v * val_mult)
    r2,g2,b2 = colorsys.hsv_to_rgb(h,s,v)
    return rgb_to_hex(r2*255,g2*255,b2*255)

def color_distance(hex1, hex2):
    r1,g1,b1 = hex_to_rgb(hex1)
    r2,g2,b2 = hex_to_rgb(hex2)
    return math.sqrt((r1-r2)**2+(g1-g2)**2+(b1-b2)**2)

def luminance(hex_c):
    r,g,b = [x/255.0 for x in hex_to_rgb(hex_c)]
    return 0.2126*r + 0.7152*g + 0.0722*b

def colorize_mask(mask_gray, hex_color, opacity=1.0):
    """
    Apply a colour to a grayscale mask.
    mask_gray: PIL Image mode 'L' — white=draw, black=transparent
    Returns RGBA layer.
    """
    r,g,b = hex_to_rgb(hex_color)
    rgba = Image.new("RGBA", mask_gray.size, (r,g,b,0))
    alpha_arr = np.array(mask_gray, dtype=np.float32)
    alpha_arr = (alpha_arr * opacity).astype(np.uint8)
    alpha_ch  = Image.fromarray(alpha_arr, "L")
    rgba.putalpha(alpha_ch)
    return rgba

# ═══════════════════════════════════════════════════════════════
# PATTERN GENERATORS — each returns a grayscale 'L' mask
# ═══════════════════════════════════════════════════════════════
SZ = 512  # default tile size

def _blank(sz): return Image.new("L",(sz,sz),0)

def gen_bg_solid(sz, **kw):
    img = Image.new("L",(sz,sz),255)
    return img

def gen_bg_texture(sz, style="linen", **kw):
    img = Image.new("L",(sz,sz),0)
    draw= ImageDraw.Draw(img)
    if style=="linen":
        for x in range(0,sz,3):
            v = random.randint(100,160)
            draw.line([(x,0),(x,sz)],fill=v,width=1)
        for y in range(0,sz,3):
            v = random.randint(80,130)
            draw.line([(0,y),(sz,y)],fill=v,width=1)
    elif style=="silk":
        for i in range(0,sz+sz,6):
            v = int(120 + 60*math.sin(i*0.08))
            draw.line([(i,0),(0,i)],fill=v,width=2)
    elif style=="rough":
        arr = np.random.randint(40,120,(sz,sz),dtype=np.uint8)
        img = Image.fromarray(arr,"L")
    else:  # smooth
        arr = np.full((sz,sz),90,dtype=np.uint8)
        img = Image.fromarray(arr,"L")
    return img.filter(ImageFilter.GaussianBlur(0.8))

def gen_gradient_wash(sz, **kw):
    arr = np.zeros((sz,sz),dtype=np.uint8)
    for y in range(sz):
        v = int(255 * (1 - y/sz) * 0.6)
        arr[y,:] = v
    return Image.fromarray(arr,"L")

def gen_floral_large(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    step=sz//4
    for cx in range(step//2,sz,step):
        for cy in range(step//2,sz,step):
            r_petal = sz//20
            for angle in range(0,360,45):
                rad=math.radians(angle)
                px=cx+int(r_petal*1.8*math.cos(rad))
                py=cy+int(r_petal*1.8*math.sin(rad))
                draw.ellipse([px-r_petal,py-int(r_petal*.6),
                              px+r_petal,py+int(r_petal*.6)],fill=220)
            # inner ring
            draw.ellipse([cx-r_petal//2,cy-r_petal//2,
                          cx+r_petal//2,cy+r_petal//2],fill=255)
            draw.ellipse([cx-r_petal//4,cy-r_petal//4,
                          cx+r_petal//4,cy+r_petal//4],fill=150)
    return img.filter(ImageFilter.GaussianBlur(0.5))

def gen_floral_medium(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    step=sz//6
    random.seed(77)
    for cx in range(step//2,sz,step):
        for cy in range(step//2,sz,step):
            ox,oy = random.randint(-step//4,step//4), random.randint(-step//4,step//4)
            cx2,cy2=cx+ox,cy+oy
            rp=sz//28
            for angle in range(0,360,60):
                rad=math.radians(angle)
                px=cx2+int(rp*1.6*math.cos(rad))
                py=cy2+int(rp*1.6*math.sin(rad))
                draw.ellipse([px-rp,py-rp,px+rp,py+rp],fill=190)
            draw.ellipse([cx2-rp//2,cy2-rp//2,cx2+rp//2,cy2+rp//2],fill=255)
    return img

def gen_floral_small(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    random.seed(33)
    for _ in range(sz//8):
        cx=random.randint(10,sz-10); cy=random.randint(10,sz-10)
        rp=random.randint(sz//50,sz//30)
        n_petals=random.choice([5,6,8])
        for angle in range(0,360,360//n_petals):
            rad=math.radians(angle)
            px=cx+int(rp*1.4*math.cos(rad)); py=cy+int(rp*1.4*math.sin(rad))
            draw.ellipse([px-rp//2,py-rp//2,px+rp//2,py+rp//2],fill=160)
        draw.ellipse([cx-rp//3,cy-rp//3,cx+rp//3,cy+rp//3],fill=240)
    return img

def gen_vines(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    random.seed(99)
    for _ in range(6):
        x=random.randint(0,sz); y=random.randint(0,sz)
        pts=[(x,y)]
        for step in range(30):
            dx=random.randint(-sz//15,sz//15); dy=random.randint(-sz//15,sz//15)
            x=max(0,min(sz-1,x+dx)); y=max(0,min(sz-1,y+dy))
            pts.append((x,y))
        if len(pts)>1:
            draw.line(pts,fill=200,width=max(1,sz//80))
        # leaves along vine
        for px,py in pts[::3]:
            leaf_r=sz//35
            draw.ellipse([px-leaf_r,py-leaf_r//2,px+leaf_r,py+leaf_r//2],fill=180)
    return img.filter(ImageFilter.GaussianBlur(0.8))

def gen_paisley_large(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    step=sz//3
    for row,cy in enumerate(range(step//2,sz,step)):
        off=(step//2) if row%2 else 0
        for cx in range(off,sz,step):
            h=sz//9; w=h//2
            draw.ellipse([cx-w,cy-h,cx+w,cy+h],fill=200)
            draw.ellipse([cx-w//2,cy+h//3,cx+w//2,cy+h],fill=230)
            # inner curl
            draw.ellipse([cx-w//3,cy-h//3,cx+w//3,cy+h//3],fill=120)
            draw.ellipse([cx-w//6,cy-h//6,cx+w//6,cy+h//6],fill=255)
    return img

def gen_paisley_small(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    random.seed(44)
    for _ in range(sz//40):
        cx=random.randint(20,sz-20); cy=random.randint(20,sz-20)
        h=sz//16; w=h//2
        draw.ellipse([cx-w,cy-h,cx+w,cy+h],fill=160)
        draw.ellipse([cx-w//2,cy+h//3,cx+w//2,cy+h],fill=200)
        draw.ellipse([cx-w//4,cy-h//4,cx+w//4,cy+h//4],fill=255)
    return img

def gen_geo_grid(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    step=sz//8
    for x in range(0,sz,step):
        draw.line([(x,0),(x,sz)],fill=180,width=max(1,sz//200))
    for y in range(0,sz,step):
        draw.line([(0,y),(sz,y)],fill=180,width=max(1,sz//200))
    return img

def gen_diamonds(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    step=sz//6
    for row,cy in enumerate(range(0,sz,step)):
        off=(step//2) if row%2 else 0
        for cx in range(off,sz,step):
            s=step//2-2
            draw.polygon([(cx,cy-s),(cx+s,cy),(cx,cy+s),(cx-s,cy)],
                         outline=200,fill=0)
    return img

def gen_ikat_bands(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    step=sz//8
    random.seed(55)
    for d in range(-sz,sz*2,step*2):
        for off in range(-6,6,2):
            jitter=random.randint(-4,4)
            v=random.randint(140,220)
            draw.line([(d+off+jitter,0),(d+off+jitter+sz,sz)],fill=v,width=3)
    return img.filter(ImageFilter.GaussianBlur(1.2))

def gen_stripes(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    w=sz//16
    for x in range(0,sz,w*2):
        draw.rectangle([x,0,x+w,sz],fill=160)
    return img

def gen_dots_large(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    step=sz//8
    for cx in range(step//2,sz,step):
        for cy in range(step//2,sz,step):
            r=max(2,sz//60)
            draw.ellipse([cx-r,cy-r,cx+r,cy+r],fill=230)
    return img

def gen_dots_small(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    random.seed(66)
    for _ in range(sz//4):
        x=random.randint(4,sz-4); y=random.randint(4,sz-4)
        r=random.randint(1,max(2,sz//80))
        draw.ellipse([x-r,y-r,x+r,y+r],fill=200)
    return img

def gen_stars(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    step=sz//6
    random.seed(88)
    for cx in range(step//2,sz,step):
        for cy in range(step//2,sz,step):
            ox,oy=random.randint(-step//4,step//4),random.randint(-step//4,step//4)
            sx,sy=cx+ox,cy+oy
            r1,r2=max(3,sz//40),max(1,sz//80)
            pts=[]
            for i in range(10):
                a=math.radians(i*36-90)
                r=r1 if i%2==0 else r2
                pts.append((sx+int(r*math.cos(a)),sy+int(r*math.sin(a))))
            draw.polygon(pts,fill=220)
    return img

def gen_sprigs(sz, **kw):
    img = Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    random.seed(111)
    for _ in range(sz//30):
        x=random.randint(10,sz-10); y=random.randint(10,sz-10)
        for angle in [0,45,90,135,180,225,270,315]:
            rad=math.radians(angle); length=random.randint(sz//30,sz//15)
            ex=x+int(length*math.cos(rad)); ey=y+int(length*math.sin(rad))
            draw.line([(x,y),(ex,ey)],fill=170,width=1)
            lr=random.randint(sz//60,sz//40)
            draw.ellipse([ex-lr,ey-lr//2,ex+lr,ey+lr//2],fill=150)
    return img

def gen_stipple(sz, **kw):
    random.seed(222)
    arr=np.zeros((sz,sz),dtype=np.uint8)
    for _ in range(sz*sz//20):
        x=random.randint(0,sz-1); y=random.randint(0,sz-1)
        arr[y,x]=random.randint(60,140)
    img=Image.fromarray(arr,"L")
    return img.filter(ImageFilter.GaussianBlur(0.4))

def gen_border_outer(sz, **kw):
    img=Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    b=max(4,sz//40)
    draw.rectangle([0,0,sz-1,b],fill=255)
    draw.rectangle([0,sz-b-1,sz-1,sz-1],fill=255)
    draw.rectangle([0,0,b,sz-1],fill=255)
    draw.rectangle([sz-b-1,0,sz-1,sz-1],fill=255)
    b2=b+max(2,sz//120)
    draw.rectangle([b2,b2,sz-b2-1,b2+max(1,sz//200)],fill=200)
    draw.rectangle([b2,sz-b2-max(1,sz//200)-1,sz-b2-1,sz-b2-1],fill=200)
    return img

def gen_border_inner(sz, **kw):
    img=Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    m=max(10,sz//30)
    lw=max(1,sz//200)
    draw.rectangle([m,m,sz-m-1,sz-m-1],outline=180,width=lw)
    m2=m+max(3,sz//100)
    draw.rectangle([m2,m2,sz-m2-1,sz-m2-1],outline=120,width=lw)
    return img

def gen_corner_ornaments(sz, **kw):
    img=Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    s=max(20,sz//14)
    for (cx,cy) in [(s,s),(sz-s,s),(s,sz-s),(sz-s,sz-s)]:
        for r in [s,s-s//3,s-2*s//3]:
            if r>0:
                draw.ellipse([cx-r,cy-r,cx+r,cy+r],outline=200,width=max(1,sz//200))
        r2=max(2,s//4)
        draw.ellipse([cx-r2,cy-r2,cx+r2,cy+r2],fill=240)
        # cross
        cw=max(1,sz//300)
        draw.line([(cx-s+2,cy),(cx+s-2,cy)],fill=150,width=cw)
        draw.line([(cx,cy-s+2),(cx,cy+s-2)],fill=150,width=cw)
    return img

def gen_shimmer(sz, **kw):
    random.seed(333)
    img=Image.new("L",(sz,sz),0); draw=ImageDraw.Draw(img)
    for _ in range(sz*sz//60):
        x=random.randint(0,sz-1); y=random.randint(0,sz-1)
        r=random.randint(1,max(2,sz//100))
        v=random.randint(100,220)
        draw.ellipse([x-r,y-r,x+r,y+r],fill=v)
    return img.filter(ImageFilter.GaussianBlur(0.6))

# Map fn keys → generator functions
PATTERN_FNS = {
    "bg_solid"     : gen_bg_solid,
    "bg_texture"   : gen_bg_texture,
    "gradient_wash": gen_gradient_wash,
    "floral_large" : gen_floral_large,
    "floral_medium": gen_floral_medium,
    "floral_small" : gen_floral_small,
    "vines"        : gen_vines,
    "paisley_large": gen_paisley_large,
    "paisley_small": gen_paisley_small,
    "geo_grid"     : gen_geo_grid,
    "diamonds"     : gen_diamonds,
    "ikat_bands"   : gen_ikat_bands,
    "stripes"      : gen_stripes,
    "dots_large"   : gen_dots_large,
    "dots_small"   : gen_dots_small,
    "stars"        : gen_stars,
    "sprigs"       : gen_sprigs,
    "stipple"      : gen_stipple,
    "border_outer" : gen_border_outer,
    "border_inner" : gen_border_inner,
    "corner_ornaments": gen_corner_ornaments,
    "shimmer"      : gen_shimmer,
}

# ═══════════════════════════════════════════════════════════════
# BLEND MODE ENGINE
# ═══════════════════════════════════════════════════════════════
def blend_layers(base, top, mode="normal"):
    """
    Blend two RGBA PIL images using the given mode.
    Returns RGBA image.
    """
    if base is None:
        return top
    # Extract RGB channels as float arrays [0,1]
    base_arr = np.array(base.convert("RGBA"), dtype=np.float32) / 255.0
    top_arr  = np.array(top.convert("RGBA"),  dtype=np.float32) / 255.0

    b_rgb = base_arr[:,:,:3]
    t_rgb = top_arr[:,:,:3]
    t_a   = top_arr[:,:,3:4]   # top alpha as mask
    b_a   = base_arr[:,:,3:4]

    if mode == "normal":
        blended = t_rgb
    elif mode == "multiply":
        blended = b_rgb * t_rgb
    elif mode == "screen":
        blended = 1 - (1-b_rgb)*(1-t_rgb)
    elif mode == "overlay":
        dark  = 2*b_rgb*t_rgb
        light = 1 - 2*(1-b_rgb)*(1-t_rgb)
        blended = np.where(b_rgb < 0.5, dark, light)
    elif mode == "soft_light":
        blended = (1-2*t_rgb)*b_rgb**2 + 2*t_rgb*b_rgb
    elif mode == "hard_light":
        dark  = 2*b_rgb*t_rgb
        light = 1 - 2*(1-b_rgb)*(1-t_rgb)
        blended = np.where(t_rgb < 0.5, dark, light)
    else:
        blended = t_rgb

    # Alpha composite: out = top_a * blended + (1-top_a) * base_rgb
    result_rgb = t_a * blended + (1-t_a) * b_rgb
    result_a   = t_a + b_a*(1-t_a)
    result     = np.clip(np.concatenate([result_rgb, result_a], axis=2) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result, "RGBA")

# ═══════════════════════════════════════════════════════════════
# COMPOSITE ENGINE — renders all visible layers
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, max_entries=50)
def render_layer_mask(fn_key, sz, fabric):
    """Cache each layer mask by (fn_key, sz, fabric_texture)."""
    texture_style = FABRIC_TEXTURE_MAP.get(fabric, "smooth")
    fn = PATTERN_FNS.get(fn_key, gen_bg_solid)
    return fn(sz, style=texture_style)

def composite_design(layers_config, sz=512, fabric="Silk"):
    """
    Composite all visible layers into a single RGBA PIL Image.
    layers_config: list of dicts with id,visible,color,opacity,blend,fn
    Returns PIL RGBA image.
    """
    result = None
    for lc in layers_config:
        if not lc["visible"]: continue
        mask  = render_layer_mask(lc["fn"], sz, fabric)
        hex_c = get_hex(lc["color"])
        layer = colorize_mask(mask, hex_c, lc["opacity"])
        result = blend_layers(result, layer, lc["blend"])
    if result is None:
        result = Image.new("RGBA", (sz,sz), (200,200,200,255))
    return result

# ═══════════════════════════════════════════════════════════════
# TRUE SEAMLESS ENGINE
# ═══════════════════════════════════════════════════════════════
def make_seamless_offset(tile_rgba, blend_w=None):
    """
    Offset-wrap seamless method.
    Shifts image 50% in X and Y, blends seam zone.
    TRULY removes visible seam lines.
    """
    W,H = tile_rgba.size
    if blend_w is None: blend_w = max(20, W//12)
    arr = np.array(tile_rgba.convert("RGBA"), dtype=np.float32)
    arr_s = np.roll(np.roll(arr, H//2, axis=0), W//2, axis=1)

    # Build smooth feather mask for seam zone
    mask = np.zeros((H,W), dtype=np.float32)
    bw = blend_w
    for i in range(bw):
        alpha = math.sin((i/bw) * math.pi/2)  # sinusoidal ease
        hy, hx = H//2, W//2
        if 0 <= hy-bw+i < H: mask[hy-bw+i,:] = np.maximum(mask[hy-bw+i,:], alpha)
        if 0 <= hy+bw-i-1 < H: mask[hy+bw-i-1,:] = np.maximum(mask[hy+bw-i-1,:], alpha)
        if 0 <= hx-bw+i < W: mask[:,hx-bw+i] = np.maximum(mask[:,hx-bw+i], alpha)
        if 0 <= hx+bw-i-1 < W: mask[:,hx+bw-i-1] = np.maximum(mask[:,hx+bw-i-1], alpha)

    mimg = Image.fromarray((mask*255).astype(np.uint8),"L")
    mimg = mimg.filter(ImageFilter.GaussianBlur(bw//3))
    mask_s = np.array(mimg, dtype=np.float32)/255.0
    mask4  = np.stack([mask_s]*4, axis=2)
    blended= np.clip(arr*(1-mask4) + arr_s*mask4, 0,255).astype(np.uint8)
    return Image.fromarray(blended,"RGBA")

# ═══════════════════════════════════════════════════════════════
# 4 REPEAT MODES — all produce zero-seam tiling
# ═══════════════════════════════════════════════════════════════
def repeat_block(tile, rows=3, cols=4):
    """Standard full-drop block repeat."""
    seamless = make_seamless_offset(tile)
    W,H = seamless.size
    canvas = Image.new("RGBA",(W*cols, H*rows),(255,255,255,255))
    for r in range(rows):
        for c in range(cols):
            canvas.paste(seamless,(c*W, r*H))
    return canvas

def repeat_half_drop(tile, rows=3, cols=4):
    """
    Half-drop: odd columns offset downward by H/2.
    Industry standard for organic/floral patterns.
    """
    seamless = make_seamless_offset(tile)
    W,H = seamless.size
    canvas = Image.new("RGBA",(W*cols, H*rows+H//2),(255,255,255,255))
    for c in range(cols):
        offset_y = (H//2) if c%2==1 else 0
        for r in range(rows+1):
            y = r*H - H//2 + offset_y
            if -H < y < H*rows+H:
                canvas.paste(seamless,(c*W, y), seamless)
    return canvas.crop((0, 0, W*cols, H*rows))

def repeat_brick(tile, rows=3, cols=4):
    """
    Brick repeat: odd rows offset right by W/2.
    Perfect for geometric and check patterns.
    """
    seamless = make_seamless_offset(tile)
    W,H = seamless.size
    canvas = Image.new("RGBA",(W*cols+W//2, H*rows),(255,255,255,255))
    for r in range(rows):
        offset_x = (W//2) if r%2==1 else 0
        for c in range(cols+1):
            x = c*W - W//2 + offset_x
            if -W < x < W*cols+W:
                canvas.paste(seamless,(x, r*H), seamless)
    return canvas.crop((0, 0, W*cols, H*rows))

def repeat_mirror(tile, rows=3, cols=4):
    """
    Mirror repeat — GUARANTEED seamless, zero visible lines.
    Mirrors tile in X and Y alternately.
    """
    W,H = tile.size
    # Create 2×2 super-tile with mirrors — perfectly seamless by definition
    tile_h  = tile.transpose(Image.FLIP_LEFT_RIGHT)
    tile_v  = tile.transpose(Image.FLIP_TOP_BOTTOM)
    tile_hv = tile.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    super_tile = Image.new("RGBA",(W*2,H*2))
    super_tile.paste(tile,   (0,0))
    super_tile.paste(tile_h, (W,0))
    super_tile.paste(tile_v, (0,H))
    super_tile.paste(tile_hv,(W,H))
    # Now tile the super-tile
    SW,SH = super_tile.size
    canvas = Image.new("RGBA",(SW*math.ceil(cols/2), SH*math.ceil(rows/2)),(255,255,255,255))
    for r in range(math.ceil(rows/2)):
        for c in range(math.ceil(cols/2)):
            canvas.paste(super_tile,(c*SW, r*SH))
    return canvas.crop((0,0,W*cols,H*rows))

def apply_repeat(tile, mode, rows=3, cols=4):
    """Dispatcher for repeat modes."""
    tile_rgba = tile.convert("RGBA")
    if mode == "Block":      return repeat_block(tile_rgba, rows, cols)
    if mode == "Half Drop":  return repeat_half_drop(tile_rgba, rows, cols)
    if mode == "Brick":      return repeat_brick(tile_rgba, rows, cols)
    if mode == "Mirror":     return repeat_mirror(tile_rgba, rows, cols)
    return repeat_block(tile_rgba, rows, cols)

# ═══════════════════════════════════════════════════════════════
# COLORWAY ENGINE — generate N color variants of a design
# ═══════════════════════════════════════════════════════════════
def generate_colorway(layers_config, season, n_variants=5):
    """
    Generate N colorway variants for a season.
    Shifts hues while keeping relative layer harmony.
    """
    season_cols = SEASON_COLORS.get(season, list(COLOR_LIBRARY.keys())[:8])
    variants = []
    hue_shifts = np.linspace(0, 300, n_variants)

    for i, hshift in enumerate(hue_shifts):
        variant = copy.deepcopy(layers_config)
        for j, lc in enumerate(variant):
            if not lc["visible"]: continue
            base_hex = get_hex(lc["color"])
            new_hex  = apply_hue_shift(base_hex, hue_shift=hshift,
                                        sat_mult=random.uniform(0.85,1.15),
                                        val_mult=random.uniform(0.9,1.1))
            variant[j]["color_hex_override"] = new_hex
        variants.append(variant)
    return variants

def suggest_season_palette(season, n=5):
    """Return top N harmony-scored palettes for a season."""
    base   = SEASON_COLORS.get(season, list(COLOR_LIBRARY.keys())[:8])
    all_c  = list(COLOR_LIBRARY.keys())
    scored = []
    random.seed(SEED)
    for _ in range(300):
        nb  = random.randint(2, min(4,len(base)))
        na  = max(0, 4-nb)
        pal = list(dict.fromkeys(random.sample(base,nb) + (random.sample(all_c,na) if na>0 else [])))[:4]
        if len(pal)<3: continue
        hexes = [get_hex(c) for c in pal]
        pairs = list(combinations(hexes,2))
        if not pairs: continue
        lums  = [luminance(h) for h in hexes]
        contrast_scores = [(max(l1,l2)+.05)/(min(l1,l2)+.05) for l1,l2 in combinations(lums,2)]
        avg_cr  = np.mean(contrast_scores)
        variety = np.std([hex_to_rgb(h)[0]+hex_to_rgb(h)[1]+hex_to_rgb(h)[2] for h in hexes])
        score   = round(min(avg_cr/5.0,.6)*10 + min(variety/100,.4)*10, 2)
        scored.append((pal, round(score,1)))
    scored.sort(key=lambda x:x[1],reverse=True)
    seen,out=[],[]
    for p,s in scored:
        k=frozenset(p)
        if k not in seen: seen.append(k); out.append((p,s))
        if len(out)>=n: break
    return out

# ═══════════════════════════════════════════════════════════════
# DEMAND PREDICTION ENGINE (cached ML model)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🧠 Training demand intelligence model...")
def build_demand_model():
    np.random.seed(SEED); random.seed(SEED); n=2000
    fab_t  = ["Cotton","Silk","Polyester","Georgette","Chiffon","Net","Velvet","Linen","Rayon","Crepe"]
    col_c  = ["red_gold","blue_silver","green_white","pink_peach","maroon_beige",
              "black_red","yellow_orange","purple_pink","teal_gold","ivory_rust","navy_cream","coral_mint"]
    pat_t  = ["floral","geometric","paisley","abstract","stripes","checks","embroidered","ikat"]
    seas   = ["Wedding","Festive","Casual","Summer","Winter","Monsoon","Bridal","Luxury"]
    mkts   = ["Premium","Mid Range","Budget","Export"]

    fabrics  = np.random.choice(fab_t,n,p=[.18,.12,.20,.10,.08,.07,.05,.08,.07,.05])
    colors   = np.random.choice(col_c,n)
    patterns = np.random.choice(pat_t,n)
    season   = np.random.choice(seas,n)
    market   = np.random.choice(mkts,n,p=[.20,.35,.30,.15])

    base_p = {"Cotton":250,"Silk":1800,"Polyester":180,"Georgette":600,"Chiffon":500,
              "Net":450,"Velvet":1200,"Linen":800,"Rayon":300,"Crepe":420}
    seas_m = {"Wedding":1.6,"Festive":1.4,"Casual":1.0,"Summer":0.9,
              "Winter":1.2,"Monsoon":0.85,"Bridal":1.8,"Luxury":2.2}
    mkt_m  = {"Premium":2.0,"Mid Range":1.2,"Budget":0.7,"Export":1.5}

    prices = np.array([int(base_p[f]*seas_m[s]*mkt_m[m]*np.random.uniform(.85,1.15))
                       for f,s,m in zip(fabrics,season,market)])
    thread = np.random.randint(60,600,n)
    trend  = np.round(np.random.beta(2,2,n)*10,1)

    ds=np.zeros(n)
    ds+=np.where(np.isin(fabrics,["Silk","Georgette","Velvet"]),2,0)
    ds+=np.where(np.isin(season,["Wedding","Bridal","Festive","Luxury"]),2,0)
    ds+=np.where(np.isin(patterns,["floral","embroidered","paisley"]),1,0)
    ds+=np.random.normal(0,1,n)
    sales=np.where(ds>=4,"High",np.where(ds>=2,"Medium","Low"))

    df=pd.DataFrame({"fabric":fabrics,"color":colors,"pattern":patterns,
                     "season":season,"market":market,"price":prices,
                     "thread":thread,"trend":trend,"demand":sales})

    cat_cols=["fabric","color","pattern","season","market"]
    feat_cols=cat_cols+["price","thread","trend"]
    encoders={}
    df_enc=df[feat_cols].copy()
    for col in cat_cols:
        le=LabelEncoder()
        df_enc[col]=le.fit_transform(df_enc[col].astype(str))
        encoders[col]=le
    le_y=LabelEncoder()
    y=le_y.fit_transform(df["demand"])
    X=df_enc.values
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    clf=RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=4,
                                random_state=SEED,class_weight="balanced")
    clf.fit(Xtr,ytr)
    acc=accuracy_score(yte,clf.predict(Xte))
    return clf,encoders,le_y,feat_cols,cat_cols,acc,df

def predict_demand(clf,encoders,le_y,feat_cols,cat_cols,fab,pat,sea,mkt,price):
    row=pd.DataFrame([{"fabric":fab,"color":"red_gold","pattern":pat.lower(),
                        "season":sea,"market":mkt,"price":price,"thread":300,"trend":7.0}])
    df_enc=row[feat_cols].copy()
    for col in cat_cols:
        le=encoders[col]
        df_enc[col]=df_enc[col].astype(str).map(
            lambda x,le=le: le.transform([x])[0] if x in le.classes_ else 0)
    label=le_y.classes_[clf.predict(df_enc.values)[0]]
    proba=clf.predict_proba(df_enc.values)[0]
    conf={le_y.classes_[i]:float(proba[i]) for i in range(len(le_y.classes_))}
    return label,conf

# ═══════════════════════════════════════════════════════════════
# PIL → bytes helper
# ═══════════════════════════════════════════════════════════════
def pil_to_bytes(img,fmt="PNG"):
    buf=io.BytesIO()
    img.convert("RGB").save(buf,format=fmt)
    return buf.getvalue()

def palette_swatch_image(colors_list, w=480, h=70):
    fig,ax=plt.subplots(figsize=(w/80,h/80))
    fig.patch.set_facecolor("#13131F"); ax.set_facecolor("#13131F")
    n=len(colors_list)
    for i,cname in enumerate(colors_list):
        hex_c=get_hex(cname) if cname in COLOR_LIBRARY else cname
        rect=mpatches.FancyBboxPatch((i/n+.005,.05),.9/n,.85,
             boxstyle="round,pad=.02",facecolor=hex_c,edgecolor="#0B0B14",linewidth=1.5)
        ax.add_patch(rect)
        lum=luminance(hex_c); tc="black" if lum>.35 else "white"
        ax.text(i/n+.5/n,.5,cname.replace(" ","\n") if " " in cname else cname,
                ha="center",va="center",fontsize=6.5,color=tc,fontweight="bold")
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    buf=io.BytesIO()
    plt.savefig(buf,format="png",dpi=100,bbox_inches="tight",facecolor="#13131F")
    plt.close()
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════
if "layers" not in st.session_state:
    st.session_state.layers = copy.deepcopy(LAYER_DEFS)
if "repeat_mode" not in st.session_state:
    st.session_state.repeat_mode = "Mirror"
if "canvas_size" not in st.session_state:
    st.session_state.canvas_size = 512
if "fabric" not in st.session_state:
    st.session_state.fabric = "Silk"
if "season" not in st.session_state:
    st.session_state.season = "Wedding"
if "market" not in st.session_state:
    st.session_state.market = "Premium"
if "tile_img" not in st.session_state:
    st.session_state.tile_img = None
if "tiled_img" not in st.session_state:
    st.session_state.tiled_img = None
if "needs_render" not in st.session_state:
    st.session_state.needs_render = True
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Studio"

# Load model once
clf,encoders,le_y,feat_cols,cat_cols,model_acc,train_df = build_demand_model()

# ═══════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">✦ Professional Textile Design Studio ✦</div>
  <div class="hero-title">MYTH AI 2.0</div>
  <div class="hero-desc">
    22-Layer compositing engine · Live colorway editor · 4 seamless repeat modes ·
    Zero-seam tiling · AI demand intelligence · Export ready
  </div>
  <div class="hero-badges">
    <span class="badge">22 Layers</span>
    <span class="badge">Colorway Engine</span>
    <span class="badge">Mirror · Half-Drop · Brick · Block</span>
    <span class="badge">AI Demand Score</span>
    <span class="badge">Export PNG</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR — LAYER CONTROL PANEL
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:4px 0 16px'>
      <div style='font-family:Cormorant Garamond,serif;font-size:1.25rem;
                  color:#C8A96E;font-weight:700;'>Layer Control Panel</div>
      <div style='color:#6B6B8A;font-size:.75rem;margin-top:3px;'>
        22 compositing layers
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Preset selector ───────────────────────────
    st.markdown('<div class="card-title-sm">⚡ Quick Presets</div>', unsafe_allow_html=True)
    preset_choice = st.selectbox("Load Preset", ["— choose —"] + list(PATTERN_PRESETS.keys()),
                                  label_visibility="collapsed")
    if preset_choice != "— choose —":
        preset = PATTERN_PRESETS[preset_choice]
        for lc in st.session_state.layers:
            lc["visible"] = lc["id"] in preset["visible_layers"]
        st.session_state.season  = preset["season"]
        st.session_state.fabric  = preset["fabric"]
        st.session_state.needs_render = True

    st.markdown("---")

    # ── Fabric / Season / Market ──────────────────
    st.markdown('<div class="card-title-sm">🧵 Design Parameters</div>', unsafe_allow_html=True)

    fab_idx = FABRICS.index(st.session_state.fabric) if st.session_state.fabric in FABRICS else 0
    fabric = st.selectbox("Fabric", FABRICS, index=fab_idx)
    if fabric != st.session_state.fabric:
        st.session_state.fabric = fabric
        st.session_state.needs_render = True

    seas_list = list(SEASON_COLORS.keys())
    sea_idx   = seas_list.index(st.session_state.season) if st.session_state.season in seas_list else 0
    season = st.selectbox("Season / Occasion", seas_list, index=sea_idx)
    if season != st.session_state.season:
        st.session_state.season = season

    mkt_idx = MARKETS.index(st.session_state.market) if st.session_state.market in MARKETS else 0
    market = st.selectbox("Target Market", MARKETS, index=mkt_idx)
    st.session_state.market = market

    st.markdown("---")

    # ── Repeat Mode ───────────────────────────────
    st.markdown('<div class="card-title-sm">🔁 Repeat Mode</div>', unsafe_allow_html=True)
    repeat_mode = st.radio("Repeat", ["Mirror","Half Drop","Brick","Block"],
                            index=["Mirror","Half Drop","Brick","Block"].index(st.session_state.repeat_mode),
                            horizontal=True, label_visibility="collapsed")
    if repeat_mode != st.session_state.repeat_mode:
        st.session_state.repeat_mode = repeat_mode

    st.markdown("---")

    # ── Canvas Size ───────────────────────────────
    st.markdown('<div class="card-title-sm">📐 Canvas Size</div>', unsafe_allow_html=True)
    sz_map = {"512 px":512, "768 px":768, "1024 px":1024}
    sz_choice = st.select_slider("Size", list(sz_map.keys()), value="512 px", label_visibility="collapsed")
    new_sz = sz_map[sz_choice]
    if new_sz != st.session_state.canvas_size:
        st.session_state.canvas_size = new_sz
        st.session_state.needs_render = True

    st.markdown("---")

    # ── 22 Layer Panel ────────────────────────────
    st.markdown('<div class="card-title-sm">🗂 22 Layer Stack</div>', unsafe_allow_html=True)

    cat_colors = {"Background":"#3DD68C","Pattern":"#F5A623",
                  "Geometric":"#5B8DEE","Detail":"#C8A96E","Accent":"#E05252"}

    # Group by category
    current_cat = None
    for i, lc in enumerate(st.session_state.layers):
        cat = lc["cat"]
        if cat != current_cat:
            current_cat = cat
            cat_col = cat_colors.get(cat,"#888")
            st.markdown(f"""
            <div style='font-size:.68rem;color:{cat_col};text-transform:uppercase;
                        letter-spacing:1.5px;margin:10px 0 4px;font-weight:600;'>
              ▸ {cat}
            </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns([1,4])
        with col_a:
            vis = st.checkbox("", value=lc["visible"],
                               key=f"vis_{lc['id']}", label_visibility="collapsed")
            if vis != lc["visible"]:
                st.session_state.layers[i]["visible"] = vis
                st.session_state.needs_render = True

        with col_b:
            hex_c = get_hex(lc["color"])
            dot_style = (f"width:10px;height:10px;border-radius:50%;"
                         f"background:{hex_c};border:1.5px solid #444;display:inline-block;margin-right:5px;")
            opacity_pct = int(lc["opacity"]*100)
            vis_style = "" if lc["visible"] else "opacity:.35;"
            st.markdown(f"""
            <div style='font-size:.78rem;color:#E2D9C8;{vis_style}'>
              <span style='{dot_style}'></span>
              {lc["name"]}
              <span style='color:#6B6B8A;font-size:.68rem;margin-left:4px;'>{opacity_pct}%</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Render button ─────────────────────────────
    if st.button("🎨 Render Design", type="primary"):
        st.session_state.needs_render = True

    st.markdown(f"""
    <div style='text-align:center;color:#6B6B8A;font-size:.72rem;margin-top:12px;'>
      Model: <span style='color:#C8A96E;font-weight:600;'>{model_acc:.0%}</span> accuracy
      · {len(train_df):,} samples
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# AUTO-RENDER on first load or when flagged
# ═══════════════════════════════════════════════════════════════
if st.session_state.needs_render:
    with st.spinner("🎨 Compositing 22 layers..."):
        tile = composite_design(st.session_state.layers,
                                 st.session_state.canvas_size,
                                 st.session_state.fabric)
        tiled = apply_repeat(tile, st.session_state.repeat_mode, rows=3, cols=4)
        st.session_state.tile_img  = tile
        st.session_state.tiled_img = tiled
        st.session_state.needs_render = False

tile_img  = st.session_state.tile_img
tiled_img = st.session_state.tiled_img

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🎨  Design Studio",
    "🎛  Colorway Editor",
    "🔁  Repeat Preview",
    "📊  Market Intelligence"
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — DESIGN STUDIO
# ─────────────────────────────────────────────────────────────
with tab1:
    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.markdown('<div class="card-title">🖼️ Design Canvas</div>', unsafe_allow_html=True)
        if tile_img:
            st.image(pil_to_bytes(tile_img), use_container_width=True,
                     caption=f"{st.session_state.fabric} · {st.session_state.season} · "
                             f"{st.session_state.canvas_size}×{st.session_state.canvas_size}px")

        # ── Layer-by-layer color editor ────────────
        st.markdown("---")
        st.markdown('<div class="card-title">🎨 Layer Colorway Editor</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#6B6B8A;font-size:.8rem;margin-bottom:14px;">'
                    'Change any layer colour in real-time — no re-generation needed.</div>',
                    unsafe_allow_html=True)

        vis_layers = [lc for lc in st.session_state.layers if lc["visible"]]
        cols_per_row = 3
        for row_start in range(0, len(vis_layers), cols_per_row):
            row_layers = vis_layers[row_start:row_start+cols_per_row]
            cols = st.columns(cols_per_row)
            for col_el, lc in zip(cols, row_layers):
                lid = lc["id"]
                with col_el:
                    hex_c = get_hex(lc["color"])
                    st.markdown(f"""
                    <div style='background:#0B0B14;border:1px solid #252540;border-radius:8px;
                                padding:10px;margin-bottom:8px;'>
                      <div style='display:flex;align-items:center;gap:6px;margin-bottom:8px;'>
                        <div style='width:12px;height:12px;border-radius:50%;
                                    background:{hex_c};border:1.5px solid #444;'></div>
                        <span style='font-size:.75rem;color:#E2D9C8;'>{lc['name']}</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    all_colors = list(COLOR_LIBRARY.keys())
                    cur_idx    = all_colors.index(lc["color"]) if lc["color"] in all_colors else 0
                    new_color  = st.selectbox("Color", all_colors, index=cur_idx,
                                              key=f"col_{lid}", label_visibility="collapsed")
                    if new_color != lc["color"]:
                        idx = next(i for i,l in enumerate(st.session_state.layers) if l["id"]==lid)
                        st.session_state.layers[idx]["color"] = new_color
                        st.session_state.needs_render = True
                        st.rerun()

                    new_op = st.slider("Opacity", 0.0, 1.0,
                                       value=float(lc["opacity"]),
                                       step=0.05, key=f"op_{lid}",
                                       label_visibility="collapsed")
                    if abs(new_op - lc["opacity"]) > 0.01:
                        idx = next(i for i,l in enumerate(st.session_state.layers) if l["id"]==lid)
                        st.session_state.layers[idx]["opacity"] = new_op
                        st.session_state.needs_render = True
                        st.rerun()

    with col_side:
        # ── Season Palette Suggestions ─────────────
        st.markdown('<div class="card-title">🎨 Season Colour Intelligence</div>',
                    unsafe_allow_html=True)
        palettes = suggest_season_palette(st.session_state.season, n=5)

        for rank,(pal,score) in enumerate(palettes):
            label = f"#{rank+1}  Score {score}/10" + ("  ✦ TOP" if rank==0 else "")
            st.caption(label)
            swatch = palette_swatch_image(pal)
            st.image(swatch, use_container_width=True)

        st.markdown("---")

        # ── Blend mode controls ────────────────────
        st.markdown('<div class="card-title">🔧 Layer Blend Modes</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#6B6B8A;font-size:.78rem;margin-bottom:10px;">'
                    'Change compositing blend mode for each visible layer.</div>',
                    unsafe_allow_html=True)

        for lc in st.session_state.layers:
            if not lc["visible"]: continue
            lid = lc["id"]
            cur_blend = lc.get("blend","normal")
            b_idx     = BLEND_MODES.index(cur_blend) if cur_blend in BLEND_MODES else 0
            new_blend = st.selectbox(
                lc["name"], BLEND_MODES, index=b_idx, key=f"blend_{lid}")
            if new_blend != cur_blend:
                idx = next(i for i,l in enumerate(st.session_state.layers) if l["id"]==lid)
                st.session_state.layers[idx]["blend"] = new_blend
                st.session_state.needs_render = True
                st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB 2 — COLORWAY EDITOR
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="card-title">🎛 Colorway Variants Generator</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#6B6B8A;font-size:.85rem;margin-bottom:18px;'>
    Generate multiple colour variants of your design instantly —
    like MYTH AI's colorway system. Each variant shifts hue while
    maintaining design harmony and layer relationships.
    </div>""", unsafe_allow_html=True)

    n_variants = st.slider("Number of Variants", 2, 8, 5, key="n_var")

    if st.button("⚡ Generate Colorway Variants", key="gen_cw"):
        variants = generate_colorway(st.session_state.layers, st.session_state.season, n_variants)

        cols_cw = st.columns(min(n_variants, 3))
        for vi, (variant, col_el) in enumerate(zip(variants, cols_cw * 4)):
            if vi >= n_variants: break
            with col_el:
                with st.spinner(f"Rendering variant {vi+1}..."):
                    # Apply overrides
                    variant_layers = copy.deepcopy(variant)
                    for lc in variant_layers:
                        if "color_hex_override" in lc:
                            # Find closest named colour
                            hex_ov = lc["color_hex_override"]
                            closest = min(COLOR_LIBRARY.keys(),
                                         key=lambda c: color_distance(get_hex(c), hex_ov))
                            lc["color"] = closest
                    vimg = composite_design(variant_layers,
                                            min(st.session_state.canvas_size, 512),
                                            st.session_state.fabric)
                    st.image(pil_to_bytes(vimg), use_container_width=True,
                             caption=f"Variant {vi+1}")
                    st.download_button(
                        f"⬇ V{vi+1}",
                        pil_to_bytes(vimg),
                        f"myth_variant_{vi+1}.png",
                        "image/png",
                        key=f"dl_v{vi}",
                        use_container_width=True
                    )

    st.markdown("---")
    st.markdown('<div class="card-title">🎨 Manual Hex Color Override</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#6B6B8A;font-size:.82rem;margin-bottom:14px;'>
    Enter any HEX code to apply to a layer — Pantone matched or custom brand colours.
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        vis_names = [l["name"] for l in st.session_state.layers if l["visible"]]
        target_layer_name = st.selectbox("Target Layer", vis_names, key="hex_layer")
    with c2:
        hex_input = st.text_input("HEX Code", value="#C0392B",
                                   placeholder="#RRGGBB", key="hex_in")
    with c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Apply HEX", key="apply_hex"):
            if hex_input.startswith("#") and len(hex_input)==7:
                try:
                    hex_to_rgb(hex_input)  # validate
                    # Find closest named colour
                    closest = min(COLOR_LIBRARY.keys(),
                                  key=lambda c: color_distance(get_hex(c), hex_input))
                    target_id = next(l["id"] for l in st.session_state.layers
                                     if l["name"]==target_layer_name)
                    idx = next(i for i,l in enumerate(st.session_state.layers)
                               if l["id"]==target_id)
                    st.session_state.layers[idx]["color"] = closest
                    st.session_state.needs_render = True
                    st.success(f"✅ Applied {hex_input} → matched to '{closest}'")
                    st.rerun()
                except:
                    st.error("Invalid HEX code")
            else:
                st.error("Enter a valid HEX code like #FF0000")

    # ── Current palette display ────────────────────
    st.markdown("---")
    st.markdown('<div class="card-title">🖌️ Current Design Palette</div>',
                unsafe_allow_html=True)

    active_colors = list(dict.fromkeys(
        l["color"] for l in st.session_state.layers if l["visible"]
    ))[:8]

    if active_colors:
        swatch_img = palette_swatch_image(active_colors, w=600, h=80)
        st.image(swatch_img, use_container_width=True)

        cols_sw = st.columns(len(active_colors))
        for col_el, cname in zip(cols_sw, active_colors):
            with col_el:
                hx = get_hex(cname)
                lum = luminance(hx)
                tc  = "black" if lum>.4 else "white"
                st.markdown(f"""
                <div style='background:{hx};border-radius:6px;padding:8px 4px;
                            text-align:center;border:1px solid #252540;'>
                  <div style='font-size:.65rem;color:{tc};font-weight:600;'>{cname}</div>
                  <div style='font-size:.6rem;color:{tc};opacity:.7;margin-top:2px;'>{hx}</div>
                </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 3 — REPEAT PREVIEW
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="card-title">🔁 Seamless Repeat Studio</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#6B6B8A;font-size:.85rem;margin-bottom:20px;'>
    Four industry-standard repeat modes. Mirror mode guarantees zero visible
    seam lines. Half-Drop and Brick stagger the repeat to break visual repetition.
    All modes use sinusoidal feathering to eliminate edge artefacts.
    </div>""", unsafe_allow_html=True)

    if tile_img:
        # ── All 4 modes side by side ───────────────
        st.markdown("#### All 4 Repeat Modes — Side by Side")
        modes = ["Block","Half Drop","Brick","Mirror"]
        mode_cols = st.columns(4)

        for col_el, mode in zip(mode_cols, modes):
            with col_el:
                with st.spinner(f"Rendering {mode}..."):
                    preview = apply_repeat(tile_img, mode, rows=2, cols=3)
                active_border = "2px solid #C8A96E" if mode==st.session_state.repeat_mode else "1px solid #252540"
                st.markdown(f"""
                <div style='border:{active_border};border-radius:8px;
                            padding:4px;margin-bottom:6px;'>
                """, unsafe_allow_html=True)
                st.image(pil_to_bytes(preview), use_container_width=True, caption=mode)
                st.markdown("</div>", unsafe_allow_html=True)

                if st.button(f"Use {mode}", key=f"use_{mode}", use_container_width=True):
                    st.session_state.repeat_mode = mode
                    st.session_state.needs_render = True
                    st.rerun()

        st.markdown("---")

        # ── Large export-quality repeat ────────────
        st.markdown("#### 🖨️ High-Resolution Export Preview")
        c1,c2,c3 = st.columns(3)
        with c1:
            exp_rows = st.slider("Rows", 2, 8, 4, key="exp_rows")
        with c2:
            exp_cols = st.slider("Cols", 2, 8, 5, key="exp_cols")
        with c3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🎨 Generate Export Repeat", key="gen_export_rep"):
                with st.spinner("Generating high-res repeat..."):
                    export_repeat = apply_repeat(tile_img, st.session_state.repeat_mode,
                                                 rows=exp_rows, cols=exp_cols)
                    st.image(pil_to_bytes(export_repeat), use_container_width=True)
                    st.download_button(
                        "⬇️ Download Repeat (PNG)",
                        pil_to_bytes(export_repeat),
                        f"myth_ai_{st.session_state.fabric}_{st.session_state.repeat_mode.replace(' ','_')}_repeat.png",
                        "image/png", key="dl_rep", use_container_width=True
                    )

        st.markdown("---")

        # ── Repeat mode guide ──────────────────────
        st.markdown("#### 📚 Repeat Mode Guide")
        guide_cols = st.columns(4)
        guides = [
            ("Block","Standard grid. Every tile identical. Good for geometric/regular patterns.","#5B8DEE"),
            ("Half Drop","Odd columns drop by 50%. Breaks grid rigidity. Industry favourite for florals.","#3DD68C"),
            ("Brick","Odd rows shift right 50%. Horizontal stagger. Great for checks & ikat.","#F5A623"),
            ("Mirror","Mirrors tile in X+Y. Mathematically seamless. Zero visible lines. Best for organic designs.","#C8A96E"),
        ]
        for col_el,(mode,desc,col) in zip(guide_cols,guides):
            with col_el:
                active = "✦ " if mode==st.session_state.repeat_mode else ""
                st.markdown(f"""
                <div style='background:#13131F;border:1px solid #252540;
                            border-left:3px solid {col};
                            border-radius:8px;padding:12px;height:120px;'>
                  <div style='color:{col};font-weight:600;font-size:.85rem;
                              margin-bottom:6px;'>{active}{mode}</div>
                  <div style='color:#6B6B8A;font-size:.76rem;line-height:1.5;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 4 — MARKET INTELLIGENCE
# ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="card-title">📊 AI Market Intelligence</div>',
                unsafe_allow_html=True)

    # Demand prediction
    price = MARKET_PRICE.get(st.session_state.market,{}).get(st.session_state.fabric, 500)
    primary_pattern = next(
        (l["fn"].replace("_"," ").replace("floral large","floral")
         for l in st.session_state.layers if l["visible"] and l["cat"]=="Pattern"),
        "floral"
    )
    demand_label, demand_conf = predict_demand(
        clf,encoders,le_y,feat_cols,cat_cols,
        st.session_state.fabric,
        primary_pattern,
        st.session_state.season,
        st.session_state.market,
        price
    )

    # ── Stat row ───────────────────────────────────
    s1,s2,s3,s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="stat">
          <div class="stat-val">{demand_label}</div>
          <div class="stat-lbl">Demand Level</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="stat">
          <div class="stat-val">₹{price:,}</div>
          <div class="stat-lbl">Price / Meter</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        high_pct = demand_conf.get("High",0)
        st.markdown(f"""
        <div class="stat">
          <div class="stat-val">{high_pct:.0%}</div>
          <div class="stat-lbl">High Demand Conf.</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        vis_count = sum(1 for l in st.session_state.layers if l["visible"])
        st.markdown(f"""
        <div class="stat">
          <div class="stat-val">{vis_count}/22</div>
          <div class="stat-lbl">Active Layers</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────
    c1,c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card-title">📊 Demand Confidence</div>', unsafe_allow_html=True)
        for label in ["High","Medium","Low"]:
            val  = demand_conf.get(label,0)
            pct  = int(val*100)
            col  = "#3DD68C" if label=="High" else "#F5A623" if label=="Medium" else "#E05252"
            st.markdown(f"""
            <div class="cbar-wrap">
              <div class="cbar-lbl">
                <span>{label}</span><span style='color:{col};font-weight:600;'>{pct}%</span>
              </div>
              <div class="cbar-bg">
                <div class="cbar-fill" style="width:{pct}%;background:{col};"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Mini pie
        fig,ax = plt.subplots(figsize=(3,3))
        fig.patch.set_facecolor("#13131F")
        ax.set_facecolor("#13131F")
        sizes  = [demand_conf.get(k,0) for k in ["High","Medium","Low"]]
        colors_d = ["#3DD68C","#F5A623","#E05252"]
        wedges,_ = ax.pie(sizes,colors=colors_d,startangle=90,
                          wedgeprops=dict(edgecolor="#0B0B14",linewidth=2))
        ax.legend(wedges,["High","Medium","Low"],loc="lower center",
                  frameon=False,fontsize=8,labelcolor="#6B6B8A",ncol=3,
                  bbox_to_anchor=(0.5,-0.05))
        st.pyplot(fig,use_container_width=True)
        plt.close()

    with c2:
        st.markdown('<div class="card-title">📈 Market Price Comparison</div>',
                    unsafe_allow_html=True)
        fab_prices = {f: MARKET_PRICE.get(st.session_state.market,{}).get(f,0) for f in FABRICS}
        fig,ax = plt.subplots(figsize=(5,4))
        fig.patch.set_facecolor("#13131F"); ax.set_facecolor("#13131F")
        fabs   = list(fab_prices.keys())
        prices_list = list(fab_prices.values())
        bar_colors  = ["#C8A96E" if f==st.session_state.fabric else "#252540" for f in fabs]
        bars = ax.barh(fabs, prices_list, color=bar_colors, edgecolor="#0B0B14")
        ax.set_facecolor("#13131F")
        for spine in ax.spines.values(): spine.set_edgecolor("#252540")
        ax.tick_params(colors="#6B6B8A")
        ax.set_xlabel("Price (INR/meter)", color="#6B6B8A", fontsize=8)
        for bar, val in zip(bars, prices_list):
            ax.text(val+20, bar.get_y()+bar.get_height()/2,
                    f"₹{val:,}", va="center", fontsize=7,
                    color="#C8A96E" if val==price else "#6B6B8A")
        st.pyplot(fig,use_container_width=True)
        plt.close()

    st.markdown("---")

    # ── Batch comparison ───────────────────────────
    st.markdown('<div class="card-title">🔍 Multi-Fabric Demand Comparison</div>',
                unsafe_allow_html=True)

    comp_rows = []
    for fab in FABRICS:
        pr = MARKET_PRICE.get(st.session_state.market,{}).get(fab,500)
        lbl,conf = predict_demand(clf,encoders,le_y,feat_cols,cat_cols,
                                   fab,"floral",st.session_state.season,
                                   st.session_state.market,pr)
        comp_rows.append({
            "Fabric":fab,
            "Demand":lbl,
            "High Confidence":f"{conf.get('High',0):.0%}",
            "Price (₹/m)":f"₹{pr:,}",
            "Score":round(conf.get('High',0)*10,1)
        })
    comp_df = pd.DataFrame(comp_rows).sort_values("Score",ascending=False).reset_index(drop=True)
    comp_df.index = range(1, len(comp_df)+1)

    # Highlight active fabric
    def highlight_fab(row):
        if row["Fabric"] == st.session_state.fabric:
            return ["background-color:#1A2B1A"]*len(row)
        return [""]*len(row)

    st.dataframe(comp_df.style.apply(highlight_fab,axis=1),
                  use_container_width=True, height=320)

    st.markdown("---")

    # ── Final export section ───────────────────────
    st.markdown('<div class="card-title">📦 Export Design Package</div>',
                unsafe_allow_html=True)
    ex1,ex2,ex3 = st.columns(3)

    if tile_img:
        with ex1:
            st.download_button(
                "⬇️ Design Tile (PNG)",
                pil_to_bytes(tile_img),
                f"MYTH_AI_{st.session_state.fabric}_{st.session_state.season}_tile.png",
                "image/png", use_container_width=True, key="dl_tile2"
            )
        with ex2:
            if tiled_img:
                st.download_button(
                    "⬇️ Tiled Repeat (PNG)",
                    pil_to_bytes(tiled_img),
                    f"MYTH_AI_{st.session_state.fabric}_{st.session_state.season}"
                    f"_{st.session_state.repeat_mode.replace(' ','_')}.png",
                    "image/png", use_container_width=True, key="dl_tiled2"
                )
        with ex3:
            # Export palette PNG
            active_colors2 = list(dict.fromkeys(
                l["color"] for l in st.session_state.layers if l["visible"]
            ))[:8]
            pal_bytes = palette_swatch_image(active_colors2, w=600, h=100)
            st.download_button(
                "⬇️ Colour Palette (PNG)",
                pal_bytes,
                f"MYTH_AI_palette_{st.session_state.season}.png",
                "image/png", use_container_width=True, key="dl_pal2"
            )

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🧵 MYTH AI 2.0 — Professional Textile Design Studio<br>
  22-Layer Engine · Mirror / Half-Drop / Brick / Block Repeats ·
  Live Colorway Editor · AI Market Intelligence<br>
  Inspired by the Surat Textile Industry
</div>
""", unsafe_allow_html=True)
