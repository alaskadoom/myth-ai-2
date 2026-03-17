"""
╔══════════════════════════════════════════════════════╗
║         MYTH AI 2.0 — Streamlit Web App              ║
║   Profit + Design Intelligence Engine                ║
║   Inspired by the Surat Textile Industry             ║
╚══════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import math
import io
import os
from itertools import combinations

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="MYTH AI 2.0",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

  :root {
    --gold:    #C9A84C;
    --dark:    #0E0E1A;
    --card:    #16162A;
    --border:  #2A2A45;
    --text:    #E8E0D0;
    --muted:   #888899;
    --green:   #2ECC71;
    --yellow:  #F39C12;
    --red:     #E74C3C;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--dark);
    color: var(--text);
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }

  /* Hero header */
  .hero {
    background: linear-gradient(135deg, #0E0E1A 0%, #1A1030 50%, #0E1A1A 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(201,168,76,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: var(--gold);
    margin: 0 0 6px 0;
    letter-spacing: -1px;
  }
  .hero-sub {
    font-size: 1.05rem;
    color: var(--muted);
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.5px;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(201,168,76,0.15);
    border: 1px solid rgba(201,168,76,0.3);
    color: var(--gold);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
  }

  /* Cards */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 18px;
  }
  .card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: var(--gold);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* Demand badge */
  .demand-high   { background:#1a3d2b; border:1px solid #2ECC71; color:#2ECC71;
                   padding:6px 18px; border-radius:20px; font-weight:600;
                   font-size:1.1rem; display:inline-block; }
  .demand-medium { background:#3d2f0a; border:1px solid #F39C12; color:#F39C12;
                   padding:6px 18px; border-radius:20px; font-weight:600;
                   font-size:1.1rem; display:inline-block; }
  .demand-low    { background:#3d1010; border:1px solid #E74C3C; color:#E74C3C;
                   padding:6px 18px; border-radius:20px; font-weight:600;
                   font-size:1.1rem; display:inline-block; }

  /* Stat boxes */
  .stat-box {
    background: #0E0E1A;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }
  .stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: var(--gold);
    font-weight: 700;
  }
  .stat-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] label {
    color: var(--text) !important;
    font-size: 0.88rem !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #C9A84C, #A07830) !important;
    color: #0E0E1A !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(201,168,76,0.3) !important;
  }

  /* Selectbox */
  [data-testid="stSelectbox"] > div > div {
    background: #0E0E1A !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
  }

  /* Divider */
  hr { border-color: var(--border) !important; }

  /* Progress bars */
  .conf-bar-wrap { margin: 6px 0; }
  .conf-label { font-size: 0.82rem; color: var(--muted); margin-bottom: 3px; }
  .conf-bar-bg { background: #1A1A2E; border-radius: 4px; height: 10px; }
  .conf-bar-fill { height: 10px; border-radius: 4px; transition: width 0.5s ease; }

  /* Footer */
  .footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.78rem;
    padding: 28px 0 10px;
    border-top: 1px solid var(--border);
    margin-top: 40px;
    letter-spacing: 0.5px;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CORE ENGINE CLASSES (same logic as notebook, streamlit-adapted)
# ══════════════════════════════════════════════════════════════

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── Color Library ──────────────────────────────────────────────
COLOR_LIBRARY = {
    'deep_red'   : '#C0392B', 'gold'       : '#F1C40F',
    'saffron'    : '#FF6B35', 'coral'      : '#FF7F7F',
    'rust'       : '#B7410E', 'peach'      : '#FFDAB9',
    'maroon'     : '#800000', 'copper'     : '#B87333',
    'royal_blue' : '#2E4057', 'silver'     : '#BDC3C7',
    'teal'       : '#1ABC9C', 'navy'       : '#001F5B',
    'mint'       : '#A8E6CF', 'sky_blue'   : '#87CEEB',
    'cobalt'     : '#0047AB', 'aqua'       : '#00CED1',
    'ivory'      : '#FFFFF0', 'cream'      : '#FFFDD0',
    'beige'      : '#F5F0E1', 'white'      : '#FFFFFF',
    'charcoal'   : '#36454F', 'black'      : '#1C1C1C',
    'emerald'    : '#27AE60', 'purple'     : '#8E44AD',
    'magenta'    : '#FF00FF', 'rose_pink'  : '#FF66B2',
    'lime'       : '#32CD32', 'amber'      : '#FFBF00',
}

SEASON_PALETTES = {
    'wedding'  : ['deep_red','gold','maroon','ivory','rose_pink','copper'],
    'festive'  : ['saffron','gold','emerald','purple','coral','amber'],
    'casual'   : ['sky_blue','mint','peach','cream','coral','lime'],
    'summer'   : ['aqua','coral','mint','white','sky_blue','peach'],
    'winter'   : ['navy','maroon','charcoal','silver','cobalt','purple'],
    'monsoon'  : ['teal','royal_blue','emerald','silver','beige','aqua'],
}

SEASON_PATTERNS = {
    'wedding' : ['floral','paisley','embroidered'],
    'festive' : ['floral','ikat','geometric'],
    'casual'  : ['stripes','checks','abstract'],
    'summer'  : ['floral','stripes','abstract'],
    'winter'  : ['geometric','checks','paisley'],
    'monsoon' : ['ikat','abstract','stripes'],
}

FABRIC_TEXTURE = {
    'silk':'silk','velvet':'rough','linen':'linen','cotton':'linen',
    'polyester':'smooth','georgette':'silk','chiffon':'silk',
    'net':'smooth','rayon':'smooth','crepe':'rough',
}

MARKET_PRICES = {
    'premium'  : {'silk':2800,'velvet':2200,'georgette':900,'chiffon':700,
                  'cotton':400,'polyester':350,'linen':1200,'net':600,'rayon':500,'crepe':650},
    'mid_range': {'silk':1500,'velvet':1000,'georgette':550,'chiffon':400,
                  'cotton':250,'polyester':200,'linen':700,'net':350,'rayon':280,'crepe':380},
    'budget'   : {'silk':800,'velvet':500,'georgette':280,'chiffon':200,
                  'cotton':120,'polyester':90,'linen':380,'net':180,'rayon':150,'crepe':200},
    'export'   : {'silk':3500,'velvet':2800,'georgette':1100,'chiffon':900,
                  'cotton':550,'polyester':400,'linen':1500,'net':750,'rayon':600,'crepe':800},
}


# ── Helpers ────────────────────────────────────────────────────
def parse_color(color):
    named = COLOR_LIBRARY.get(color)
    hex_c = (named or color).lstrip('#')
    return tuple(int(hex_c[i:i+2], 16) for i in (0,2,4))

def luminance(hex_color):
    r,g,b = [x/255.0 for x in parse_color(hex_color.lstrip('#') if hex_color.startswith('#') else hex_color)]
    return 0.2126*r + 0.7152*g + 0.0722*b

def contrast_ratio(c1, c2):
    def lum(c):
        r,g,b = [x/255.0 for x in parse_color(c)]
        return 0.2126*r + 0.7152*g + 0.0722*b
    l1,l2 = lum(c1),lum(c2)
    li,da = max(l1,l2), min(l1,l2)
    return (li+0.05)/(da+0.05)


# ══════════════════════════════════════════════════════════════
# DATA + MODEL (cached so it only runs once)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🧵 Training AI model on textile market data...")
def build_model():
    """Generate data + train RandomForest. Cached after first run."""
    np.random.seed(SEED); random.seed(SEED)
    n = 2000

    fabric_types  = ['cotton','silk','polyester','georgette','chiffon','net','velvet','linen','rayon','crepe']
    color_combos  = ['red_gold','blue_silver','green_white','pink_peach','maroon_beige',
                     'black_red','yellow_orange','purple_pink','teal_gold','ivory_rust','navy_cream','coral_mint']
    pattern_types = ['floral','geometric','paisley','abstract','stripes','checks','embroidered','printed','plain','ikat']
    seasons       = ['wedding','festive','casual','summer','winter','monsoon']
    markets       = ['premium','mid_range','budget','export']

    fabrics  = np.random.choice(fabric_types,  n, p=[.18,.12,.20,.10,.08,.07,.05,.08,.07,.05])
    colors   = np.random.choice(color_combos,  n, p=[.12,.10,.09,.11,.10,.08,.08,.09,.09,.07,.04,.03])
    patterns = np.random.choice(pattern_types, n, p=[.18,.15,.14,.10,.10,.09,.08,.08,.05,.03])
    season   = np.random.choice(seasons,        n, p=[.22,.20,.18,.15,.13,.12])
    market   = np.random.choice(markets,        n, p=[.20,.35,.30,.15])

    base_p   = {'cotton':250,'silk':1800,'polyester':180,'georgette':600,'chiffon':500,
                'net':450,'velvet':1200,'linen':800,'rayon':300,'crepe':420}
    seas_m   = {'wedding':1.6,'festive':1.4,'casual':1.0,'summer':0.9,'winter':1.2,'monsoon':0.85}
    mkt_m    = {'premium':2.0,'mid_range':1.2,'budget':0.7,'export':1.5}

    prices = np.array([int(base_p[f]*seas_m[s]*mkt_m[m]*np.random.uniform(.85,1.15))
                       for f,s,m in zip(fabrics,season,market)])
    thread = np.random.randint(60,600,n)
    trend  = np.round(np.random.beta(2,2,n)*10,1)

    ds = np.zeros(n)
    ds += np.where(np.isin(fabrics,['silk','georgette','velvet']),2,0)
    ds += np.where(np.isin(season,['wedding','festive']),2,0)
    ds += np.where(np.isin(patterns,['floral','embroidered','paisley']),1,0)
    ds += np.where(np.isin(colors,['red_gold','maroon_beige','teal_gold']),1,0)
    ds += np.random.normal(0,1,n)

    sales = np.where(ds>=4,'high',np.where(ds>=2,'medium','low'))

    df = pd.DataFrame({'fabric_type':fabrics,'color_combo':colors,'pattern_type':patterns,
                       'season':season,'target_market':market,'price_inr':prices,
                       'thread_count':thread,'trend_score':trend,'sales_performance':sales})

    cat_cols = ['fabric_type','color_combo','pattern_type','season','target_market']
    feat_cols = cat_cols + ['price_inr','thread_count','trend_score']
    encoders = {}

    df_enc = df[feat_cols].copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    le_y = LabelEncoder()
    y    = le_y.fit_transform(df['sales_performance'])
    X    = df_enc.values

    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    clf = RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_leaf=5,
                                  random_state=SEED,class_weight='balanced')
    clf.fit(X_tr,y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    return clf, encoders, le_y, feat_cols, cat_cols, acc, df


def predict_demand(clf, encoders, le_y, feat_cols, cat_cols,
                   fabric, color, pattern, season, market, price):
    row = pd.DataFrame([{'fabric_type':fabric,'color_combo':color,'pattern_type':pattern,
                          'season':season,'target_market':market,'price_inr':price,
                          'thread_count':300,'trend_score':7.0}])
    df_enc = row[feat_cols].copy()
    for col in cat_cols:
        le = encoders[col]
        df_enc[col] = df_enc[col].astype(str).map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else 0)
    pred  = clf.predict(df_enc.values)[0]
    proba = clf.predict_proba(df_enc.values)[0]
    classes = le_y.classes_
    conf = {classes[i]: float(proba[i]) for i in range(len(classes))}
    return classes[pred], conf


# ══════════════════════════════════════════════════════════════
# DESIGN ENGINE
# ══════════════════════════════════════════════════════════════

def create_base_layer(size, bg_color, texture_style='smooth'):
    W,H = size
    img = Image.new('RGBA', size, (*parse_color(bg_color),255))
    draw = ImageDraw.Draw(img)
    if texture_style == 'linen':
        r,g,b = parse_color(bg_color)
        sh = tuple(max(0,c-15) for c in (r,g,b))
        for x in range(0,W,4): draw.line([(x,0),(x,H)],fill=(*sh,60),width=1)
        for y in range(0,H,4): draw.line([(0,y),(W,y)],fill=(*sh,40),width=1)
    elif texture_style == 'silk':
        r,g,b = parse_color(bg_color)
        for i in range(0,W+H,8):
            br = int(20*math.sin(i*0.12))
            sh = tuple(min(255,max(0,c+br)) for c in (r,g,b))
            draw.line([(i,0),(0,i)],fill=(*sh,80),width=2)
    elif texture_style == 'rough':
        r,g,b = parse_color(bg_color)
        np.random.seed(7)
        for _ in range(3000):
            x,y = np.random.randint(0,W),np.random.randint(0,H)
            n   = np.random.randint(-25,25)
            c   = tuple(min(255,max(0,v+n)) for v in (r,g,b))
            draw.ellipse([x-1,y-1,x+1,y+1],fill=(*c,120))
    return img


def create_pattern_layer(size, pattern_type, pc_name, sc_name):
    W,H = size
    layer = Image.new('RGBA',size,(0,0,0,0))
    draw  = ImageDraw.Draw(layer)
    pc = (*parse_color(pc_name),200)
    sc = (*parse_color(sc_name),160)

    if pattern_type == 'floral':
        for cx in range(40,W,80):
            for cy in range(40,H,80):
                for angle in range(0,360,72):
                    rad = math.radians(angle)
                    px,py = cx+int(22*math.cos(rad)), cy+int(22*math.sin(rad))
                    draw.ellipse([px-14,py-8,px+14,py+8],fill=pc)
                draw.ellipse([cx-12,cy-12,cx+12,cy+12],fill=sc)
                draw.ellipse([cx-6,cy-6,cx+6,cy+6],fill=pc)
    elif pattern_type == 'geometric':
        step=64
        for row,cy in enumerate(range(0,H,step)):
            off = step//2 if row%2 else 0
            for cx in range(off,W,step):
                col = pc if (cx//step+row)%2==0 else sc
                draw.polygon([(cx,cy+step//2),(cx+step//2,cy),(cx+step,cy+step//2),(cx+step//2,cy+step)],
                             fill=col,outline=(255,255,255,80))
    elif pattern_type in ('paisley','embroidered'):
        for row,cy in enumerate(range(30,H,100)):
            off = 50 if row%2 else 0
            for cx in range(off,W,100):
                draw.ellipse([cx-20,cy-30,cx+20,cy+30],fill=pc,outline=sc)
                draw.ellipse([cx-8,cy+10,cx+8,cy+30],fill=sc)
                draw.ellipse([cx-5,cy-10,cx+5,cy+10],fill=(255,255,255,120))
    elif pattern_type == 'stripes':
        sw=32
        for x in range(0,W,sw*2):
            draw.rectangle([x,0,x+sw,H],fill=pc)
            draw.rectangle([x+sw,0,x+sw*2,H],fill=sc)
    elif pattern_type == 'checks':
        cell=48
        for row in range(0,H,cell):
            for col in range(0,W,cell):
                c = pc if (row//cell+col//cell)%2==0 else sc
                draw.rectangle([col,row,col+cell,row+cell],fill=c)
    elif pattern_type == 'ikat':
        for d in range(-H,W,48):
            col = pc if (d//48)%2==0 else sc
            for off in range(-8,8,2):
                o = off + random.randint(-3,3)
                draw.line([(d+o,0),(d+o+H,H)],fill=col,width=3)
    else:
        np.random.seed(12)
        margin = max(10, min(50, W//6))
        for _ in range(30):
            x,y = np.random.randint(margin,max(margin+1,W-margin)),np.random.randint(margin,max(margin+1,H-margin))
            r   = np.random.randint(5,max(6,min(70,W//4)))
            c   = pc if np.random.rand()>.4 else sc
            draw.ellipse([x-r,y-r,x+r,y+r],fill=c,outline=(255,255,255,60))
    return layer


def create_detail_layer(size, detail_color, density='medium'):
    W,H = size
    layer = Image.new('RGBA',size,(0,0,0,0))
    draw  = ImageDraw.Draw(layer)
    dc    = (*parse_color(detail_color),200)
    n_m   = {'low':30,'medium':80,'high':160}.get(density,80)
    np.random.seed(77)
    for _ in range(n_m):
        x,y  = np.random.randint(10,W-10), np.random.randint(10,H-10)
        kind = np.random.choice(['dot','star','diamond','cross'])
        if kind=='dot':
            r=np.random.randint(3,7); draw.ellipse([x-r,y-r,x+r,y+r],fill=dc)
        elif kind=='star':
            pts=[]
            for i in range(10):
                a=math.radians(i*36-90); rad=8 if i%2==0 else 4
                pts.append((x+int(rad*math.cos(a)),y+int(rad*math.sin(a))))
            draw.polygon(pts,fill=dc)
        elif kind=='diamond':
            s=np.random.randint(5,10); draw.polygon([(x,y-s),(x+s,y),(x,y+s),(x-s,y)],fill=dc)
        else:
            t,s=2,np.random.randint(5,9)
            draw.rectangle([x-t,y-s,x+t,y+s],fill=dc)
            draw.rectangle([x-s,y-t,x+s,y+t],fill=dc)
    return layer


def create_accent_layer(size, accent_color, style='border'):
    W,H = size
    layer = Image.new('RGBA',size,(0,0,0,0))
    draw  = ImageDraw.Draw(layer)
    ac    = (*parse_color(accent_color),180)
    if style=='border':
        b = max(2, min(12, W//20))
        draw.rectangle([0,0,W,b],fill=ac); draw.rectangle([0,H-b,W,H],fill=ac)
        draw.rectangle([0,0,b,H],fill=ac); draw.rectangle([W-b,0,W,H],fill=ac)
        if W > b*3: draw.rectangle([b,b,W-b,min(b+4,H-b)],fill=ac)
        if H > b*3: draw.rectangle([b,max(b,H-b-4),W-b,H-b],fill=ac)
    elif style=='shimmer':
        r,g,b_v=parse_color(accent_color); np.random.seed(33)
        for _ in range(200):
            x,y=np.random.randint(0,W),np.random.randint(0,H)
            s=np.random.randint(1,5); a=np.random.randint(80,200)
            draw.ellipse([x-s,y-s,x+s,y+s],fill=(r,g,b_v,a))
    elif style=='corner':
        for cx,cy in [(0,0),(W,0),(0,H),(W,H)]:
            for i in range(3):
                r=60-i*12
                x0=cx-r if cx>0 else cx; y0=cy-r if cy>0 else cy
                x1=cx+r if cx==0 else cx; y1=cy+r if cy==0 else cy
                draw.ellipse([min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)],outline=ac,width=2)
    elif style=='vignette':
        max_rings = min(60, W//8, H//8)
        for ring in range(max_rings):
            a=int(ring*2); m=ring*4
            if m < W//2 and m < H//2:
                draw.rectangle([m,m,max(m+1,W-m),max(m+1,H-m)],outline=(0,0,0,a),width=2)
    return layer


def compose_design(bg, pattern, pc, sc, dc, ac, texture, density, accent_style, size=(512,512)):
    base    = create_base_layer(size, bg, texture)
    pat     = create_pattern_layer(size, pattern, pc, sc)
    detail  = create_detail_layer(size, dc, density)
    accent  = create_accent_layer(size, ac, accent_style)
    result  = Image.alpha_composite(base, pat)
    result  = Image.alpha_composite(result, detail)
    result  = Image.alpha_composite(result, accent)
    return result


def make_seamless(tile_img, blend_width=40):
    W,H = tile_img.size
    arr = np.array(tile_img.convert('RGBA'), dtype=np.float32)
    arr_s = np.roll(np.roll(arr,H//2,axis=0),W//2,axis=1)
    mask = np.zeros((H,W),dtype=np.float32)
    bw = blend_width
    for i in range(bw):
        a = (i/bw)**0.7
        if H//2-bw+i < H: mask[H//2-bw+i,:] = a
        if H//2+bw-i-1 < H: mask[H//2+bw-i-1,:] = a
        if W//2-bw+i < W: mask[:,W//2-bw+i] = np.maximum(mask[:,W//2-bw+i],a)
        if W//2+bw-i-1 < W: mask[:,W//2+bw-i-1] = np.maximum(mask[:,W//2+bw-i-1],a)
    mask_img   = Image.fromarray((mask*255).astype(np.uint8),'L')
    mask_img   = mask_img.filter(ImageFilter.GaussianBlur(radius=bw//3))
    mask_s     = np.array(mask_img,dtype=np.float32)/255.0
    mask4      = np.stack([mask_s]*4,axis=2)
    blended    = np.clip(arr*(1-mask4)+arr_s*mask4,0,255).astype(np.uint8)
    return Image.fromarray(blended,'RGBA')


def tile_image(tile_img, rows=2, cols=3):
    tile  = make_seamless(tile_img)
    W,H   = tile.size
    canvas= Image.new('RGBA',(W*cols,H*rows),(255,255,255,255))
    for r in range(rows):
        for c in range(cols):
            canvas.paste(tile,(c*W,r*H))
    return canvas


def suggest_palettes(season, n=5):
    base   = SEASON_PALETTES.get(season, list(COLOR_LIBRARY.keys())[:6])
    all_c  = list(COLOR_LIBRARY.keys())
    scored = []
    random.seed(SEED)
    for _ in range(200):
        nb = random.randint(2,min(3,len(base)))
        na = 4-nb
        pal = list(dict.fromkeys(random.sample(base,nb)+random.sample(all_c,na)))[:4]
        if len(pal)<3: continue
        hexes = [COLOR_LIBRARY.get(c,'#888') for c in pal]
        pairs = list(combinations(hexes,2))
        if not pairs: continue
        avg_cr = np.mean([contrast_ratio(a,b) for a,b in pairs])
        score  = round(min(avg_cr/5.0,1.0)*10,2)
        scored.append((pal,score))
    scored.sort(key=lambda x:x[1],reverse=True)
    seen,out=[],[]
    for p,s in scored:
        k=frozenset(p)
        if k not in seen: seen.append(k); out.append((p,s))
        if len(out)>=n: break
    return out


def pil_to_bytes(img):
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='PNG')
    return buf.getvalue()


def render_palette_image(palette, width=500, height=80):
    fig,ax = plt.subplots(figsize=(width/100,height/100))
    fig.patch.set_facecolor('#16162A')
    ax.set_facecolor('#16162A')
    n = len(palette)
    for i,cname in enumerate(palette):
        hex_c = COLOR_LIBRARY.get(cname,'#888')
        rect  = mpatches.FancyBboxPatch(
            (i/n+0.005,0.05), 0.95/n, 0.85,
            boxstyle='round,pad=0.02',
            facecolor=hex_c, edgecolor='#0E0E1A', linewidth=1.5
        )
        ax.add_patch(rect)
        lum = luminance(hex_c)
        tc  = 'black' if lum>0.35 else 'white'
        ax.text(i/n+0.5/n, 0.5, cname.replace('_',' ').title(),
                ha='center',va='center',fontsize=7,color=tc,fontweight='bold')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf,format='png',dpi=120,bbox_inches='tight',facecolor='#16162A')
    plt.close()
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════

# Load model once
clf, encoders, le_y, feat_cols, cat_cols, model_acc, train_df = build_model()

# ── HERO ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">✦ Surat Textile Industry ✦</div>
  <div class="hero-title">🧵 MYTH AI 2.0</div>
  <p class="hero-sub">Profit + Design Intelligence Engine — AI-powered textile design, demand forecasting & color intelligence</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 20px;'>
      <div style='font-family: Playfair Display, serif; font-size:1.3rem;
                  color:#C9A84C; font-weight:700;'>Design Studio</div>
      <div style='color:#888899; font-size:0.78rem; margin-top:4px;'>
        Configure your textile design
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🧵 Fabric & Market**")

    fabric = st.selectbox("Fabric Type", [
        'silk','georgette','cotton','polyester','velvet',
        'chiffon','linen','rayon','crepe','net'
    ], index=0)

    market = st.selectbox("Target Market", [
        'premium','mid_range','budget','export'
    ], index=0)

    st.markdown("---")
    st.markdown("**🌸 Season & Style**")

    season = st.selectbox("Season / Occasion", [
        'wedding','festive','casual','summer','winter','monsoon'
    ], index=0)

    pattern_options = SEASON_PATTERNS.get(season, ['floral'])
    all_patterns    = ['floral','geometric','paisley','stripes','checks','ikat','abstract','embroidered']
    pattern_choice  = st.selectbox("Pattern Type",
                                    all_patterns,
                                    index=all_patterns.index(pattern_options[0])
                                    if pattern_options[0] in all_patterns else 0)

    accent_style = st.selectbox("Accent Style", ['border','shimmer','corner','vignette'], index=0)

    st.markdown("---")
    st.markdown("**🖼️ Output Size**")
    design_size = st.selectbox("Design Resolution", ['512×512','768×768','1024×1024'], index=0)
    size_map    = {'512×512':(512,512),'768×768':(768,768),'1024×1024':(1024,1024)}
    sz          = size_map[design_size]

    st.markdown("---")

    generate_btn = st.button("🚀 Generate Design", type="primary")

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#888899; text-align:center;'>
      Model Accuracy<br>
      <span style='color:#C9A84C; font-size:1.1rem; font-weight:700;'>{model_acc:.0%}</span>
      <br><span style='font-size:0.7rem;'>on {len(train_df):,} samples</span>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN AREA ──────────────────────────────────────────────────

if not generate_btn:
    # Show welcome state with stats
    st.markdown("### 📊 Market Overview")

    c1,c2,c3,c4 = st.columns(4)
    fabric_count   = train_df['fabric_type'].nunique()
    high_pct       = (train_df['sales_performance']=='high').mean()
    avg_price      = train_df['price_inr'].mean()
    top_season     = train_df.groupby('season')['trend_score'].mean().idxmax()

    with c1:
        st.markdown(f"""
        <div class="stat-box">
          <div class="stat-value">{fabric_count}</div>
          <div class="stat-label">Fabric Types</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-box">
          <div class="stat-value">{high_pct:.0%}</div>
          <div class="stat-label">High Demand Rate</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-box">
          <div class="stat-value">₹{avg_price:,.0f}</div>
          <div class="stat-label">Avg Market Price</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-box">
          <div class="stat-value">{top_season.title()}</div>
          <div class="stat-label">Top Season</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Sales Performance Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3))
        fig.patch.set_facecolor('#16162A'); ax.set_facecolor('#16162A')
        sp = train_df['sales_performance'].value_counts()
        bars = ax.bar(sp.index, sp.values,
                      color=['#2ECC71','#F39C12','#E74C3C'],
                      edgecolor='#0E0E1A', linewidth=1.5, width=0.55)
        ax.set_facecolor('#16162A')
        for spine in ax.spines.values(): spine.set_edgecolor('#2A2A45')
        ax.tick_params(colors='#888899'); ax.yaxis.label.set_color('#888899')
        for bar,val in zip(bars,sp.values):
            ax.text(bar.get_x()+bar.get_width()/2, val+8, str(val),
                    ha='center', color='white', fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🧵 Avg Price by Fabric</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5,3))
        fig.patch.set_facecolor('#16162A'); ax.set_facecolor('#16162A')
        avg_by_fabric = train_df.groupby('fabric_type')['price_inr'].mean().sort_values()
        colors_fab = plt.cm.YlOrBr(np.linspace(0.3,0.9,len(avg_by_fabric)))
        ax.barh(avg_by_fabric.index, avg_by_fabric.values,
                color=colors_fab, edgecolor='#0E0E1A', linewidth=0.5)
        for spine in ax.spines.values(): spine.set_edgecolor('#2A2A45')
        ax.tick_params(colors='#888899')
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; padding:32px 0; color:#444466;'>
      <div style='font-size:2rem;'>←</div>
      <div style='font-size:0.9rem; margin-top:8px;'>
        Configure your design in the sidebar and click <strong style='color:#C9A84C;'>Generate Design</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ═══════════════ GENERATION ═══════════════
    with st.spinner("🧵 Generating your textile design..."):

        # Get palette
        palettes    = suggest_palettes(season, n=5)
        best_pal, harmony_score = palettes[0]
        bg_c   = best_pal[0]
        pc     = best_pal[1] if len(best_pal)>1 else 'gold'
        sc     = best_pal[2] if len(best_pal)>2 else 'ivory'
        dc     = best_pal[3] if len(best_pal)>3 else pc
        ac     = pc
        texture = FABRIC_TEXTURE.get(fabric,'smooth')
        density = 'high' if season in ['wedding','festive'] else 'medium'
        price   = MARKET_PRICES[market].get(fabric,500)

        # Demand prediction
        d_color = 'red_gold' if season in ['wedding','festive'] else 'blue_silver'
        demand_label, demand_conf = predict_demand(
            clf, encoders, le_y, feat_cols, cat_cols,
            fabric, d_color, pattern_choice, season, market, price
        )

        # Generate design
        tile    = compose_design(bg_c, pattern_choice, pc, sc, dc, ac,
                                  texture, density, accent_style, sz)
        tiled   = tile_image(tile, rows=2, cols=3)

    # ── Results layout ──────────────────────────────────────────
    st.markdown("### 🎨 Generated Textile Design")

    top_col1, top_col2 = st.columns([1,1])

    with top_col1:
        st.markdown('<div class="card" style="padding:12px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🖼️ Design Tile (Single Repeat)</div>', unsafe_allow_html=True)
        st.image(pil_to_bytes(tile), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with top_col2:
        st.markdown('<div class="card" style="padding:12px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔁 Seamless Tiled Preview (3×2)</div>', unsafe_allow_html=True)
        st.image(pil_to_bytes(tiled), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bottom row ──────────────────────────────────────────────
    bot1, bot2, bot3 = st.columns([1.4, 1, 1])

    # Palette panel
    with bot1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎨 Top 5 Color Palettes</div>', unsafe_allow_html=True)
        for i, (pal, score) in enumerate(palettes):
            label = f"#{i+1}  Harmony: {score}/10"
            if i == 0: label += "  ✦ BEST"
            st.caption(label)
            st.image(render_palette_image(pal), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Demand panel
    with bot2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Demand Prediction</div>', unsafe_allow_html=True)

        d_emoji = {'high':'🟢','medium':'🟡','low':'🔴'}.get(demand_label,'⚪')
        d_class = f"demand-{demand_label}"
        st.markdown(f"""
        <div style='text-align:center; margin:12px 0 20px;'>
          <div class='{d_class}'>{d_emoji} {demand_label.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        for label, val in demand_conf.items():
            pct = int(val*100)
            bar_color = '#2ECC71' if label=='high' else '#F39C12' if label=='medium' else '#E74C3C'
            st.markdown(f"""
            <div class='conf-bar-wrap'>
              <div class='conf-label'>{label.title()} — {pct}%</div>
              <div class='conf-bar-bg'>
                <div class='conf-bar-fill' style='width:{pct}%; background:{bar_color};'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence pie
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        fig.patch.set_facecolor('#16162A'); ax.set_facecolor('#16162A')
        sizes  = [demand_conf.get(k,0) for k in ['high','medium','low']]
        colors_d = ['#2ECC71','#F39C12','#E74C3C']
        wedges,_ = ax.pie(sizes, colors=colors_d, startangle=90,
                          wedgeprops=dict(edgecolor='#0E0E1A',linewidth=2))
        ax.legend(wedges,['High','Medium','Low'], loc='lower center',
                  frameon=False, fontsize=8,
                  labelcolor='#888899', ncol=3,
                  bbox_to_anchor=(0.5,-0.08))
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Specs panel
    with bot3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Design Specifications</div>', unsafe_allow_html=True)

        specs = [
            ("🧵 Fabric",    fabric.title()),
            ("🌸 Season",    season.title()),
            ("🏪 Market",    market.replace('_',' ').title()),
            ("🎨 Pattern",   pattern_choice.title()),
            ("✨ Texture",   texture.title()),
            ("💰 Price/m",  f"₹{price:,}"),
            ("🔮 Demand",   f"{d_emoji} {demand_label.upper()}"),
            ("🎨 Harmony",  f"{harmony_score}/10"),
            ("📐 Size",     design_size),
        ]
        for label, value in specs:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between;
                        padding:7px 0; border-bottom:1px solid #2A2A45;'>
              <span style='color:#888899; font-size:0.83rem;'>{label}</span>
              <span style='color:#E8E0D0; font-size:0.83rem; font-weight:500;'>{value}</span>
            </div>
            """, unsafe_allow_html=True)

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        tile_bytes = pil_to_bytes(tile)
        st.download_button(
            label="⬇️ Download Design Tile",
            data=tile_bytes,
            file_name=f"MYTH_AI_{fabric}_{season}_{pattern_choice}.png",
            mime="image/png",
            use_container_width=True
        )
        tiled_bytes = pil_to_bytes(tiled)
        st.download_button(
            label="⬇️ Download Tiled Repeat",
            data=tiled_bytes,
            file_name=f"MYTH_AI_{fabric}_{season}_tiled.png",
            mime="image/png",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ─────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🧵 MYTH AI 2.0 &nbsp;·&nbsp; Profit + Design Intelligence Engine<br>
  Inspired by the Surat Textile Industry &nbsp;·&nbsp;
  AI-Powered · Data-Driven · Market-Optimized
</div>
""", unsafe_allow_html=True)
