#!/usr/bin/env python3
"""Generate the GoiBERT-M architecture figure as a draw.io file (single page)."""
import html, sys

OUT = sys.argv[1]
cells = []
origin = {"1": (0, 0)}
_n = [1]
BOXES = []  # (id, parent, x, y, w, h) for overlap checks


def esc(s):
    return html.escape(s, quote=True)


def nid():
    _n[0] += 1
    return f"n{_n[0]}"


FONT = "fontFamily=Helvetica;"
PAL = {
    "emb":  "fillColor=#DCE6F2;strokeColor=#5B7FA6;",
    "attn": "fillColor=#F9E3D2;strokeColor=#C0773F;",
    "ffn":  "fillColor=#DDEEDD;strokeColor=#5E9461;",
    "norm": "fillColor=#EDEDED;strokeColor=#7F7F7F;",
    "head": "fillColor=#E8DDF0;strokeColor=#8663A3;",
    "io":   "fillColor=#FFFFFF;strokeColor=#444444;",
    "layer": "fillColor=#FFFFFF;strokeColor=#3B4B63;",
}
BOX = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;strokeWidth=1.2;fontSize=11;fontColor=#1A1A1A;" + FONT


def node(label, x, y, w, h, style, parent="1", ident=None, check=True):
    ident = ident or nid()
    ox, oy = origin[parent]
    origin[ident] = (x, y)
    cells.append(
        f'<mxCell id="{ident}" value="{esc(label)}" style="{style}" vertex="1" parent="{parent}">'
        f'<mxGeometry x="{x-ox}" y="{y-oy}" width="{w}" height="{h}" as="geometry"/></mxCell>'
    )
    if check:
        BOXES.append((ident, parent, x, y, w, h, label))
    return ident


def box(label, x, y, w, h, kind, parent="1", extra="", ident=None):
    return node(label, x, y, w, h, BOX + PAL[kind] + extra, parent, ident)


def text(label, x, y, w, h, parent="1", size=10, align="left", valign="top", color="#1A1A1A", extra=""):
    st = (f"text;html=1;whiteSpace=wrap;overflow=hidden;fontSize={size};fontColor={color};align={align};"
          f"verticalAlign={valign};spacing=2;" + FONT + extra)
    return node(label, x, y, w, h, st, parent)


def panel(label, x, y, w, h, ident=None):
    st = ("rounded=1;arcSize=2;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#B5B5B5;strokeWidth=1;"
          "verticalAlign=top;align=left;spacingLeft=10;spacingTop=5;fontSize=12;fontStyle=1;fontColor=#1A1A1A;" + FONT)
    return node(label, x, y, w, h, st, "1", ident, check=False)


def edge(src, tgt, exit=(0.5, 1), entry=(0.5, 0), points=None, dashed=False, label=None, arrow="block", color="#333333", width=1.2, straight=False):
    eid = nid()
    st = (f"edgeStyle={'none' if straight else 'orthogonalEdgeStyle'};rounded=0;html=1;endArrow={arrow};endFill=1;strokeColor={color};strokeWidth={width};"
          f"exitX={exit[0]};exitY={exit[1]};entryX={entry[0]};entryY={entry[1]};jettySize=auto;fontSize=10;" + FONT)
    if dashed:
        st += "dashed=1;dashPattern=4 3;"
    pts = ""
    if points:
        pts = '<Array as="points">' + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points) + "</Array>"
    cells.append(
        f'<mxCell id="{eid}" value="{esc(label or "")}" style="{st}" edge="1" parent="1" source="{src}" target="{tgt}">'
        f'<mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>'
    )
    return eid




# ------------------------------------------------------------------ geometry (A4 portrait: 827 x 1169)
RX, RW = 290, 517            # right column (zoom panels)
LX, LW = 20, 250             # left column (main pipeline)
text("<b>GoiBERT-M</b>: a 12-layer, 512-d Transformer encoder (49.4M parameters)", 20, 5, 787, 26, size=15, valign="middle")
text("Post-LN RoBERTa encoder · BamiBERT tokenizer (|V| = 20,481) · absolute positions · trained on 256-token sequences",
     20, 30, 787, 16, size=9, color="#555555", valign="middle")

B_Y, B_H = 58, 230
C_Y, C_H = B_Y + B_H + 10, 482
D_Y, D_H = C_Y + C_H + 10, 290
F_Y = D_Y + D_H + 10

# ------------------------------------------------------------------ left column: main pipeline
b1 = box("Raw text (Vietnamese, unsegmented)", LX, 62, LW, 32, "io")
b2 = box("BamiBERT tokenizer · |V| = 20,481", LX, 108, LW, 32, "io")
b3 = box("Token ids: &lt;s&gt; <i>x</i><sub>1</sub> … <i>x</i><sub>n</sub> &lt;/s&gt;<br><i>n</i> ≤ 254 (max length 256)", LX, 154, LW, 46, "io")
b4 = box("<b>Embedding layer</b><br>11.54M parameters", LX, 214, LW, 44, "emb")
enc = node("Transformer encoder × 12<br>37.83M parameters (3.15M per layer)", LX, 284, LW, 366,
           "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#F4F6F9;strokeColor=#3B4B63;strokeWidth=1.4;"
           "verticalAlign=bottom;spacingBottom=10;fontSize=11;fontStyle=1;fontColor=#1A1A1A;" + FONT)
l1 = box("Layer 1", LX + 40, 318, 170, 62, "layer", enc)
l2 = box("Layer 2", LX + 40, 398, 170, 62, "layer", enc)
ld = text("⋮", LX + 40, 470, 170, 34, enc, size=22, align="center", valign="middle")
l12 = box("Layer 12", LX + 40, 520, 170, 62, "layer", enc)
b6 = box("<i>H</i><sup>12</sup> ∈ ℝ<sup><i>n</i>×512</sup><br>contextual representations", LX, 676, LW, 44, "io")
b7 = box("<b>Task head</b><br>added at fine-tuning<br>0.01–0.25M parameters", LX, 812, LW, 62, "head")
edge(b1, b2); edge(b2, b3); edge(b3, b4); edge(b4, enc)
edge(l1, l2); edge(l2, ld, arrow="none"); edge(ld, l12)
edge(enc, b6); edge(b6, b7)
# legend
text("<b>Legend</b>", LX, 906, 120, 18, size=11)
ly = 930
for name, kind in [("Embedding", "emb"), ("Attention", "attn"), ("Feed-forward", "ffn"), ("Norm / dropout", "norm"), ("Task head", "head")]:
    node("", LX + 4, ly, 14, 14, "rounded=0;html=1;strokeWidth=1.2;" + PAL[kind], "1", check=False)
    text(name, LX + 26, ly - 3, 200, 20, size=11, valign="middle")
    ly += 26

# ------------------------------------------------------------------ (b) embedding layer
pb = panel("(b) Embedding layer", RX, B_Y, RW, B_H)
cxm = RX + RW // 2   # 548
i1 = box("<i>input_ids</i>", 303, B_Y + 32, 150, 20, "io", pb, "fontSize=10;dashed=1;")
i2 = box("<i>position_ids</i>", 473, B_Y + 32, 150, 20, "io", pb, "fontSize=10;dashed=1;")
i3 = box("<i>token_type_ids</i>", 643, B_Y + 32, 150, 20, "io", pb, "fontSize=10;dashed=1;")
e1 = box("Word embedding <i>E</i><sub>w</sub><br>20,481 × 512 · 10.49M", 303, B_Y + 62, 150, 44, "emb", pb)
e2 = box("Position embedding <i>E</i><sub>p</sub><br>2,050 × 512 · 1.05M", 473, B_Y + 62, 150, 44, "emb", pb)
e3 = box("Token-type <i>E</i><sub>t</sub><br>1 × 512 · 0.5K", 643, B_Y + 62, 150, 44, "emb", pb)
edge(i1, e1); edge(i2, e2); edge(i3, e3)
sm = node("+", cxm - 12, B_Y + 120, 24, 24,
          "ellipse;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#444444;strokeWidth=1.2;fontSize=14;fontStyle=1;" + FONT, pb)
edge(e1, sm, (0.5, 1), (0, 0.5)); edge(e2, sm, (0.5, 1), (0.5, 0)); edge(e3, sm, (0.5, 1), (1, 0.5))
lndp = box("LayerNorm (ε = 1e-5)<br>Dropout (<i>p</i> = 0.1)", cxm - 100, B_Y + 154, 200, 34, "norm", pb)
h0 = box("<i>H</i><sup>0</sup> ∈ ℝ<sup><i>n</i>×512</sup>", cxm - 100, B_Y + 198, 200, 24, "io", pb)
edge(sm, lndp); edge(lndp, h0)
text("<i>E</i><sub>w</sub> initialised by PCA (768 → 512) of BamiBERT's word embeddings.<br>11,537,408 parameters in total.",
     299, B_Y + 150, 142, 74, pb, size=9)
text("<i>position_ids</i> start at 2 (padding_idx = 1). The table has 2,050 rows; only positions ≤ 256 were trained.",
     657, B_Y + 150, 144, 74, pb, size=9)

# ------------------------------------------------------------------ (c) transformer layer
pc = panel("(c) Transformer encoder layer ℓ (post-LN), ℓ = 1 … 12", RX, C_Y, RW, C_H)
o = C_Y
cin = box("<i>x</i><sup>(ℓ−1)</sup> ∈ ℝ<sup><i>n</i>×512</sup>", cxm - 110, o + 34, 220, 24, "io", pc)
mha = node("Multi-head self-attention (<i>h</i> = 8, <i>d</i><sub>k</sub> = 64)", cxm - 145, o + 72, 290, 150,
           "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#FDF3EA;strokeColor=#C0773F;strokeWidth=1.2;"
           "verticalAlign=bottom;spacingBottom=4;fontSize=11;fontStyle=1;fontColor=#1A1A1A;" + FONT, pc)
qx = cxm - 139
q = box("<i>W</i><sub>Q</sub><br>512 → 512", qx, o + 82, 86, 34, "attn", mha)
k = box("<i>W</i><sub>K</sub><br>512 → 512", qx + 96, o + 82, 86, 34, "attn", mha)
v = box("<i>W</i><sub>V</sub><br>512 → 512", qx + 192, o + 82, 86, 34, "attn", mha)
att = box("8 heads: softmax(<i>Q</i><sub>h</sub><i>K</i><sub>h</sub><sup>T</sup> / √64 + <i>M</i>) <i>V</i><sub>h</sub>", qx, o + 122, 278, 34, "attn", mha)
wo = box("Concat(head<sub>1</sub> … head<sub>8</sub>) → <i>W</i><sub>O</sub> (512 → 512)", qx, o + 162, 278, 28, "attn", mha)
edge(cin, q, (0.2, 1), (0.5, 0)); edge(cin, k, (0.5, 1), (0.5, 0)); edge(cin, v, (0.8, 1), (0.5, 0))
edge(q, att, (0.5, 1), (0.17, 0)); edge(k, att, (0.5, 1), (0.5, 0)); edge(v, att, (0.5, 1), (0.83, 0))
edge(att, wo)
an1 = box("Dropout (0.1)<br>Add &amp; LayerNorm (ε = 1e-5)", cxm - 110, o + 232, 220, 32, "norm", pc)
edge(mha, an1)
ffn = node("Position-wise feed-forward network", cxm - 145, o + 276, 290, 118,
           "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#F1F8F1;strokeColor=#5E9461;strokeWidth=1.2;"
           "verticalAlign=bottom;spacingBottom=4;fontSize=11;fontStyle=1;fontColor=#1A1A1A;" + FONT, pc)
f1 = box("Linear 512 → 2,048", qx, o + 286, 278, 26, "ffn", ffn)
f2 = box("GELU", qx, o + 318, 278, 20, "ffn", ffn)
f3 = box("Linear 2,048 → 512", qx, o + 344, 278, 26, "ffn", ffn)
edge(f1, f2); edge(f2, f3); edge(an1, f1)
an2 = box("Dropout (0.1)<br>Add &amp; LayerNorm (ε = 1e-5)", cxm - 110, o + 404, 220, 32, "norm", pc)
edge(ffn, an2)
cout = box("<i>x</i><sup>(ℓ)</sup> ∈ ℝ<sup><i>n</i>×512</sup>", cxm - 110, o + 448, 220, 24, "io", pc)
edge(an2, cout)
r1x, r2x = cxm + 146 + 22, cxm + 146 + 2   # right of the 290-wide containers
edge(cin, an1, (1, 0.5), (1, 0.25), points=[(r1x, o + 46), (r1x, o + 240)])
edge(an1, an2, (1, 0.75), (1, 0.5), points=[(r2x, o + 256), (r2x, o + 420)])
text("residual", r1x + 4, o + 120, 48, 16, pc, size=9, color="#555555", extra="fontStyle=2;")
text("residual", r2x + 4, o + 330, 48, 16, pc, size=9, color="#555555", extra="fontStyle=2;")
text("<i>M</i>: additive padding mask", RX + 8, o + 232, 96, 30, pc, size=9, color="#555555")

# ------------------------------------------------------------------ (d) task heads
pd = panel("(d) Task heads (added at fine-tuning only)", RX, D_Y, RW, D_H)
o = D_Y
hin = box("<i>H</i><sup>12</sup> ∈ ℝ<sup><i>n</i>×512</sup>", RX + 14, o + 34, RW - 28, 26, "io", pd)
lx0, lw = RX + 14, 224
d1 = [
    ("[CLS] vector<br><i>h</i> = <i>H</i><sup>12</sup>[0] ∈ ℝ<sup>512</sup>", "io", 78, 36),
    ("Dropout (0.1) · Linear 512 → 384", "head", 124, 28),
    ("tanh · Dropout (0.1)", "head", 162, 28),
    ("Linear 384 → <i>K</i>", "head", 200, 28),
    ("logits ∈ ℝ<sup><i>K</i></sup>", "io", 238, 28),
]
prev = None
for lab, kd, dy, hh in d1:
    cur = box(lab, lx0, o + dy, lw, hh, kd, pd)
    edge(prev, cur) if prev else edge(hin, cur, ((lx0 + lw / 2 - (RX + 14)) / (RW - 28), 1), (0.5, 0))
    prev = cur
rx0 = RX + RW - 14 - lw
d2 = [("Dropout (0.1) · Linear 512 → 21", "head", 78, 28), ("per-token logits<br>ℝ<sup><i>n</i>×21</sup>", "io", 124, 40)]
prev = None
for lab, kd, dy, hh in d2:
    cur = box(lab, rx0, o + dy, lw, hh, kd, pd)
    edge(prev, cur) if prev else edge(hin, cur, ((rx0 + lw / 2 - (RX + 14)) / (RW - 28), 1), (0.5, 0))
    prev = cur
text("<b>Token-level head</b> (NER)<br>21 BIO labels, 10,773 parameters", rx0, o + 176, lw, 34, pd, size=10, align="center")
text("<b>Sequence-level head</b> (left)<br>classification, ABSA: <i>K</i> = Σ<sub>a</sub> <i>C</i><sub>a</sub><br>196,992 + 385·<i>K</i> parameters",
     rx0, o + 226, lw, 50, pd, size=10, align="center")

# zoom connectors (horizontal)
def zoom(src, src_y, panel_id, py, ph, ty):
    edge(src, panel_id, (1, (src_y[0] / src_y[1])), (0, (ty - py) / ph), dashed=True, arrow="open", color="#7F7F7F", width=1, straight=True)
zoom(b4, (22, 44), pb, B_Y, B_H, 236)
zoom(enc, (183, 366), pc, C_Y, C_H, 467)
zoom(b7, (31, 62), pd, D_Y, D_H, 843)

# ------------------------------------------------------------------ footer
pe = panel("", 20, F_Y, 787, 64)
text("<b>Parameters</b> &nbsp;11,537,408 (embeddings) + 12 × 3,152,384 (attention 1,050,624 · feed-forward 2,099,712 · LayerNorm 2,048) = <b>49,366,016</b>.<br>"
     "<b>Training</b> &nbsp;Stage 1, general distillation from a frozen BamiBERT (1:1 layer map, 120K steps): "
     "<i>L</i> = Σ<sub>i</sub> MSE(<i>W</i><sub>i</sub> <i>h</i><sub>i</sub><sup>S</sup>, <i>h</i><sub>i</sub><sup>T</sup>) + Σ<sub>j</sub> MSE(mean<sub>heads</sub> <i>A</i><sub>j</sub><sup>S</sup>, mean<sub>heads</sub> <i>A</i><sub>j</sub><sup>T</sup>), "
     "<i>W</i><sub>i</sub> ∈ ℝ<sup>512×768</sup> discarded afterwards. "
     "Stage 2, fine-tuning: <i>L</i> = 0.5·CE + 0.5·<i>T</i><sup>2</sup>·KL(<i>p</i><sup>CafeBERT</sup> ‖ <i>p</i><sup>S</sup>), <i>T</i> = 2, "
     "out-of-fold CafeBERT soft labels, layer-wise lr decay γ = 0.8.",
     28, F_Y + 3, 772, 58, pe, size=9)

W, H = 827, 1169
xml = (
    '<mxfile host="app.diagrams.net" agent="Claude Code" version="24.0.0">'
    f'<diagram id="goibert-m-arch" name="GoiBERT-M architecture (A4 portrait)">'
    f'<mxGraphModel dx="827" dy="1169" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
    f'pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0" background="#ffffff">'
    '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>"
)
open(OUT, "w", encoding="utf-8").write(xml)
bad = 0
for i, a in enumerate(BOXES):
    for b in BOXES[i + 1:]:
        if a[1] != b[1]:
            continue
        ax, ay, aw, ah = a[2:6]; bx, by, bw, bh = b[2:6]
        if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
            bad += 1; print("OVERLAP", a[6][:40], "|", b[6][:40])
print("cells", len(cells), "page", W, H, "footer bottom", F_Y + 64, "overlaps", bad)
