#!/usr/bin/env python3
"""Two paper figures for GoiBERT-M, as separate draw.io files.
Fig. 1: dual-teacher training pipeline (text width, ~16 cm).  Fig. 2: encoder architecture (one column, ~7.5 cm).
Canvas sizes are chosen so that 11 px text prints at about 8 pt at those widths."""
import html, sys

FONT = "fontFamily=Helvetica;"
PAL = {
    "emb":  "fillColor=#DCE6F2;strokeColor=#5B7FA6;",   # teacher 1 (BamiBERT) / embeddings
    "cafe": "fillColor=#F9E3D2;strokeColor=#C0773F;",   # teacher 2 (CafeBERT) / attention
    "stud": "fillColor=#DDEEDD;strokeColor=#5E9461;",   # student / feed-forward
    "norm": "fillColor=#EDEDED;strokeColor=#7F7F7F;",
    "head": "fillColor=#E8DDF0;strokeColor=#8663A3;",
    "io":   "fillColor=#FFFFFF;strokeColor=#444444;",
}
BOXS = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;strokeWidth=1.2;fontSize=11;fontColor=#1A1A1A;" + FONT


class Doc:
    def __init__(self, w, h, name):
        self.w, self.h, self.name = w, h, name
        self.cells, self.origin, self.n = [], {"1": (0, 0)}, 1
        self.boxes = []

    def _id(self):
        self.n += 1
        return f"n{self.n}"

    def node(self, label, x, y, w, h, style, parent="1", check=True):
        i = self._id()
        ox, oy = self.origin[parent]
        self.origin[i] = (x, y)
        self.cells.append(f'<mxCell id="{i}" value="{html.escape(label, quote=True)}" style="{style}" vertex="1" parent="{parent}">'
                          f'<mxGeometry x="{x-ox}" y="{y-oy}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        if check:
            self.boxes.append((i, parent, x, y, w, h, label))
        return i

    def box(self, label, x, y, w, h, kind, parent="1", extra=""):
        return self.node(label, x, y, w, h, BOXS + PAL[kind] + extra, parent)

    def text(self, label, x, y, w, h, parent="1", size=9, align="center", valign="top", color="#333333", extra=""):
        st = (f"text;html=1;whiteSpace=wrap;overflow=hidden;fontSize={size};fontColor={color};align={align};"
              f"verticalAlign={valign};spacing=2;" + FONT + extra)
        return self.node(label, x, y, w, h, st, parent, check=False)

    def panel(self, title, sub, x, y, w, h):
        st = ("rounded=1;arcSize=2;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=#B5B5B5;strokeWidth=1;"
              "verticalAlign=top;align=left;spacingLeft=8;spacingTop=4;fontSize=12;fontStyle=1;fontColor=#1A1A1A;" + FONT)
        p = self.node(title, x, y, w, h, st, check=False)
        if sub:
            self.text(sub, x + 6, y + 24, w - 12, 26, p, size=9, align="left", color="#555555", extra="fontStyle=2;")
        return p

    def edge(self, s, t, exit=(0.5, 1), entry=(0.5, 0), points=None, dashed=False, label="", arrow="block", straight=False, color="#333333", width=1.2):
        i = self._id()
        es = "none" if straight else "orthogonalEdgeStyle"
        st = (f"edgeStyle={es};rounded=0;html=1;endArrow={arrow};endFill=1;strokeColor={color};strokeWidth={width};"
              f"exitX={exit[0]};exitY={exit[1]};entryX={entry[0]};entryY={entry[1]};fontSize=10;labelBackgroundColor=#FAFAFA;" + FONT)
        if dashed:
            st += "dashed=1;dashPattern=4 3;"
        pts = ""
        if points:
            pts = '<Array as="points">' + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points) + "</Array>"
        self.cells.append(f'<mxCell id="{i}" value="{html.escape(label, quote=True)}" style="{st}" edge="1" parent="1" source="{s}" target="{t}">'
                          f'<mxGeometry relative="1" as="geometry">{pts}</mxGeometry></mxCell>')
        return i

    def save(self, path):
        xml = ('<mxfile host="app.diagrams.net" agent="Claude Code" version="24.0.0">'
               f'<diagram id="{self.name}" name="{self.name}">'
               f'<mxGraphModel dx="{self.w}" dy="{self.h}" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" '
               f'pageScale="1" pageWidth="{self.w}" pageHeight="{self.h}" math="0" shadow="0" background="#ffffff">'
               '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self.cells) + "</root></mxGraphModel></diagram></mxfile>")
        open(path, "w", encoding="utf-8").write(xml)
        bad = 0
        for a_i, a in enumerate(self.boxes):
            for b in self.boxes[a_i + 1:]:
                if a[1] != b[1]:
                    continue
                ax, ay, aw, ah = a[2:6]; bx, by, bw, bh = b[2:6]
                if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                    bad += 1; print("OVERLAP", a[6][:30], "|", b[6][:30])
        print(path, "cells", len(self.cells), "canvas", self.w, self.h, "overlaps", bad)


# =========================================================== Fig. 1: dual-teacher pipeline
def fig1(path):
    d = Doc(660, 290, "fig1-dual-teacher")
    # ① general distillation
    p1 = d.panel("① General distillation", "Teacher 1 · shares the tokenizer · token-level", 10, 10, 206, 270)
    corpus = d.box("Unlabeled Vietnamese text<br>329M tokens", 22, 60, 182, 34, "io", p1)
    teacher = d.box("<b>BamiBERT</b><br>12 × 768<br>frozen", 22, 110, 84, 56, "emb", p1)
    student = d.box("<b>GoiBERT-M</b><br>12 × 512<br>trained", 114, 110, 90, 56, "stud", p1)
    loss1 = d.box("<i>L</i><sub>hid</sub> + <i>L</i><sub>attn</sub><br>layer <i>i</i> ↔ layer <i>i</i>", 22, 186, 182, 40, "norm", p1)
    backbone = d.box("<b>Distilled backbone</b>", 22, 240, 182, 28, "stud", p1)
    d.edge(corpus, teacher, (0.2308, 1), (0.5, 0)); d.edge(corpus, student, (0.7527, 1), (0.5, 0))
    d.edge(teacher, loss1, (0.5, 1), (0.2308, 0)); d.edge(student, loss1, (0.5, 1), (0.7527, 0))
    d.edge(loss1, backbone)
    # ② out-of-fold soft labels
    p2 = d.panel("② Soft labels", "Teacher 2 · other tokenizer · output-level", 228, 10, 176, 270)
    train = d.box("Task training set<br>split into 5 folds", 240, 60, 152, 40, "io", p2)
    cafe = d.box("<b>CafeBERT</b> (560M) × 5<br>each trained on 4 folds", 240, 120, 152, 56, "cafe", p2)
    soft = d.box("Soft labels <i>p</i><sup>Cafe</sup><br>for the held-out fold", 240, 196, 152, 40, "io", p2)
    d.edge(train, cafe); d.edge(cafe, soft)
    d.text("initialise student", 244, 256, 144, 14, p2, size=9, color="#555555", extra="fontStyle=2;")
    # ③ fine-tuning
    p3 = d.panel("③ Task fine-tuning", "cross-entropy + distillation", 432, 10, 218, 270)
    note = d.node("Layer-wise lr decay γ = 0.8<br>30 epochs · batch 32 · 3 seeds", 444, 60, 194, 44,
                  "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#B5B5B5;dashed=1;fontSize=10;fontColor=#555555;" + FONT, p3)
    loss2 = d.box("<b>Loss</b><br>0.5 · CE(<i>y</i>, <i>p</i><sup>S</sup>)<br>+ 0.5 · <i>T</i><sup>2</sup> · KL(<i>p</i><sup>Cafe</sup> ‖ <i>p</i><sup>S</sup>)", 444, 128, 194, 56, "norm", p3)
    final = d.box("<b>GoiBERT-M</b> + task head<br>fine-tuned per task", 444, 232, 194, 44, "stud", p3)
    d.edge(final, loss2, (0.5, 0), (0.5, 1), label="<i>p</i><sup>S</sup>")
    d.edge(soft, loss2, (1, 0.5), (0, 0.5), points=[(418, 216), (418, 156)])
    d.edge(backbone, final, (1, 0.5), (0, 0.5))
    d.save(path)


# =========================================================== Fig. 2: encoder architecture
def fig2(path):
    d = Doc(300, 492, "fig2-architecture")
    X, W = 50, 200
    head = d.box("<b>Task head</b><br>classification · ABSA · NER", X, 8, W, 34, "head")
    cont = d.node("Encoder layer", 30, 72, 240, 196,
                  "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#F4F6F9;strokeColor=#3B4B63;strokeWidth=1.4;"
                  "verticalAlign=top;align=left;spacingLeft=8;spacingTop=3;fontSize=11;fontStyle=1;fontColor=#1A1A1A;" + FONT)
    d.text("× 12 · 37.8M", 176, 73, 90, 20, cont, size=11, align="right", color="#1A1A1A", extra="fontStyle=1;")
    bx, bw = 60, 180
    an2 = d.box("Add &amp; LayerNorm", bx, 100, bw, 24, "norm", cont)
    ffn = d.box("Feed-forward (GELU)<br>512 → 2,048 → 512", bx, 136, bw, 34, "stud", cont)
    an1 = d.box("Add &amp; LayerNorm", bx, 182, bw, 24, "norm", cont)
    mha = d.box("Multi-head self-attention<br>8 heads, <i>d</i><sub>k</sub> = 64", bx, 218, bw, 34, "cafe", cont)
    dot_style = "ellipse;html=1;fillColor=#333333;strokeColor=none;"
    dot1 = d.node("", 147, 257, 6, 6, dot_style, cont, check=False)
    dot2 = d.node("", 147, 173, 6, 6, dot_style, cont, check=False)
    # embedding block
    emb = d.node("Embedding · 11.5M", 30, 286, 240, 138,
                 "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#EEF3F9;strokeColor=#5B7FA6;strokeWidth=1.2;"
                 "verticalAlign=top;align=left;spacingLeft=8;spacingTop=3;fontSize=11;fontStyle=1;fontColor=#1A1A1A;" + FONT)
    lndp = d.box("LayerNorm · Dropout", 60, 314, 180, 22, "norm", emb)
    sm = d.node("+", 141, 348, 18, 18,
                "ellipse;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#444444;strokeWidth=1.2;fontSize=12;fontStyle=1;" + FONT, emb, check=False)
    lk = []
    for i, (nm, sh) in enumerate([("Word", "20,481×512"), ("Position", "2,050×512"), ("Token-type", "1×512")]):
        lk.append(d.box(f"{nm}<br>{sh}", 48 + i * 70, 380, 64, 34, "emb", emb, "fontSize=9;"))
    tok = d.box("Vietnamese text → BamiBERT<br>tokenizer → token ids (≤ 256)", X, 440, W, 38, "io")
    # data flow (upwards)
    for i, l in enumerate(lk):
        d.edge(tok, l, ((78 + 70 * i - X) / W if False else (0.15, 0.5, 0.85)[i], 0), (0.5, 1))
    d.edge(lk[0], sm, (0.5, 0), (0, 0.5)); d.edge(lk[1], sm, (0.5, 0), (0.5, 1)); d.edge(lk[2], sm, (0.5, 0), (1, 0.5))
    d.edge(sm, lndp, (0.5, 0), (0.5, 1))
    d.edge(lndp, mha, (0.5, 0), (0.5, 1))
    d.edge(mha, an1, (0.5, 0), (0.5, 1)); d.edge(an1, ffn, (0.5, 0), (0.5, 1)); d.edge(ffn, an2, (0.5, 0), (0.5, 1))
    d.edge(an2, head, (0.5, 0), (0.5, 1))
    d.text("<i>H</i> ∈ ℝ<sup><i>n</i>×512</sup>", 156, 46, 100, 20, size=10, align="left", color="#333333")
    # residual connections
    d.edge(dot1, an1, (1, 0.5), (1, 0.5), points=[(256, 260), (256, 194)], width=1.1)
    d.edge(dot2, an2, (1, 0.5), (1, 0.5), points=[(256, 176), (256, 112)], width=1.1)
    d.save(path)


if __name__ == "__main__":
    fig1(sys.argv[1]); fig2(sys.argv[2])
