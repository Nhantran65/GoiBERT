"""Shared contract for the mmBERT Vietnamese CPT pilot v2 data fix (docs/MMBERT_VI_CPT_PILOT_PLAN_V2_DATA_FIX.md).

v2 operates on the v1 pool (the 1/8 uniform subsample of HPLT 3.0 `vie_Latn` bins 7-10 that already passed the v1
filters, dedup and benchmark decontamination; `data/mmbert_vi_cpt_pilot/{pool,scan,tokens}`) and re-reads the pinned raw
shards (`data/raw/hplt3_vie/`) only for the host census.  Nothing in v1 is modified; all v2 outputs live under
`data/mmbert_vi_cpt_pilot/v2/` and `artifacts/mmbert_vi_cpt_pilot/v2/`.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

V2_DIR = Path(__file__).resolve().parent
PILOT_DIR = V2_DIR.parent
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))
import common  # noqa: E402  (v1 shared contract: paths, hashing, tokenizer, benchmarks)

ROOT = common.ROOT
DATA_V1 = common.DATA_DIR                      # data/mmbert_vi_cpt_pilot
ART_V1 = common.ART_DIR                        # artifacts/mmbert_vi_cpt_pilot
DATA_V2 = ROOT / "data" / "mmbert_vi_cpt_pilot" / "v2"
ART_V2 = ROOT / "artifacts" / "mmbert_vi_cpt_pilot" / "v2"
SEED = common.SEED

# ------------------------------------------------------------------ S1 reference: mainstream Vietnamese news hosts
# Fixed list (plan v2 §2 "vá 1"): official/general news publishers whose HPLT text is human-written Vietnamese.
MAINSTREAM_NEWS_HOSTS = {
    "tuoitre.vn", "thanhnien.vn", "vnexpress.net", "dantri.com.vn", "laodong.vn", "vietnamnet.vn",
    "tienphong.vn", "nld.com.vn", "sggp.org.vn", "baomoi.com", "vov.vn", "vietnamplus.vn",
    "nhandan.vn", "cand.com.vn", "plo.vn", "qdnd.vn", "baotintuc.vn", "infonet.vn", "vtv.vn",
    "afamily.vn", "soha.vn", "danviet.vn", "nongnghiep.vn", "thethaovanhoa.vn", "bongdaplus.vn",
    "genk.vn", "ictnews.vn", "vietnamfinance.vn", "baodautu.vn", "cafef.vn", "vietstock.vn",
    "kinhtedothi.vn", "hanoimoi.vn", "tuoitrethudo.vn", "phapluat.vn", "luatvietnam.vn",
}

# ------------------------------------------------------------------ v2 §2 "vá 2": UGC platform whitelist (with reasons)
UGC_WHITELIST: dict[str, str] = {
    # diễn đàn / hỏi đáp (user-voice threads and Q&A)
    "webtretho.com": "seed list: parenting forum, first-person posts",
    "webtretho.vn": "seed list: parenting forum, first-person posts",
    "voz.vn": "seed list: general forum, user posts",
    "vozforums.com": "seed list: general forum, user posts",
    "tinhte.vn": "seed list: technology forum with hands-on reviews",
    "otofun.net": "seed list: automotive forum with ownership reviews",
    "lamchame.com": "seed list: parenting forum",
    "hoidap247.com": "seed list: Q&A platform, user questions",
    "hoidapphapluat.com": "seed list: legal Q&A, user questions",
    "hoidapphapluat.vn": "seed list: legal Q&A, user questions",
    "edoctor.io": "seed list: medical Q&A, user questions",
    "cuasotinhyeu.com": "seed list: counselling Q&A, user letters",
    "cuasotinhyeu.vn": "seed list: counselling Q&A, user letters",
    # review / du lịch / ẩm thực (place and product reviews by users)
    "foody.vn": "seed list: restaurant reviews by users",
    "diadiemanuong.com": "seed list: restaurant reviews by users",
    "riviu.vn": "seed list: food review platform",
    "pasgo.vn": "seed list: restaurant booking with user reviews",
    "ivivu.com": "seed list: travel reviews by users",
    "mytour.vn": "seed list: travel reviews by users",
    "gody.vn": "seed list: travel reviews by users",
    "toidi.net": "seed list: travel reviews by users",
    "dulichchat.com": "seed list: travel forum/reviews",
    "wesave.vn": "seed list: voucher site with user reviews",
    # thương mại điện tử có mục đánh giá (product reviews by buyers)
    "tiki.vn": "seed list: e-commerce with buyer reviews",
    "shopee.vn": "seed list: e-commerce with buyer reviews",
    "lazada.vn": "seed list: e-commerce with buyer reviews",
    "thegioididong.com": "seed list: electronics retailer with buyer reviews",
    "cellphones.com.vn": "seed list: electronics retailer with buyer reviews",
    "fptshop.com.vn": "seed list: electronics retailer with buyer reviews",
    "hoanghamobile.com": "seed list: electronics retailer with buyer reviews",
}

# v2 §2: these are excluded unless a dedicated >=70 % audit is run (their Vietnamese pages are largely machine translated)
UGC_EXCLUDED_HOSTS = {"tripadvisor.com", "tripadvisor.com.vn", "agoda.com", "agoda.vn", "booking.com"}

UGC_GROUP = {  # for the two register audit cells
    "forum_qa": {"webtretho.com", "webtretho.vn", "voz.vn", "vozforums.com", "tinhte.vn", "otofun.net", "lamchame.com",
                 "hoidap247.com", "hoidapphapluat.com", "hoidapphapluat.vn", "edoctor.io", "cuasotinhyeu.com", "cuasotinhyeu.vn"},
    "review_commerce": {"foody.vn", "diadiemanuong.com", "riviu.vn", "pasgo.vn", "ivivu.com", "mytour.vn", "gody.vn", "toidi.net",
                        "dulichchat.com", "wesave.vn", "tiki.vn", "shopee.vn", "lazada.vn", "thegioididong.com",
                        "cellphones.com.vn", "fptshop.com.vn", "hoanghamobile.com"},
}

V3_PARQUET = ROOT / "data" / "bami_as_cpt_pilot" / "raw" / "vn_msa_sar_v3" / "data" / "train-00000-of-00001.parquet"
V3_REVISION = "6f3f17b390dcf1bf411804e1bc8246bf88ea1bf4"

SYLLABLE = re.compile(r"[0-9a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+", re.I)
_WS = re.compile(r"\s+")


def host_of(url: str | None) -> str:
    if not url:
        return ""
    try:
        netloc = urlparse(url if "://" in url else "http://" + url).netloc.lower()
    except ValueError:
        return ""
    if "@" in netloc:
        netloc = netloc.split("@", 1)[1]
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    return netloc[4:] if netloc.startswith("www.") else netloc


def host_matches(host: str, hosts: set[str]) -> bool:
    """True when `host` is one of `hosts` or a subdomain of it."""
    return any(host == h or host.endswith("." + h) for h in hosts)


def syllables(text: str) -> list[str]:
    """Lowercase Vietnamese syllables (whitespace/punctuation are boundaries); diacritics are kept."""
    return SYLLABLE.findall(unicodedata.normalize("NFC", text).lower())


def write_json(path: Path, payload: Any) -> None:
    common.write_json(path, payload)


def read_json(path: Path) -> Any:
    return common.read_json(path)


def iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


# ------------------------------------------------------------------ pool table (one row per v1 pool document)
def pool_table() -> list[dict]:
    """Rows of the v1 tokenised pool: part, local index, pool index, group, bin, url, host, token count, split."""
    rows: list[dict] = []
    token_dir = DATA_V1 / "tokens"
    for f in sorted(token_dir.glob("*.docs.jsonl")):
        part = f.name.replace(".docs.jsonl", "")
        shard = part.split("_")[0] + "_" + part.split("_")[1]
        with f.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                r = json.loads(line)
                rows.append({"part": part, "shard": shard, "local_index": i, "pool_index": r["pool_index"], "split": r.get("split", "pool"),
                             "group": r["group"], "bin": str(r["bin"]), "url": r.get("url"), "host": host_of(r.get("url")),
                             "n_tokens": int(r["n_tokens"]), "chars": int(r.get("chars") or 0), "kh64": int(r["kh64"])})
    rows.sort(key=lambda r: r["pool_index"])
    return rows


def part_texts(part: str, wanted_kh64: set[int] | None = None) -> dict[int, str]:
    """Texts of the v1 pool documents of one part, keyed by the document's kh64 (read from the stage-1 docs file)."""
    out: dict[int, str] = {}
    path = DATA_V1 / "scan" / f"{part}.docs.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            k = int(rec["kh64"])
            if wanted_kh64 is None or k in wanted_kh64:
                out[k] = rec["text"]
    return out
