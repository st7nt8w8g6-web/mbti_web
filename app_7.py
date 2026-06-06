# ============================================================
#  문체 MBTI 분석기 — Streamlit 웹앱  (app.py)
#
#  ★ 실행 방법 ★
#  1. app.py 와 style_mbti_v2.py 를 같은 폴더에 두세요.
#  2. images/ 폴더에 ESIF.png 등 16개 이미지를 넣으세요.
#  3. pip install streamlit matplotlib numpy
#  4. streamlit run app.py
# ============================================================

import io
import sys
import os
import warnings
import base64

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import streamlit as st

# ── style_mbti_v2.py import ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_mbti_v2 import (
    classify,
    get_persona,
    _build_persona_description_html,
    _build_type_description_html,
    PERSONA_MAP,
    AXIS_COLORS,
    BG_CHART, PANEL_BG, TXT_DARK, TXT_GRAY, TRACK_BG,
)


# ─────────────────────────────────────────────────────────────
# 한글 폰트 설정 (macOS / Windows / Linux 모두 커버)
# style_mbti_v2.py 의 _setup_korean_font 는 코랩/Linux 기준이라
# macOS 에서 차트 한글이 깨지므로 app.py 에서 별도로 재설정
# ─────────────────────────────────────────────────────────────
def _setup_font_for_chart():
    candidates = [
        # macOS 기본 한글 폰트
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/AppleGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        # macOS Homebrew nanum
        "/opt/homebrew/share/fonts/nanum-fonts/NanumGothic.ttf",
        "/usr/local/share/fonts/nanum/NanumGothic.ttf",
        # Windows
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        # Linux / 코랩
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/nanum/NanumGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                fe = fm.FontEntry(fname=path, name="AppKorFont")
                fm.fontManager.ttflist.insert(0, fe)
                plt.rcParams["font.family"] = "AppKorFont"
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue

    # 후보 없으면 시스템 폰트 목록에서 한글 폰트 탐색
    for f in fm.fontManager.ttflist:
        n = (f.name + f.fname).lower()
        if any(k in n for k in ["nanum", "malgun", "gulim", "applegothic",
                                  "applesdgothic", "notosanscjk", "noto sans cjk"]):
            try:
                fe = fm.FontEntry(fname=f.fname, name="AppKorFont")
                fm.fontManager.ttflist.insert(0, fe)
                plt.rcParams["font.family"] = "AppKorFont"
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue

    plt.rcParams["axes.unicode_minus"] = False

_setup_font_for_chart()


# ─────────────────────────────────────────────────────────────
# 이미지 탐색: images/ 폴더 우선, 없으면 None
# 반환값: 파일 경로 문자열 (존재 확인 완료) or None
# ─────────────────────────────────────────────────────────────
def find_persona_image(code):
    base = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.path.join(base, "images"),
        base,
    ]
    exts = [".png", ".jpg", ".jpeg"]
    for folder in search_dirs:
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            nl = fname.lower()
            if nl.startswith(code.lower()) and any(nl.endswith(e) for e in exts):
                full = os.path.join(folder, fname)
                if os.path.isfile(full):
                    return full
    return None


# ─────────────────────────────────────────────────────────────
# 이미지 파일 → bytes  (st.image에 bytes로 넘겨야 로컬 경로 제한 우회)
# ─────────────────────────────────────────────────────────────
def load_image_bytes(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# 차트 생성 → PNG bytes
# ─────────────────────────────────────────────────────────────
def make_chart_bytes(result):
    code = result["type_code"]
    name = get_persona(code)[0]

    axes_meta = [
        ("axis1", "감성형(E)", "논리형(L)"),
        ("axis2", "서사형(S)", "압축형(C)"),
        ("axis3", "직관형(I)", "구체형(D)"),
        ("axis4", "유보형(F)", "단정형(J)"),
    ]
    ax_keys  = [m[0] for m in axes_meta]
    colors   = [AXIS_COLORS[k] for k in ax_keys]
    values   = [result[k]["radar"] for k in ax_keys]
    labels_r = [result[k]["label"] for k in ax_keys]

    fig = plt.figure(figsize=(15, 6), facecolor=BG_CHART)
    fig.suptitle(
        f"문체 MBTI  [{code}]   {name}",
        color=TXT_DARK, fontsize=15, fontweight="bold", y=1.02,
    )

    # ── 막대 그래프 ──────────────────────────────────────────
    ax_bar = fig.add_axes([0.03, 0.10, 0.46, 0.82], facecolor=PANEL_BG)
    for i, ((key, left_lbl, right_lbl), val, color, res_lbl) in enumerate(
            zip(axes_meta, values, colors, labels_r)):
        y = 3 - i
        ax_bar.barh(y, 1.0, left=0, height=0.48,
                    color=TRACK_BG, zorder=1, linewidth=0)
        ax_bar.barh(y, val, left=0, height=0.48,
                    color=color, alpha=0.88, zorder=2, linewidth=0)
        ax_bar.barh(y, 1.0 - val, left=val, height=0.48,
                    color=color, alpha=0.18, zorder=2, linewidth=0)
        ax_bar.text(-0.03, y, left_lbl, va="center", ha="right",
                    color=color, fontsize=11, fontweight="bold")
        ax_bar.text(1.03, y, right_lbl, va="center", ha="left",
                    color=TXT_GRAY, fontsize=10)
        ax_bar.text(val / 2, y, f"{val:.0%}", va="center", ha="center",
                    color="#ffffff", fontsize=12, fontweight="bold")
        ax_bar.text(0.50, y + 0.33, f"→  {res_lbl}",
                    va="center", ha="center", color=color, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor=BG_CHART, edgecolor=color, linewidth=1.1))
    ax_bar.set_xlim(-0.38, 1.40)
    ax_bar.set_ylim(-0.55, 4.0)
    ax_bar.axis("off")
    ax_bar.set_title("  4개 축 분석 결과", color=TXT_DARK,
                     fontsize=12, loc="left", pad=10)

    # ── 레이더 차트 ──────────────────────────────────────────
    N        = 4
    angles   = [n / N * 2 * np.pi for n in range(N)]
    angles_c = angles + [angles[0]]
    vals_c   = values + [values[0]]
    radar_labels = [f"{m[1]}\n/ {m[2]}" for m in axes_meta]

    ax_r = fig.add_axes([0.54, 0.05, 0.44, 0.90],
                         projection="polar", facecolor=PANEL_BG)
    for rv in [0.25, 0.5, 0.75, 1.0]:
        ax_r.plot(angles_c, [rv] * 5, color="#ccccdd", linewidth=0.8, zorder=1)
    for ang in angles:
        ax_r.plot([ang, ang], [0, 1], color="#ccccdd", linewidth=0.8, zorder=1)

    ax_r.fill(angles_c, vals_c, alpha=0.18, color="#4a7c59", zorder=2)
    ax_r.plot(angles_c, vals_c, color="#4a7c59", linewidth=2.2, zorder=3)
    for ang, val, color in zip(angles, values, colors):
        ax_r.plot(ang, val, "o", color=color, markersize=12, zorder=5,
                  markeredgecolor="#ffffff", markeredgewidth=1.5)
        ax_r.plot(ang, val, "o", color=color, markersize=6, zorder=6)
        ax_r.text(ang, val + 0.14, f"{val:.0%}",
                  ha="center", va="center", color=color,
                  fontsize=8, fontweight="bold")

    ax_r.set_xticks(angles)
    ax_r.set_xticklabels(radar_labels, color=TXT_DARK, fontsize=9)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(["25%", "50%", "75%", "100%"],
                          color=TXT_GRAY, fontsize=7)
    ax_r.set_ylim(0, 1)
    ax_r.grid(False)
    ax_r.spines["polar"].set_visible(False)

    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(buf, format="png", dpi=150,
                    bbox_inches="tight", facecolor=BG_CHART)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────
# Streamlit 앱
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="문체 MBTI 분석기",
        page_icon="✍️",
        layout="wide",
    )

    # ── 전역 CSS (밝은 연두 배경 + 검은 글씨) ────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;700;900&family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', sans-serif;
        color: #1a1a1a;
    }
    .stApp {
        background: linear-gradient(160deg, #e8f5e2 0%, #d4edcc 40%, #c8e6bd 100%);
        min-height: 100vh;
    }
    .main-title {
        text-align: center;
        font-family: 'Noto Serif KR', serif;
        font-size: 2.6rem;
        font-weight: 900;
        color: #2d5a27;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .main-sub {
        text-align: center;
        font-size: 1rem;
        color: #4a7c59;
        margin-bottom: 2rem;
        letter-spacing: 2px;
        font-weight: 500;
    }
    .stTextArea textarea {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border: 1.5px solid #a8d5a2 !important;
        border-radius: 12px !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.8 !important;
    }
    .stTextArea textarea:focus {
        border-color: #4a7c59 !important;
        box-shadow: 0 0 0 3px rgba(74,124,89,0.15) !important;
    }
    .stTextArea label { color: #2d5a27 !important; font-weight: 600 !important; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4a7c59, #6aab78) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        font-family: 'Noto Sans KR', sans-serif !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 16px rgba(74,124,89,0.35) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3a6647, #5a9a68) !important;
        box-shadow: 0 6px 24px rgba(74,124,89,0.5) !important;
        transform: translateY(-1px) !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] { color: #5a7a60 !important; }
    .intro-card {
        background: #ffffff;
        border: 1.5px solid #b8ddb2;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        box-shadow: 0 2px 12px rgba(74,124,89,0.08);
    }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # ── 헤더 ──────────────────────────────────────────────────
    st.markdown('<div class="main-title">✍️ 문체 MBTI 분석기</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-sub">나의 글쓰기 유형을 4개의 축으로 진단합니다</div>',
                unsafe_allow_html=True)

    # ── 입력 ──────────────────────────────────────────────────
    col_in, _ = st.columns([3, 1])
    with col_in:
        user_text = st.text_area(
            "분석할 글을 붙여넣어 주세요",
            height=260,
            placeholder=(
                "에세이, 감상문, 블로그 글, SNS 장문 포스트 등 자신이 쓴 글을 붙여넣으세요.\n\n"
                "최소 100자 이상이면 더 정확한 결과가 나옵니다."
            ),
        )
        wc = len(user_text.split()) if user_text.strip() else 0
        cc = len(user_text.replace(" ", "").replace("\n", ""))
        st.caption(f"어절 수: {wc}  |  글자 수(공백 제외): {cc}")
        run_btn = st.button("내 문bti 확인! ✨")

    # ── 분석 실행 ─────────────────────────────────────────────
    if run_btn:
        if len(user_text.strip()) < 20:
            st.warning("⚠️ 글이 너무 짧습니다. 최소 20자 이상 입력해 주세요.")
            return

        with st.spinner("분석 중..."):
            result = classify(user_text)

        code = result["type_code"]

        # ── ① 유형 상세 설명 HTML ─────────────────────────────
        # 이미지는 HTML embed 대신 st.image()로 따로 출력하므로
        # persona_img_path=None 으로 호출 (HTML 안에 img 태그 없음)
        persona_html = _build_persona_description_html(code, result, None)
        st.markdown(persona_html, unsafe_allow_html=True)

        # ── ② 유형 이미지 — bytes로 읽어서 st.image() 출력 ───
        # st.image()에 로컬 파일 경로(str)를 넘기면 Streamlit이
        # 보안상 서빙을 거부하는 경우가 있으므로, 파일을 직접
        # bytes로 읽어서 넘기면 확실하게 표시됨
        img_path = find_persona_image(code)
        if img_path:
            img_bytes = load_image_bytes(img_path)
            if img_bytes:
                col_il, col_ic, col_ir = st.columns([2, 3, 2])
                with col_ic:
                    st.image(img_bytes, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ③ 통합 차트 (막대 + 레이더) ──────────────────────
        chart_bytes = make_chart_bytes(result)
        col_l, col_c, col_r = st.columns([1, 6, 1])
        with col_c:
            st.image(chart_bytes, use_container_width=True)

        # ── ④ 4개 축 유형 설명 HTML ───────────────────────────
        type_html = _build_type_description_html(code)
        st.markdown(type_html, unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align:center; color:#5a7a60; font-size:0.83rem;"
            " padding:1.5rem 0;'>다른 글로 다시 분석하려면 위 입력창에 새 글을 붙여넣고 버튼을 누르세요.</div>",
            unsafe_allow_html=True,
        )

    else:
        # ── 시작 전 안내 ──────────────────────────────────────
        st.markdown("""
        <div style="max-width:640px; margin:0 auto; padding:2rem 0;">
          <div class="intro-card">
            <div style="font-size:1rem; font-weight:700; color:#2d5a27; margin-bottom:1rem;">
              🔍 분석하는 4개의 축
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;">
              <div style="background:#fff5f5; border:1px solid #fecaca;
                          border-radius:8px; padding:0.8rem;">
                <div style="color:#e53e3e; font-weight:700; font-size:0.9rem;">감성형(E) vs 논리형(L)</div>
                <div style="color:#555; font-size:0.8rem; margin-top:4px;">감정어·평가부사 vs 논리연결어·분석명사</div>
              </div>
              <div style="background:#f0fffe; border:1px solid #99e6e0;
                          border-radius:8px; padding:0.8rem;">
                <div style="color:#2a9d8f; font-weight:700; font-size:0.9rem;">서사형(S) vs 압축형(C)</div>
                <div style="color:#555; font-size:0.8rem; margin-top:4px;">시간흐름·긴 문장 vs 요약표현·단문</div>
              </div>
              <div style="background:#fffbeb; border:1px solid #fde68a;
                          border-radius:8px; padding:0.8rem;">
                <div style="color:#b45309; font-weight:700; font-size:0.9rem;">직관형(I) vs 구체형(D)</div>
                <div style="color:#555; font-size:0.8rem; margin-top:4px;">추상명사·학술어 vs 감각어·행동동사</div>
              </div>
              <div style="background:#f5f3ff; border:1px solid #c4b5fd;
                          border-radius:8px; padding:0.8rem;">
                <div style="color:#7c3aed; font-weight:700; font-size:0.9rem;">유보형(F) vs 단정형(J)</div>
                <div style="color:#555; font-size:0.8rem; margin-top:4px;">hedge 표현 vs assertive·명사형 종결</div>
              </div>
            </div>
            <div style="margin-top:1.2rem; color:#4a7c59; font-size:0.85rem;
                        text-align:center; font-weight:500;">
              자신이 쓴 글을 붙여넣고 버튼을 눌러보세요.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
