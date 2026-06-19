import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
import numpy as np

# ── 폰트 설정 ──────────────────────────────────────────────────────────────
def _configure_fonts():
    candidates = ["Apple SD Gothic Neo", "Nanum Gothic", "Pretendard", "Noto Sans KR", "Malgun Gothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False


# ── 샘플 데이터 (이미지 기준 시계방향 순서) ───────────────────────────────
SEGMENTS = [
    ("그 외",       8.4,  "중립타겟"),
    ("25-34 여성", 25.1,  "메인타겟"),
    ("35-44 여성", 20.0,  "메인타겟"),
    ("25-34 남성", 14.9,  "메인타겟"),
    ("65+ 여성",   10.2,  "기피타겟"),
    ("55-64 남성",  8.4,  "중립타겟"),
    ("65+ 남성",    6.2,  "기피타겟"),
]

GROUP_COLORS = {
    "메인타겟": "#5B8A38",
    "중립타겟": "#B0B0B0",
    "기피타겟": "#F5A623",
}

# 그룹별 라벨 텍스트 색상 (회색 배경엔 어두운 색, 나머지는 흰색)
TEXT_COLORS = {
    "메인타겟": "white",
    "중립타겟": "#333333",
    "기피타겟": "white",
}


def draw_pie_chart(output_path: str = "db_update/chart_sample.png"):
    _configure_fonts()

    labels = [s[0] for s in SEGMENTS]
    sizes  = [s[1] for s in SEGMENTS]
    groups = [s[2] for s in SEGMENTS]
    colors = [GROUP_COLORS[g] for g in groups]
    total  = sum(sizes)

    fig, ax = plt.subplots(figsize=(5.8, 7.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="#EEEEEE", linewidth=0.1),
    )

    # ── wedge 내부 라벨 ────────────────────────────────────────────────────
    BASE_FS = 8.0

    for i, w in enumerate(wedges):
        pct   = sizes[i]  # 샘플 데이터의 퍼센트 값을 직접 사용
        angle = (w.theta2 + w.theta1) / 2.0
        rad   = np.deg2rad(angle)
        tc    = TEXT_COLORS[groups[i]]

        if pct > 20:
            fs = BASE_FS + 1.0
        elif pct < 10:
            fs = BASE_FS - 1.5
        else:
            fs = BASE_FS

        # 작은 조각(< 7%)은 바깥쪽에 배치
        if pct < 7.0:
            r = 1.18
            x = r * np.cos(rad)
            y = r * np.sin(rad)
            ax.text(x, y + 0.07, labels[i], ha="center", va="center",
                    fontsize=fs, fontweight="bold", color="#333333")
            ax.text(x, y - 0.09, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=fs, color="#555555")
        else:
            r = 0.63
            x = r * np.cos(rad)
            y = r * np.sin(rad)
            ax.text(x, y + 0.07, labels[i], ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=tc)
            ax.text(x, y - 0.09, f"{pct:.1f}%", ha="center", va="center",
                    fontsize=fs, color=tc)

    # ── 하단 범례 (세로 배열) ──────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor=GROUP_COLORS[g], label=g, linewidth=0)
        for g in ["메인타겟", "중립타겟", "기피타겟"]
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=1,
        frameon=False,
        fontsize=11.5,
        handlelength=1.6,
        handleheight=1.6,
        labelspacing=1.0,
    )

    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    draw_pie_chart()
