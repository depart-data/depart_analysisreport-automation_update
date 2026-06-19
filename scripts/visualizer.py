import warnings
import logging
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('matplotlib.font_manager').setLevel(logging.CRITICAL)

# scripts/visualizer.py
import io
import os
import colorsys
from typing import Any, Dict, List, Optional


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import base64
import math 



# 경고 끄기
plt.rcParams["figure.max_open_warning"] = 100

DEFAULT_THEME = "#4e73df"



def _configure_matplotlib_fonts() -> None:
    # Prefer Korean-capable fonts to avoid broken glyphs in SVG.
    preferred = [
        "Apple SD Gothic Neo",
        "Noto Sans KR",
        "Malgun Gothic",
        #"Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["font.family"] = preferred
    plt.rcParams["axes.unicode_minus"] = False
    # Embed glyphs as paths for consistent rendering in SVG/PDF.
    plt.rcParams["svg.fonttype"] = "path"


_configure_matplotlib_fonts()


def _normalize_hex(hex_color: str) -> str:
    if not hex_color:
        return DEFAULT_THEME
    color = hex_color.strip().lower()
    if color.startswith("#"):
        color = color[1:]
    if len(color) == 3:
        color = "".join([c * 2 for c in color])
    if len(color) != 6 or any(c not in "0123456789abcdef" for c in color):
        return DEFAULT_THEME
    return "#" + color


def _hex_to_rgb01(hex_color: str) -> tuple:
    hex_color = _normalize_hex(hex_color)[1:]
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


def _rgb01_to_hex(r: float, g: float, b: float) -> str:
    r_i = int(max(0.0, min(1.0, r)) * 255)
    g_i = int(max(0.0, min(1.0, g)) * 255)
    b_i = int(max(0.0, min(1.0, b)) * 255)
    return "#{:02x}{:02x}{:02x}".format(r_i, g_i, b_i)


def _adjust_lightness(hex_color: str, delta: float) -> str:
    r, g, b = _hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + delta))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb01_to_hex(r, g, b)


def complementary_hex(hex_color: str) -> str:
    """hue를 180도 회전한 보색 반환"""
    r, g, b = _hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + 0.5) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb01_to_hex(r, g, b)


def _adjust_hls(hex_color: str, delta_l: float, delta_s: float) -> str:
    r, g, b = _hex_to_rgb01(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l + delta_l))
    s = max(0.0, min(1.0, s + delta_s))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return _rgb01_to_hex(r, g, b)


def build_color_map(theme_color: str) -> Dict[str, Any]:
    base = _normalize_hex(theme_color)
    light = _adjust_lightness(base, 0.22)
    lighter = _adjust_lightness(base, 0.38)
    dark = _adjust_lightness(base, -0.18)
    darker = _adjust_lightness(base, -0.32)
    # 채도 낮은 테마에서도 색이 살아있도록 최소 채도 보장
    _, _, s0 = colorsys.rgb_to_hls(*_hex_to_rgb01(base))
    header = _adjust_hls(base, 0.20, max(-0.45, 0.15 - s0))
    highlight = _adjust_hls(base, 0.65, max(-0.40, 0.20 - s0))  # 형광펜용: 매우 밝고 채도 낮춤
    series = [base, dark, light, darker]
    return {
        "base": base,
        "light": light,
        "lighter": lighter,
        "dark": dark,
        "darker": darker,
        "header": header,
        "highlight": highlight,
        "series": series,
        "grid": "#9b9b9b",
        "text": "#111111",
        "muted": "#666666",
    }


def relative_luminance(hex_color: str) -> float:
    r, g, b = _hex_to_rgb01(hex_color)

    def to_lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r_l, g_l, b_l = map(to_lin, (r, g, b))
    return 0.2126 * r_l + 0.7152 * g_l + 0.0722 * b_l


def is_dark_color(hex_color: str) -> bool:
    return relative_luminance(hex_color) < 0.5


def _fig_to_svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    idx = svg.find("<svg")
    if idx != -1:
        svg = svg[idx:]
    return svg


def _style_axes(ax, color_map: Dict[str, Any], grid_axis: Optional[str] = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    ax.tick_params(colors="#666666", labelsize=10)
    if grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis, color=color_map["grid"], linewidth=0.8)
    else:
        ax.grid(False)


def _value_colors(
    values: List[float],
    color_map: Dict[str, Any],
    palette: Optional[List[str]] = None,
):
    if not values:
        return [color_map["base"]]
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [color_map["base"] for _ in values]
    if palette:
        cmap = LinearSegmentedColormap.from_list("theme", palette, N=256).reversed()
    else:
        cmap = LinearSegmentedColormap.from_list(
            "theme",
            [color_map["lighter"], color_map["light"], color_map["base"], color_map["dark"]],
        )
    return [cmap((v - vmin) / (vmax - vmin)) for v in values]


def _contrast_text_color(rgba, threshold: float = 0.45) -> str:
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < threshold else "#1a1a1a"


def _extract_month_spans(labels: List[Any]) -> List[Dict[str, Any]]:
    parsed = pd.to_datetime(pd.Series(labels), errors="coerce")
    if parsed.isna().all():
        return []

    month_keys = parsed.dt.to_period("M")
    spans: List[Dict[str, Any]] = []
    current_month = None
    start_idx = None

    for idx, month_key in enumerate(month_keys):
        if pd.isna(month_key):
            if current_month is not None and start_idx is not None:
                spans.append({"month": current_month, "start": start_idx, "end": idx - 1})
            current_month = None
            start_idx = None
            continue

        if current_month is None:
            current_month = month_key
            start_idx = idx
            continue

        if month_key != current_month and start_idx is not None:
            spans.append({"month": current_month, "start": start_idx, "end": idx - 1})
            current_month = month_key
            start_idx = idx

    if current_month is not None and start_idx is not None:
        spans.append({"month": current_month, "start": start_idx, "end": len(month_keys) - 1})

    return spans


def _line_label_indices(values: List[Any]) -> List[int]:
    valid_points: List[tuple[int, float]] = []
    for idx, value in enumerate(values):
        if pd.isna(value):
            continue
        valid_points.append((idx, float(value)))

    if not valid_points:
        return []

    first_idx, first_value = valid_points[0]
    label_idxs: List[int] = [first_idx]
    ref_peak_idx = first_idx
    ref_peak_value = first_value
    prev_value = first_value
    decline_found = False

    for idx, value in valid_points[1:]:
        if not decline_found:
            if value >= ref_peak_value:
                ref_peak_value = value
                ref_peak_idx = idx
            if value < prev_value:
                decline_found = True
                if ref_peak_idx not in label_idxs:
                    label_idxs.append(ref_peak_idx)
        elif value > ref_peak_value:
            ref_peak_value = value
            label_idxs.append(idx)

        prev_value = value

    if not decline_found and ref_peak_idx not in label_idxs:
        label_idxs.append(ref_peak_idx)

    label_idxs.sort()
    return label_idxs

def render_line_chart(dataset: Dict[str, Any], color_map: Dict[str, Any], compact: bool = False) -> str:
    if not dataset:
        return ""

    labels = dataset.get("labels") or []
    series = dataset.get("series") or []

    if not labels or not series:
        return ""

    title_text = str(dataset.get("title") or "").strip()
    show_average = dataset.get("show_average", False)
    
    # 평균선 표시 대상 차트
    if title_text in [ "오가닉 조회수 추이 (주별)", "프로필 방문 수(주별)"] :
        show_average = True
    # =========================================================

    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8.0, 4.4))

    unit = str(dataset.get("unit") or "").strip()
    plotted_values: List[float] = []
    show_legend = bool(dataset.get("show_legend"))

    chart_label = "주별" if "주별" in title_text else "월별" if "월별" in title_text else None

    label_map = {
        "spend": "광고비",
        "revenue": "매출발생",
    }

    color_map_line = {
        "spend": "#1565C0",
        "revenue": "#C62828",
    }

    # 데이터 라인 그리기
    for idx, s in enumerate(series):
        raw_data = s.get("data") or []
        if not raw_data:
            continue

        data = pd.to_numeric(pd.Series(raw_data), errors="coerce").tolist()
        x_values = x[: len(data)]
        if not x_values:
            continue

        series_name = s.get("name") or f"series_{idx}"

        color = (
            color_map_line.get(series_name, color_map["series"][idx % len(color_map["series"])])
            if show_legend
            else color_map["series"][idx % len(color_map["series"])]
        )

        plot_label = label_map.get(series_name, series_name)

        ax.plot(
            x_values,
            data,
            color=color,
            linewidth=2,
            marker="o",
            markersize=3.8,
            label=plot_label if show_legend else None,
        )

        label_idx_set = set(_line_label_indices(data))
        last_labeled_x = None
        last_labeled_y = None

        valid_values = [float(v) for v in data if pd.notna(v)]
        series_y_min = min(valid_values)
        series_y_max = max(valid_values)
        series_y_span = max(series_y_max - series_y_min, 1)

        for point_idx, (x_val, y_val) in enumerate(zip(x_values, data)):
            if pd.isna(y_val):
                continue

            y_num = float(y_val)
            plotted_values.append(y_num)

            if x_val not in label_idx_set:
                continue

            if last_labeled_x is not None and last_labeled_y is not None:
                x_gap = abs(x_val - last_labeled_x)
                y_gap = abs(y_num - last_labeled_y)

                if x_gap <= 2 and y_gap < series_y_span * 0.12:
                    continue

            base_offset = 6
            va = "bottom"

            if idx == 0 and show_legend:
                if series_name == "spend":
                    base_offset = 6
                    va = "bottom"
                elif series_name == "revenue":
                    base_offset = 17
                    va = "bottom"

            xytext = (0, base_offset)
            ha = "center"

            ax.annotate(
                f"{_format_chart_value(y_num)}{unit}" if unit else _format_chart_value(y_num),
                (x_val, y_num),
                textcoords="offset points",
                xytext=xytext,
                ha=ha,
                va=va,
                fontsize=8,
                color=color,
                clip_on=False,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85
                ),
            )

            last_labeled_x = x_val
            last_labeled_y = y_num
            
    if not plotted_values:
        return ""

   # 이전 분기 평균선 정보 (Y축 범위 계산 + 평균선 그리기 양쪽에서 공통 사용)
    prev_quarter_avg = dataset.get("prev_quarter_avg")
    prev_avg_val = None
    if prev_quarter_avg and prev_quarter_avg.get("avg") is not None:
        prev_avg_val = float(prev_quarter_avg["avg"])

    # Y축 범위 설정
    data_min, data_max = min(plotted_values), max(plotted_values)

    # 이전 분기 평균이 주별 데이터 범위를 벗어나면 y축 범위 계산에 포함
    range_min = min(data_min, prev_avg_val) if prev_avg_val is not None else data_min
    range_max = max(data_max, prev_avg_val) if prev_avg_val is not None else data_max

    y_span = range_max - range_min
    y_pad = max(y_span * 0.22, 0.3)

    if unit == "%":
        # CTR처럼 변동폭이 작은 % 지표는 0부터 시작하면 변화량이 잘 보이지 않으므로,
        # 주별 데이터 최솟값이 1.5%를 초과하면 0.5 단위로 내림한 값을 시작값으로 사용한다.
        if data_min > 1.5:
            y_low = max(0.0, math.floor((data_min - 0.5) * 2) / 2)
        else:
            y_low = 0.0

        # 이전 분기 평균이 시작값보다 낮으면, 평균선이 가려지지 않도록 시작값을 더 낮춘다.
        if prev_avg_val is not None and prev_avg_val < y_low:
            y_low = max(0.0, math.floor((prev_avg_val - 0.5) * 2) / 2)
    else:
        y_low = range_min - y_pad * 0.2

    y_high = range_max + y_pad

    if abs(y_high - y_low) < 1e-12:
        y_high = y_low + 1.0

    ax.set_ylim(y_low, y_high)

    # 월 밴드
    month_spans = _extract_month_spans(labels)

    if month_spans:
        ax.set_xlim(-0.55, len(labels) - 0.5)

        month_band_ymin, month_band_ymax = -0.06, 0.0

        for span_idx, span in enumerate(month_spans):
            start, end, period = span["start"], span["end"], span["month"]

            ax.add_patch(
                Rectangle(
                    (start - 0.5, month_band_ymin),
                    (end - start) + 1.0,
                    month_band_ymax - month_band_ymin,
                    transform=ax.get_xaxis_transform(),
                    facecolor="#f5f5f5" if span_idx % 2 == 0 else "#fafafa",
                    edgecolor="none",
                    zorder=0,
                    clip_on=False,
                )
            )

            ax.text(
                (start + end) / 2,
                (month_band_ymin + month_band_ymax) / 2,
                f"{period.year}.{period.month:02d}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                fontsize=7.5,
                color="#7a7a7a",
                zorder=1,
                clip_on=False,
            )

        for span in month_spans[1:]:
            ax.axvline(
                span["start"] - 0.5,
                color="#d9d9d9",
                linestyle=(0, (2, 3)),
                linewidth=0.9,
                zorder=1.5,
            )

    ax.set_xticks([])
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda v, _: f"{int(round(v)):,}" if abs(v - round(v)) < 1e-9 else f"{v:,.2f}"
        )
    )

    _style_axes(ax, color_map, grid_axis=None)
    ax.tick_params(axis="x", length=0, labelbottom=False)

    if chart_label:
        ax.set_title(
            chart_label,
            fontsize=17,
            color="#2E2E2E",
            fontweight="bold",
            pad=0,
            y=1.08
        )

    if show_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=len(series),
            frameon=False,
            fontsize=9,
        )

    # ================= [최종 평균선 그리기] =================
    if show_average:
        all_vals = [float(v) for s in series for v in (s.get("data") or []) if pd.notna(v)]
        if all_vals:
            avg_val = sum(all_vals) / len(all_vals)
            
            # 1. 점선 그리기 (빨간색 - 이번분기)
            ax.axhline(y=avg_val, color="#e53935", linestyle="--", linewidth=1.5, zorder=999)
            
            # 현재 기간 평균 라벨: "{연도}년 {분기}분기 평균" 형식으로 통일
            current_quarter = dataset.get("current_quarter")
            if current_quarter and current_quarter.get("year") and current_quarter.get("quarter"):
                cur_label = f"{current_quarter['year']}년 {current_quarter['quarter']}분기 평균"
            else:
                # current_quarter 메타가 없는 차트를 위한 기존 방식 유지
                if labels and len(labels) >= 2:
                    period_text = f"{str(labels[0])} ~ {str(labels[-1])}"
                else:
                    period_text = "분석 기간 전체"
                cur_label = f"{period_text} 평균"

            # 2. 텍스트 라벨 (왼쪽 고정, 빨간 글씨)
            ax.text(
                x=0.02,
                y=avg_val,
                s=f"{cur_label}: {avg_val:,.1f}{unit}",
                color="#e53935",
                fontsize=10,
                fontweight="bold",
                ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3),
                zorder=1000,
                transform=ax.get_yaxis_transform()
            )

            # 3. 차트 하단 평균선 기준 설명
            ax.text(
                0.01, -0.13,
                f"----- : {cur_label}",
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=10,
                color="#e53935",
                clip_on=False,
            )
            
            # 4. 이전 분기 평균선 (회색)
            if prev_quarter_avg and prev_quarter_avg.get("avg") is not None:
                prev_year = prev_quarter_avg.get("year")
                prev_quarter = prev_quarter_avg.get("quarter")

                # 4-1. 점선 그리기
                ax.axhline(y=prev_avg_val, color="#858585", linestyle="--", linewidth=1.5, zorder=998)

                # 4-2. 텍스트 라벨
                ax.text(
                    x=0.02,
                    y=prev_avg_val,
                    s=f"{prev_year}년 {prev_quarter}분기 평균: {prev_avg_val:,.1f}{unit}",
                    color="#858585",
                    fontsize=10,
                    fontweight="bold",
                    ha="left", va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3),
                    zorder=997,
                    transform=ax.get_yaxis_transform()
                )

                # 4-3. 차트 하단 설명 
                ax.text(
                    0.01, -0.24,
                    f"----- : {prev_year}년 {prev_quarter}분기 평균",
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=10,
                    color="#858585",
                    clip_on=False,
                )
    # =======================================================

    return _fig_to_svg(fig)

def render_bar_h_chart(
    dataset: Dict[str, Any],
    color_map: Dict[str, Any],
    compact: bool = False,
    chart_width: float = None,
    chart_height: float = None,
    palette: Optional[List[str]] = None,
) -> str:
    if not dataset:
        return ""
    labels = dataset.get("labels") or []
    series = dataset.get("series") or []
    if not labels or not series:
        return ""

    values = series[0].get("data") or []
    if not values:
        return ""

    labels = labels[: len(values)]
    y = list(range(len(labels)))
    if compact:
        width, height = 3.2, 1.8
    else:
        # Match example.py barh ratio (4:6) so charts are quarter-width and vertically long.
        width = chart_width if isinstance(chart_width, (int, float)) else 4.4
        height = chart_height if isinstance(chart_height, (int, float)) else 6.2
    fig, ax = plt.subplots(figsize=(width, height))

    colors = _value_colors(values, color_map, palette=palette)
    ax.barh(y, values, color=colors)
    ax.invert_yaxis()

    if compact:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5)
        unit = dataset.get("unit", "")
        if unit:
            ax.set_xlabel(unit, fontsize=9.5, color=color_map["muted"])
        _style_axes(ax, color_map, grid_axis=None)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=0.6)
    return _fig_to_svg(fig)

def _format_chart_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def render_bar_v_chart(
    dataset: Dict[str, Any],
    color_map: Dict[str, Any],
    compact: bool = False,
    show_labels: bool = False,
    show_values: bool = False,
) -> str:
    if not dataset:
        return ""
    labels = dataset.get("labels") or []
    series = dataset.get("series") or []
    if not labels or not series:
        return ""

    values = pd.Series(pd.to_numeric(series[0].get("data") or [], errors="coerce")).fillna(0.0).tolist()
    if not values:
        return ""

    labels = labels[: len(values)]
    x = list(range(len(labels)))
    if compact and (show_labels or show_values):
        fig_size = (3.4, 2.7)
    else:
        fig_size = (6.8, 3.8) if not compact else (3.4, 2.0)
    fig, ax = plt.subplots(figsize=fig_size)

    colors = _value_colors(values, color_map)
    bars = ax.bar(x, values, color=colors)

    max_val = max(values) if values else 0.0
    y_pad = max(max_val * 0.2, 0.4)
    y_top = max(max_val + y_pad, 1.0)
    ax.set_ylim(0, y_top)

    if show_values:
        unit = str(dataset.get("unit") or "").strip()
        suffix = unit if unit in {"%", "회", "명"} else ""
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_pad * 0.08,
                f"{_format_chart_value(float(value))}{suffix}",
                ha="center",
                va="bottom",
                fontsize=7 if compact else 8.5,
                color=color_map["muted"],
            )

    if compact:
        if show_labels:
            display_labels = [str(label).replace("<br>", "\n") for label in labels]
            ax.set_xticks(x)
            ax.set_xticklabels(display_labels, fontsize=6.5, rotation=0, ha="center")
            ax.tick_params(axis="x", length=0, pad=1, colors="#666666")
        else:
            ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5, rotation=30, ha="right")
        unit = dataset.get("unit", "")
        if unit:
            ax.set_ylabel(unit, fontsize=9.5, color=color_map["muted"])
        _style_axes(ax, color_map, grid_axis=None)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(pad=0.6)
    return _fig_to_svg(fig)


def _render_heatmap(rows: List[Dict[str, Any]], metric: str, color_map: Dict[str, Any]) -> str:
    df = pd.DataFrame(rows)
    if df.empty or metric not in df.columns:
        return ""
    if "age" not in df.columns or "gender" not in df.columns:
        return ""

    pivot = df.pivot_table(index="gender", columns="age", values=metric, aggfunc="mean")
    imp_pivot = None
    if metric == "ctr" and "impressions" in df.columns:
        imp_pivot = df.pivot_table(index="gender", columns="age", values="impressions", aggfunc="mean")

    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    gender_order = ["female", "male"]
    pivot = pivot.reindex(
        index=[g for g in gender_order if g in pivot.index],
        columns=[a for a in age_order if a in pivot.columns],
    )
    if imp_pivot is not None:
        imp_pivot = imp_pivot.reindex(index=pivot.index, columns=pivot.columns)
    if pivot.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    cmap = LinearSegmentedColormap.from_list(
        "theme",
        [color_map["lighter"], color_map["light"], color_map["base"], color_map["dark"]],
    )
    heat_values = pivot.values.astype(float)
    vmin = float(np.nanmin(heat_values))
    vmax = float(np.nanmax(heat_values))
    im = ax.imshow(heat_values, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=10, colors="#666666")
    if metric == "impressions":
        cbar.formatter = FuncFormatter(lambda x, _: f"{int(round(float(x))):,}")
        cbar.update_ticks()

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                continue
            norm = 0.5 if abs(vmax - vmin) < 1e-12 else (float(val) - vmin) / (vmax - vmin)
            cell_color = cmap(norm)
            if metric == "impressions":
                label = f"{int(round(float(val))):,}"
            else:
                label = f"{float(val):.2f}"
                if imp_pivot is not None:
                    imp_val = imp_pivot.iloc[i, j]
                    if pd.notna(imp_val):
                        label += f"\n({int(round(float(imp_val))):,})"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=11,
                color=_contrast_text_color(cell_color, threshold=0.45),
            )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(c) for c in pivot.index], fontsize=11)
    ax.tick_params(axis="x", bottom=True, top=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.6)
    return _fig_to_svg(fig)


def _render_simple_table(rows: List[Dict[str, Any]]) -> str:
    df = pd.DataFrame(rows)
    if df.empty:
        return ""

    max_rows = 12
    if len(df) > max_rows:
        df = df.head(max_rows)

    fig_height = 0.35 * len(df) + 1.2
    fig, ax = plt.subplots(figsize=(6, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.2)

    fig.tight_layout(pad=0.6)
    return _fig_to_svg(fig)


def render_table_chart(dataset: Dict[str, Any], color_map: Dict[str, Any], metric: str = None) -> str:
    if not dataset:
        return ""
    rows = dataset.get("rows") or []
    if not rows:
        return ""

    if metric == "purchases":
        heatmap_svg = _render_purchase_conversion_heatmap(rows, color_map)
        if heatmap_svg:
            return heatmap_svg

    if metric:
        heatmap_svg = _render_heatmap(rows, metric, color_map)
        if heatmap_svg:
            return heatmap_svg

    if "age" in rows[0] and "gender" in rows[0] and "ctr" in rows[0]:
        heatmap_svg = _render_heatmap(rows, "ctr", color_map)
        if heatmap_svg:
            return heatmap_svg

    return _render_simple_table(rows)


def render_content_card(dataset: Dict[str, Any], color_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not dataset:
        return []

    items = dataset.get("items") or []
    rendered = []

    for item in items:
        new_item = dict(item)
        details = item.get("target_details") or []
        chart_svg = ""

        if details:
            detail_df = pd.DataFrame(details)
            if (
                not detail_df.empty
                and {"age", "gender", "ctr"}.issubset(detail_df.columns)
            ):
                detail_df["gender"] = detail_df["gender"].astype(str).str.strip()
                detail_df = detail_df[detail_df["gender"].str.lower() != "unknown"]
                detail_df["ctr"] = pd.to_numeric(detail_df["ctr"], errors="coerce")
                detail_df = detail_df.dropna(subset=["ctr"])
                detail_df = detail_df[detail_df["ctr"] > 0]
                if detail_df.empty:
                    new_item["chart"] = chart_svg
                    rendered.append(new_item)
                    continue

                detail_df = detail_df.sort_values("ctr", ascending=False).head(6)
                labels = []
                for _, row in detail_df.iterrows():
                    age_text = str(row["age"]).strip()
                    gender_text = str(row["gender"]).strip()
                    gender_low = gender_text.lower()
                    if gender_low == "female":
                        gender_text = "여성"
                    elif gender_low == "male":
                        gender_text = "남성"
                    labels.append(f"{age_text}<br>{gender_text}")
                values = detail_df["ctr"].tolist()
                mini_ds = {
                    "kind": "bar_v",
                    "labels": labels,
                    "series": [{"name": "ctr", "data": values}],
                    "unit": "%",
                }
                chart_svg = render_bar_v_chart(
                    mini_ds,
                    color_map,
                    compact=True,
                    show_labels=True,
                    show_values=True,
                )

        new_item["chart"] = chart_svg
        rendered.append(new_item)

    return rendered


def render_reaction_bar(dataset: Dict[str, Any], color_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    반응 지표(좋아요/저장/공유) 상하위 콘텐츠 통합 렌더링.

    반환값 구조:
        {
            "items":      List[Dict]  - 썸네일·업로드일 정보 (template 상단 그리드용)
            "chart_svg":  str         - 가로형 막대그래프 SVG (template 하단 차트용)
        }

    차트 구성:
        - Y축: 각 콘텐츠의 업로드 날짜 (말줄임 없음, 날짜 형식으로 간결)
        - X축 막대: metric 수치, 막대 끝에 정확한 수치 표시
        - 최하단 막대: '전체 평균' 행 (색상으로 구분)
    """
    import io

    if not dataset:
        return {"items": [], "chart_svg": ""}

    items      = dataset.get("items") or []
    metric_col = dataset.get("metric_col", "total_likes")
    metric_avg = dataset.get("metric_avg", 0.0)
    metric     = dataset.get("metric", "likes")

    # 지표 한국어 레이블 매핑
    metric_label_map = {
        "likes":  "좋아요",
        "saves":  "저장",
        "shares": "공유",
    }
    metric_label = metric_label_map.get(metric, metric)

    # 상단 그리드용 아이템 리스트 (썸네일·업로드일·동순위 rank 전달)
    thumb_items = [
        {
            "thumbnail":     item.get("thumbnail"),
            "uploaded_at":   item.get("uploaded_at"),
            "ig_media_type": item.get("ig_media_type"),
            "ctr":           item.get("ctr"),
            "rank":          item.get("rank"),
            "caption_label": item.get("caption_label"),
        }
        for item in items[:5]
    ]

    if not items:
        return {"cards": [], "chart_svg": ""}

    labels, values, ranks, thumbnails = [], [], [], []
    for idx, item in enumerate(items, 1):
        caption_label = (
            str(item.get("caption_label") or "")
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )
        label = caption_label or str(item.get("uploaded_at") or f"콘텐츠 {idx}")
        labels.append(label)
        values.append(float(item.get(metric_col, 0) or 0))
        ranks.append(item.get("rank", idx))
        thumbnails.append(item.get("thumbnail"))

    n = len(labels)
   

    TOP_STEP    = 1.43   # ★ 상위(1~5위) 막대끼리 간격 (키우면 상위 막대들이 벌어짐)
    BOTTOM_STEP = 1.43   # ★ 하위(6위~) 막대끼리 간격 (키우면 하위 막대들이 벌어짐)
    GAP_AFTER_5 = 0.17   # 상위그룹 ↔ 하위그룹 사이 추가 간격

    y_pos = [0.0] * n
    y = 0.0
    # 맨 아래(꼴찌)부터 위로 한 칸씩 쌓는다
    for i in range(n - 1, -1, -1):
        y_pos[i] = y
        if i > 5:
            y += BOTTOM_STEP            # 하위 막대끼리 간격
        elif i == 5:
            y += BOTTOM_STEP + GAP_AFTER_5   # 그룹 경계
        else:
            y += TOP_STEP


    # ── 막대 색상 결정 (평균 이상 = 진하게, 평균 이하 = 연하게) ──
    base_hex = color_map.get("base", "#999999")
    is_top   = dataset.get("is_top", True)

    if not values:
        highlight_val = None
    elif is_top:
        highlight_val = max(values)
    else:
        highlight_val = min(values)

    bar_colors = []
    for v in values:
        r, g, b = _hex_to_rgb01(base_hex)
        if highlight_val is not None and v == highlight_val:
            bar_colors.append((r, g, b, 1.0))
        else:
            bar_colors.append((r, g, b, 0.4))

    x_scale_max_from_json = float(dataset.get("x_scale_max") or 0)
    x_data_max = max(values) if values else 0
    if x_scale_max_from_json > 0:
        # JSON에서 공유 기준값이 전달된 경우: 해당 값을 우선 사용
        x_scale_ref = x_scale_max_from_json
    elif x_data_max > 0:
        # JSON 값이 없으면 현재 데이터 최대값으로 대체
        x_scale_ref = x_data_max
    else:
        # 모든 값이 0인 경우: 최소값 1 보장 (division by zero 방지)
        x_scale_ref = 1

    # ── 그래프 캔버스 생성 ──
    all_zero = all(v == 0 for v in values) if values else True
    # 검증후조정: 차트가 우측 폭(좌측 33%로 축소되며 넓어짐)에 맞춰 height:auto로 렌더되면
    # 세로가 과대해져 제목과 한 페이지에 못 들어가 다음 페이지로 넘어갔다.
    # 막대 겹침은 데이터 단위(GAP/ylim/height)로 제어되므로 fig_h를 낮춰도 안전하다.
    fig_h = min(5.7, max(2.6, n * 0.50))
    fig, ax = plt.subplots(figsize=(7, fig_h))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    # 하위 썸네일을 막대 영역 안쪽 좌측에 배치하므로 left 여백을 줄인다.
    fig.subplots_adjust(left=0.06, right=0.97, top=0.97, bottom=0.08)

    # 전부 0이면 막대를 최소 폭으로 표시
    plot_values = [x_scale_ref * 0.006 if all_zero else v for v in values]

    bars = ax.barh(
        y_pos, plot_values,
        color=bar_colors,
        height=0.85,   # 검증후조정(0.68~0.8) — 막대 굵게
        edgecolor="none",
    )

    # y축 위아래 여백 명시 — 막대·라벨이 캔버스 경계와 겹치지 않도록 (검증후조정)
    # 여백을 줄여 막대가 차트 높이를 더 채우도록(=두꺼워 보이도록) 한다.
    if y_pos:
        ax.set_ylim(min(y_pos) - 0.5, max(y_pos) + 0.5)


    # 지표별 유니코드 아이콘 매핑
    metric_icon_map = {
        "likes":  "\u2665",   # ♥ 하트 (좋아요)
        "saves":  "\u2605",   # ★ 별 (저장)
        "shares": "\u21a6",   # ↦ 오른쪽 화살표 (공유)
    }
    metric_icon = metric_icon_map.get(metric, "")

    for bar, val in zip(bars, values):
        ax.text(
            # 막대 끝 오프셋을 x_scale_ref 기준으로 계산하여 하위 차트에서
            # 수치가 작아도 텍스트가 과도하게 우측으로 밀리지 않도록 한다.
            bar.get_width() + x_scale_ref * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{metric_icon} {int(val):,}",
            va="center", ha="left",
            fontsize=9, color="#333333",   # 검증후조정 — 막대 끝 수치 가독성 확대
            zorder=10,
            bbox=dict(
                facecolor=(1.0, 1.0, 1.0, 0.5),
                edgecolor="none",
                pad=1.5,
                boxstyle="round,pad=0.15",
            ),
        )
    ax.set_yticks(y_pos)
    # 상·하위 모두 제목을 막대 안쪽 흰색 라벨로 통일하므로 y축 바깥 라벨은 전부 비운다.
    ax.set_yticklabels(["" for _ in labels], fontsize=7, color="#333333")
    ax.yaxis.set_tick_params(length=0)

    # 1~5위(상위): 막대 안쪽 왼쪽에 흰색으로 제목 표시
    # (하위 6위~ 라벨/썸네일은 x축 범위 확정 후 아래 별도 블록에서 처리)
    for i, (bar, lbl) in enumerate(zip(bars, labels)):
        if i >= 5:
            break
        ax.text(
            x_scale_ref * 0.012,
            bar.get_y() + bar.get_height() / 2,
            lbl,
            va="center", ha="left",
            fontsize=9,   # 검증후조정 — 막대 안 제목 가독성 확대
            color="#FFFFFF",
            zorder=11,
        )

    ax.set_xlabel(metric_label, fontsize=8, color="#555555", labelpad=4)
    # x축 상한을 x_scale_ref 기준으로 설정한다.
    # 1.18 배율은 수치 텍스트가 막대 바깥에 표시될 공간을 확보하기 위함이다.
    ax.set_xlim(0, x_scale_ref * 1.18)   # 수치 텍스트 공간 확보
    ax.xaxis.set_visible(False)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── 하위(6위~) 막대: 막대 영역 안쪽 좌측 인라인 썸네일 + 막대 안쪽 흰색 라벨 ──
    # 상위 5개는 좌측 그리드에 이미 노출되므로 썸네일을 추가하지 않는다.
    if n > 5:
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        # 데이터↔픽셀 변환을 확정하기 위해 한 번 렌더한다(xlim/ylim 확정 후).
        fig.canvas.draw()
        _p0 = ax.transData.transform((0.0, 0.0))
        _p1y = ax.transData.transform((0.0, 1.0))
        _p1x = ax.transData.transform((1.0, 0.0))
        px_per_unit_y = abs(_p1y[1] - _p0[1]) or 1.0
        px_per_unit_x = abs(_p1x[0] - _p0[0]) or 1.0

        # 썸네일 표시 높이를 막대 높이(0.55 데이터 단위)와 비슷하게 맞춘다. // 검증후조정
        thumb_data_h = 1.5  # 검증후조정 — 하위 썸네일을 막대보다 약간 크게 확대
        target_px_h  = thumb_data_h * px_per_unit_y
        # OffsetImage(zoom)은 dpi 보정이 적용되므로 72/dpi 로 환산해 픽셀 높이를 맞춘다.
        _dpi = fig.get_dpi() or 100.0

        for i in range(5, n):
            thumb = thumbnails[i] if i < len(thumbnails) else None
            arr = _load_thumbnail_array(thumb) if thumb else None
            if arr is None:
                # URL 없음/로드 실패 → 회색(#D0D0D0) placeholder (4:5 = 240×300)
                arr = np.full((300, 240, 3), 0xD0, dtype=np.uint8)

            img_h_px = arr.shape[0]
            img_w_px = arr.shape[1]
            zoom = (target_px_h / img_h_px) * (72.0 / _dpi)   # // 검증후조정

            oimg = OffsetImage(arr, zoom=zoom)
            ab = AnnotationBbox(
                oimg,
                (0.0, y_pos[i]),                       # 축 좌측 안쪽(x=0), 막대 중심 y
                xycoords=("axes fraction", "data"),
                box_alignment=(0.0, 0.5),
                frameon=False,
                pad=0.0,
                zorder=12,
            )
            ax.add_artist(ab)

            # 라벨 x 시작점을 썸네일 표시 폭 뒤로 두어 겹치지 않게 한다.
            thumb_w_px   = img_w_px * zoom * (_dpi / 72.0)
            thumb_w_data = thumb_w_px / px_per_unit_x
            label_x = thumb_w_data + x_scale_ref * 0.012
            ax.text(
                label_x,
                y_pos[i],
                labels[i],
                va="center", ha="left",
                fontsize=9,   # 검증후조정 — 하위 막대 안 제목 가독성 확대
                color="#FFFFFF",
                zorder=13,
            )

    # 5위/6위 사이 가로 구분선 제거 (목표 디자인에 없음).
    # 상·하위 간격은 GAP_AFTER_5로 유지하고 선만 삭제한다.

    # 전부 0일 때 안내 문구
    if all_zero:
        ax.text(
            0.5, -0.08,
            "해당 기간 집계된 데이터가 없습니다",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7, color="#aaaaaa",
        )



    # 전체 평균 세로 점선 렌더링.
    _raw_avg = dataset.get("overall_avg")
    if _raw_avg is None:
        _raw_avg = dataset.get("metric_avg")
    if _raw_avg is None:
        _raw_avg = 0
    overall_avg_val = float(_raw_avg)

    avg_label_text = f"평균 : {int(round(overall_avg_val)):,}"

    if x_scale_max_from_json > 0 and overall_avg_val > 0:
        # 평균값이 유효한 경우에만 선과 레이블을 표시
        ax.axvline(
            x=overall_avg_val,
            color="#888888",
            linestyle=(0, (4, 4)),
            linewidth=1.2,
            zorder=5,
        )
        ax.text(
            overall_avg_val,
            -0.06,
            avg_label_text,
            ha="center",
            va="top",
            fontsize=8,
            color="#888888",
            zorder=6,
            bbox=dict(
                facecolor=(1.0, 1.0, 1.0, 0.5),
                edgecolor="none",
                pad=2,
                boxstyle="round,pad=0.2",
            ),
            transform=ax.get_xaxis_transform(),
        )
    # else 분기 제거: 평균이 0이거나 scale이 없으면 선을 그리지 않음

    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    plt.rcParams["svg.fonttype"] = "path"

    svg = buf.getvalue()
    idx = svg.find("<svg")
    chart_svg = svg[idx:] if idx != -1 else svg

    return {"cards": thumb_items, "chart_svg": chart_svg}



def render_target_spend_bubble(dataset: Dict[str, Any], color_map: Dict[str, Any]) -> str:
    """타겟(연령×성별) 광고비 비중 버블 그리드 (히트맵 스타일)."""
    from matplotlib.patches import Patch
    import io
    import pandas as pd

    rows = dataset.get("rows") or []
    if not rows:
        return ""

    main_age    = dataset.get("main_age")
    main_gender = dataset.get("main_gender")
    avoid_age   = dataset.get("avoid_age")
    avoid_gender= dataset.get("avoid_gender")

    df = pd.DataFrame(rows)
    df["spend_ratio"] = pd.to_numeric(df["spend_ratio"], errors="coerce").fillna(0)
    df["cpc"]         = pd.to_numeric(df["cpc"],         errors="coerce").fillna(0)

    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    ages    = [a for a in age_order if a in df["age_range"].values]
    genders = ["female", "male"]
    gender_labels = {"female": "여성", "male": "남성"}

    def _canon_age(s):
        return str(s).strip().lower().replace("_", "-").replace("–", "-").replace(" ", "")

    def _norm(val):
        if val is None: return []
        if isinstance(val, (list, tuple)):
            return [_canon_age(v) for v in val if v]
        s = _canon_age(val)
        return [s] if s else []

    def _map_g(g):
        g = g.lower()
        return "female" if g in ("f", "여성", "female") else \
               "male"   if g in ("m", "남성", "male")   else g

    main_ages_n    = _norm(main_age)
    main_genders_n = [_map_g(g) for g in _norm(main_gender)]
    avoid_ages_n   = _norm(avoid_age)
    avoid_genders_n= [_map_g(g) for g in _norm(avoid_gender)]

    COLOR_MAIN  = "#d0f4bc"   # 메인 타겟 (초록)
    COLOR_AVOID = "#b84541"   # 기피 타겟 (빨강)
    COLOR_MID   = "#f9bd63"   # 중간 (주황)

    def cell_color(age, gender):
        a, g = _canon_age(age), gender.lower()
        is_main  = (not main_ages_n  or a in main_ages_n)  and \
                   (not main_genders_n or g in main_genders_n)
        is_avoid = bool(avoid_ages_n) and a in avoid_ages_n and \
                   (not avoid_genders_n or g in avoid_genders_n)
        if is_avoid: return COLOR_AVOID
        if is_main:  return COLOR_MAIN
        return COLOR_MID

    n_ages = len(ages)
    
    plt.rcParams["svg.fonttype"] = "none"

    fig_w = 7.5   # 고정 너비 (필요에 따라 조정)
    fig_h = 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # 양옆(left, right) 여백을 극단적으로 줄여 차트를 꽉 채웁니다.
    fig.subplots_adjust(top=0.88, bottom=0.32, left=0.07, right=0.90)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.grid(False)

    max_ratio = df["spend_ratio"].max() or 1
    
    BASE = 3500
    
    plot_width = fig_w * (0.90 - 0.07)
    col_width = plot_width / n_ages
    col_pt  = col_width * 72
    MAX_S   = (col_pt * 0.72) ** 2    # 72%로 축소하여 최대 버블이 셀 안에 머물도록 함
    MIN_S   = MAX_S * 0.12

    for j, age in enumerate(ages):
        for i, gender in enumerate(genders):
            row = df[(df["age_range"] == age) & (df["gender"] == gender)]
            if row.empty:
                continue

            ratio = float(row["spend_ratio"].iloc[0])
            cpc   = float(row["cpc"].iloc[0])
            size  = MIN_S + (MAX_S - MIN_S) * (ratio / max_ratio)
            color = cell_color(age, gender)

            ax.scatter(
                j, -i, s=size, color=color,
                edgecolors="white", linewidth=1.5, alpha=0.9, zorder=2,
                clip_on=False,  # [수정 2] 축 경계에 의한 버블 클리핑을 비활성화
            )

            ax.text(j, -i, f"{ratio:.1f}%\n{int(cpc):,}원",
                    ha="center", va="center",
                    fontsize=5, fontweight="bold", zorder=5,
                    clip_on=False,  # 텍스트도 동일하게 클리핑 해제
            )

    # 축 라벨 폰트 세팅
    ax.set_xticks(range(n_ages))
    ax.set_xticklabels(ages, fontsize=9, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    
    ax.set_yticks([0, -1])
    ax.set_yticklabels([gender_labels.get(g, g) for g in genders], fontsize=9, fontweight="bold")
    
    ax.set_xlim(-0.52, n_ages - 0.48)
    ax.set_ylim(-1.65, 0.65)

    legend_els = [
        Patch(facecolor=COLOR_MAIN, label="메인타겟"),
        Patch(facecolor=COLOR_MID, label="중간타겟"),
        Patch(facecolor=COLOR_AVOID, label="기피타겟"),
    ]
    leg = ax.legend(
    handles=legend_els,
    loc='lower left',                      # 기준점: 좌하단
    bbox_to_anchor=(0.0, -0.28),           # axes 좌표 기준, Y=-0.28로 고정
    bbox_transform=ax.transAxes,           # axes 비율 좌표계 기준
    ncol=3,
    fontsize=5,
    frameon=False,
    borderpad=0,
    handletextpad=0.4,
    title="[색상범례]",
    title_fontproperties={'weight': 'bold', 'size': 5}
)
    leg._legend_box.align = "left"
    leg.get_title().set_ha("left")

    ax.text(
        0.0, -0.28,                           # x: 색상범례 우측 공간, y: 범례와 동일 기준
        "원 크기 = 광고비 비중(%)\n원 안 숫자 = CPC",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=5, color="#555555", linespacing=1.6
    )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    buf = io.StringIO()

    fig.savefig(buf, format="svg")
    plt.close(fig)
    plt.rcParams["svg.fonttype"] = "path"

    svg = buf.getvalue()
    idx = svg.find("<svg")
    return svg[idx:] if idx != -1 else svg


def render_dataset(dataset: Dict[str, Any], color_map: Dict[str, Any], **kwargs):
    if not dataset:
        return ""

    kind = dataset.get("kind")
    renderers = {
        "line": render_line_chart,
        "bar_h": render_bar_h_chart,
        "bar_v": render_bar_v_chart,
        "bubble": render_bubble_chart,
        "table": render_table_chart,
        "content_card": render_content_card,
        "reaction_card": render_reaction_bar,
        "reaction_bar":  render_reaction_bar,          # ← 추가
        "target_bubble":  render_target_spend_bubble, # ← 추가
    }

    renderer = renderers.get(kind)
    if not renderer:
        return ""

    return renderer(dataset, color_map, **kwargs)

def render_bubble_chart(
    dataset: Dict[str, Any],
    color_map: Dict[str, Any],
    compact: bool = True,
    palette: Optional[List[str]] = None,
) -> str:
    labels = dataset.get("labels") or []
    series = dataset.get("series") or []
    if not labels or not series:
        return ""

    ctr_raw = pd.to_numeric(series[0].get("data") or [], errors="coerce")
    size_raw = ctr_raw
    if len(series) > 1:
        size_raw = pd.to_numeric(series[1].get("data") or [], errors="coerce")

    n = min(len(labels), len(ctr_raw), len(size_raw))
    if n == 0:
        return ""

    rows = []
    for i in range(n):
        ctr = ctr_raw[i]
        size_val = size_raw[i]
        if pd.isna(ctr) or pd.isna(size_val):
            continue
        rows.append({
            "label": str(labels[i]),
            "ctr": float(ctr),
            "size": max(float(size_val), 1.0),
        })
    if not rows:
        return ""

    rows.sort(key=lambda x: x["size"], reverse=True)

    size_values = np.sqrt(np.array([r["size"] for r in rows], dtype=float))
    size_norm = size_values / size_values.max() if size_values.max() > 0 else np.ones(len(rows))
    max_r = 0.58 if len(rows) <= 3 else (0.45 if len(rows) <= 6 else 0.36)
    min_r = max(0.12, max_r * 0.38)
    radii = (min_r + (max_r - min_r) * size_norm).tolist()

    positions = [np.array([0.0, 0.0])]
    if len(radii) > 1:
        positions.append(np.array([radii[0] + radii[1], 0.0]))

    for i in range(2, len(radii)):
        r_new = radii[i]
        placed = False
        for j in range(len(positions)):
            for k in range(j + 1, len(positions)):
                r1, r2 = radii[j], radii[k]
                p1, p2 = positions[j], positions[k]
                dist = np.linalg.norm(p1 - p2)
                if dist <= 1e-9:
                    continue
                if dist > (r1 + r_new) + (r2 + r_new):
                    continue

                d1 = r1 + r_new
                d2 = r2 + r_new
                a = (d1**2 - d2**2 + dist**2) / (2 * dist)
                h = np.sqrt(max(0.0, d1**2 - a**2))
                p3 = p1 + a * (p2 - p1) / dist

                for sign in (-1, 1):
                    test_pos = np.array([
                        p3[0] + sign * h * (p2[1] - p1[1]) / dist,
                        p3[1] - sign * h * (p2[0] - p1[0]) / dist,
                    ])
                    if all(
                        np.linalg.norm(test_pos - p) >= (radii[idx] + r_new) * 0.99
                        for idx, p in enumerate(positions)
                    ):
                        positions.append(test_pos)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            positions.append(np.array([radii[0] + r_new, r_new * i]))

    if palette:
        bubble_palette = palette
    else:
        bubble_palette = [
            color_map["lighter"],
            color_map["light"],
            color_map["base"],
            color_map["dark"],
        ]
    cmap = LinearSegmentedColormap.from_list("bubble_palette", bubble_palette, N=256).reversed()

    ctr_values = [r["ctr"] for r in rows]
    vmin, vmax = min(ctr_values), max(ctr_values)

    fig_size = (4, 4) if compact else (5.4, 5.4)
    fig, ax = plt.subplots(figsize=fig_size)

    for idx, pos in enumerate(positions):
        ctr = rows[idx]["ctr"]
        label = rows[idx]["label"]
        radius = radii[idx]
        norm = 0.5 if abs(vmax - vmin) < 1e-12 else (ctr - vmin) / (vmax - vmin)
        color = cmap(norm)
        circle = plt.Circle(pos, radius, facecolor=color, edgecolor="white", linewidth=1.6, alpha=1.0)
        ax.add_patch(circle)

        font_size = max(7, min(13, 5 + radius * 11))
        ax.text(
            pos[0],
            pos[1],
            f"{label}\n({ctr:.2f}%)",
            fontsize=font_size,
            ha="center",
            va="center",
            color=_contrast_text_color(color, threshold=0.45),
            fontweight="bold",
        )

    all_points = np.array(
        [p + np.array([r, r]) for p, r in zip(positions, radii)] +
        [p - np.array([r, r]) for p, r in zip(positions, radii)]
    )
    x_min, x_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    y_min, y_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    x_pad = max((x_max - x_min) * 0.05, 0.08)
    y_pad = max((y_max - y_min) * 0.05, 0.08)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal")
    ax.axis("off")

    return _fig_to_svg(fig)


# 구매 콘텐츠 파이차트
def render_purchase_pie_chart(rows: List[Dict[str, Any]], color_map: Dict[str, Any]) -> str:
    df = pd.DataFrame(rows)
    if df.empty or "purchases" not in df.columns:
        return ""

    df = df.copy()
    df["purchases"] = pd.to_numeric(df["purchases"], errors="coerce").fillna(0)
    df = df[df["purchases"] > 0]
    if df.empty:
        return ""

    def _gender_label(g):
        g = str(g).strip().lower()
        if g == "female":
            return "여성"
        if g == "male":
            return "남성"
        return str(g)

    labels = [
        f"{str(row['age']).strip()} {_gender_label(row['gender'])}"
        for _, row in df.iterrows()
    ]
    values = df["purchases"].tolist()
    total = int(df["purchases"].sum())
    percentages = [(v / total) * 100 for v in values]

    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    colors = _value_colors(
        values,
        color_map,
        palette=[
            color_map["dark"],
            color_map["base"],
            color_map["light"],
            color_map["lighter"],
        ],
    )

    pie_radius = 1.21

    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=pie_radius,
        wedgeprops=dict(edgecolor="none")
    )

    label_items = []

    for i, w in enumerate(wedges):
        angle = (w.theta2 + w.theta1) / 2.0
        x = np.cos(np.deg2rad(angle))
        y = np.sin(np.deg2rad(angle))

        label_items.append({
            "idx": i,
            "x": x,
            "y": y,
            "line_start": (x * pie_radius * 0.96, y * pie_radius * 0.96),
            "label_x": 1.34 if x >= 0 else -1.34,
            "label_y": y * 1.08,
            "ha": "left" if x >= 0 else "right",
            "side": "right" if x >= 0 else "left",
        })

    # 한쪽에 라벨이 너무 몰리면 구매건수가 작은 조각부터 반대편으로 분산
    left_items = [d for d in label_items if d["side"] == "left"]
    right_items = [d for d in label_items if d["side"] == "right"]

    if len(left_items) - len(right_items) >= 2:
        move_count = (len(left_items) - len(right_items)) // 2
        left_items_sorted = sorted(left_items, key=lambda d: values[d["idx"]])  
        for item in left_items_sorted[:move_count]:
            item["side"] = "right"
            item["label_x"] = 1.34
            item["ha"] = "left"

    elif len(right_items) - len(left_items) >= 2:
        move_count = (len(right_items) - len(left_items)) // 2
        right_items_sorted = sorted(right_items, key=lambda d: values[d["idx"]]) 
        for item in right_items_sorted[:move_count]:
            item["side"] = "left"
            item["label_x"] = -1.34
            item["ha"] = "right"

    # 같은 쪽 라벨끼리 간격 벌려서 겹침 방지
    def _adjust(items, min_gap=0.32):
        items = sorted(items, key=lambda d: d["label_y"], reverse=True)
        for j in range(1, len(items)):
            upper = items[j - 1]
            cur = items[j]
            if upper["label_y"] - cur["label_y"] < min_gap:
                cur["label_y"] = upper["label_y"] - min_gap
        return items

    left_items = _adjust([d for d in label_items if d["side"] == "left"], min_gap=0.32)
    right_items = _adjust([d for d in label_items if d["side"] == "right"], min_gap=0.32)

    adjusted = {d["idx"]: d for d in left_items + right_items}

    for i in range(len(labels)):
        item = adjusted[i]
        label_x = item["label_x"]
        label_y = item["label_y"]
        ha = item["ha"]

        ax.annotate(
            "",
            xy=item["line_start"],
            xytext=(label_x, label_y),
            arrowprops=dict(
                arrowstyle="-",
                color="#8c8c8c",
                lw=0.8,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="angle3,angleA=0,angleB=90"
            )
        )

        ax.text(
            label_x,
            label_y + 0.04,
            labels[i],
            ha=ha,
            va="bottom",
            fontsize=11,
            color="#555555"
        )

        ax.text(
            label_x,
            label_y - 0.01,
            f"{percentages[i]:.1f}%",
            ha=ha,
            va="top",
            fontsize=10,
            color="#9a9a9a"
        )

    # 차트 크기 일정하게 보이도록 축 범위 고정
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.45, 1.45)

    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.08)
    return _fig_to_svg(fig)

# -----------------------------------
# 팔로워 인구통계 추가

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=350, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

# 팔로워 연령대별 누적 가로막대 차트
def render_follower_age_gender_stacked_barh_chart(chart_data, color_map):
    labels = chart_data.get("labels", [])
    series = chart_data.get("series", [])

    if not labels or not series:
        return ""

    # 시리즈 이름/데이터 정리
    parsed = []
    for s in series:
        name = str(s.get("name", "")).strip().lower()
        data = pd.to_numeric(pd.Series(s.get("data", [])), errors="coerce").fillna(0).tolist()

        if name in ["male", "남성"]:
            parsed.append(("남성", data, color_map["base"], "white"))
        elif name in ["female", "여성"]:
            parsed.append(("여성", data, color_map["lighter"], "#252525"))
        elif name in ["unknown", "알 수 없음"]:
            parsed.append(("알 수 없음", data, color_map["dark"], "white"))
        elif name in ["known", "남/여 전체"]:
            parsed.append(("남/여 전체", data, color_map["lighter"], "#252525"))

    if not parsed:
        return ""

    # 길이 맞추기
    n = len(labels)
    normalized = []
    for display_name, data, color, text_color in parsed:
        if len(data) < n:
            data = data + [0] * (n - len(data))
        elif len(data) > n:
            data = data[:n]
        normalized.append((display_name, data, color, text_color))

    df = pd.DataFrame({"age_range": labels})
    for i, (display_name, data, color, text_color) in enumerate(normalized):
        df[f"v{i}"] = data

    value_cols = [f"v{i}" for i in range(len(normalized))]
    df["total"] = df[value_cols].sum(axis=1)
    df = df[df["total"] > 0]
    if df.empty:
        return ""

    age_order = ["13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["age_order"] = df["age_range"].apply(lambda x: age_order.index(x) if x in age_order else 99)
    df = df.sort_values("age_order", ascending=False)

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.5, 5))

    left = np.zeros(len(df))
    for i, (display_name, data, color, text_color) in enumerate(normalized):
        vals = df[f"v{i}"].tolist()

        ax.barh(
            y, vals, left=left,
            color=color, edgecolor="none", height=0.89,
            label=display_name
        )

        for j, v in enumerate(vals):
            if v >= max(df["total"]) * 0.08:   # 너무 작은 값은 라벨 생략
                ax.text(
                    left[j] + v / 2, j, f"{int(v):,}",
                    ha="center", va="center",
                    fontsize=13, color=text_color, fontweight="bold"
                )
        left += np.array(vals)

    ax.set_yticks(y)
    ax.set_yticklabels(df["age_range"].astype(str).tolist(), fontsize=14)

    max_total = df["total"].max()
    ax.set_xlim(0, max_total * 1.12)

    ax.tick_params(axis="x", labelsize=13, colors="#666666")
    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.12)
    return fig_to_base64(fig)

#--------------------------------
# 팔로워 성별 도넛 차트
def render_follower_gender_doughnut_chart(chart_data, color_map):
    labels = chart_data.get("labels", [])
    series = chart_data.get("series", [])

    if not labels or not series:
        return ""

    values = series[0].get("data", [])
    if not values:
        return ""

    values = [float(v) for v in values]

    def _pick_color(label: str) -> str:
        label = str(label).strip()
        if label == "여성":
            return color_map["lighter"]
        elif label == "남성":
            return color_map["base"]
        elif label in ["남/여 전체", "연령 확인 가능", "확인 가능", "Known"]:
            return color_map["lighter"]
        elif label in ["알 수 없음", "Unknown", "unknown"]:
            return color_map["dark"]
        return color_map["dark"]

    colors = [_pick_color(label) for label in labels]

    fig, ax = plt.subplots(figsize=(5.4, 4.8))

    fig.suptitle(
        chart_data.get("title", ""),
        y=0.94,
        fontsize=14,
        fontweight="bold"
    )

    def autopct_func(pct):
        return f"{pct:.0f}%" if pct >= 5 else ""

    wedges, texts, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        radius=0.98,
        wedgeprops=dict(width=0.48, edgecolor="white"),
        autopct=autopct_func,
        pctdistance=0.74
    )

    for i, t in enumerate(autotexts):
        if colors[i] == color_map["lighter"]:
            t.set_color("#252525")
        else:
            t.set_color("white")

        t.set_fontsize(14)
        t.set_fontweight("bold")

    total = sum(values)

    center_text = chart_data.get("center_text")
    center_subtext = chart_data.get("center_subtext")

    if center_text is not None:
        center_main = str(center_text)
        center_sub = str(center_subtext or "")
    else:
        is_ratio_chart = 99.5 <= total <= 100.5
        center_main = "100%" if is_ratio_chart else f"{int(total):,}"
        center_sub = "비율" if is_ratio_chart else "팔로워"

    ax.text(
        0, 0.05, center_main,
        ha="center", va="center",
        fontsize=17, fontweight="bold"
    )
    ax.text(
        0, -0.16, center_sub,
        ha="center", va="center",
        fontsize=13, color="#666"
    )

    ax.legend(
        wedges, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=11,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0
    )

    ax.set(aspect="equal")
    ax.axis("off")

    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.05, right=0.95)
    return fig_to_base64(fig)

# 구매전환 히트맵
def _render_purchase_conversion_heatmap(
    rows: List[Dict[str, Any]],
    color_map: Dict[str, Any],
) -> str:
    df = pd.DataFrame(rows)
    if df.empty or "purchases" not in df.columns:
        return ""
    if "age" not in df.columns or "gender" not in df.columns:
        return ""

    pivot = df.pivot_table(index="gender", columns="age", values="purchases", aggfunc="sum")

    age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    gender_order = ["female", "male"]
    pivot = pivot.reindex(
        index=gender_order,    
        columns=age_order,    
        fill_value=0,
    )

    if pivot.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    cmap = LinearSegmentedColormap.from_list(
        "theme",
        [color_map["lighter"], color_map["light"], color_map["base"], color_map["dark"]],
    )

    pivot = pivot.fillna(0)
    heat_values = pivot.values.astype(float)
    vmin = float(np.nanmin(heat_values))
    vmax = float(np.nanmax(heat_values))

    im = ax.imshow(heat_values, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.035)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=10, colors="#666666")
    cbar.formatter = FuncFormatter(lambda x, _: f"{int(round(float(x))):,}")
    cbar.update_ticks()

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                continue

            norm = 0.5 if abs(vmax - vmin) < 1e-12 else (float(val) - vmin) / (vmax - vmin)
            cell_color = cmap(norm)

            ax.text(
                j,
                i,
                f"{int(round(float(val))):,}",
                ha="center",
                va="center",
                fontsize=11,
                linespacing=1.2,
                color=_contrast_text_color(cell_color, threshold=0.45),
            )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(c) for c in pivot.index], fontsize=11)
    ax.tick_params(axis="x", bottom=True, top=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.6)
    return _fig_to_svg(fig)



# ─────────────────────────────────────────────────────────────
# CTR × 팔로우 4사분면 스캐터 + K-Means 대표 콘텐츠 레이아웃
# ─────────────────────────────────────────────────────────────

def _load_thumbnail_array(path: str):
    """
    로컬 경로 또는 URL에서 이미지를 읽어 numpy array로 반환.
    실패 시 None 반환 (호출부에서 placeholder 처리).
    """
    try:
        from PIL import Image
        import numpy as np

        # './static/thumbnail/xxx.jpg' 형태의 상대 경로 처리
        clean = path.strip()
        if clean.startswith("./"):
            clean = clean[2:]

        img = Image.open(clean).convert("RGB")
        # 정사각형 크롭 (중앙)
        w, h = img.size
        target_ratio = 4 / 5
        img_ratio = w / h


        if img_ratio > target_ratio:
            # 가로가 너무 길면 양옆을 자름
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        else:
            # 세로가 너무 길면 위아래를 자름
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
            
        # 4:5 비율인 (240, 300) 사이즈로 리사이징
        img = img.resize((240, 300), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None



def _draw_thumbnail_cell(ax, img_array, label_text: str = "", bg_color: str = "#f0f0f0"):
    """
    단일 subplot axis에 썸네일 이미지(또는 placeholder)를 채워 그립니다.
    """
    ax.axis("off")
    if img_array is not None:
        # aspect="equal"로 변경하여 이미지 종횡비를 유지.
        # 셀 크기에 맞게 중앙 정렬(extent 미지정 시 기본값)로 렌더링된다.
        ax.imshow(img_array, aspect="equal")
    else:
        # placeholder: 배경 색상 + 텍스트
        ax.set_facecolor(bg_color)
        ax.text(
            0.5, 0.5, "이미지\n없음",
            ha="center", va="center",
            fontsize=9, color="#888888",
            transform=ax.transAxes,
        )

    if label_text:
        ax.set_xlabel(
            label_text,
            fontsize=8, color="#555555",
            labelpad=3,
        )


def _get_quadrant_representatives(df_quad: pd.DataFrame, n_clusters: int = 2) -> list[dict]:
    """
    사분면 내 K-Means (k=2) 클러스터링으로 대표 콘텐츠 2개 선정.
    StandardScaler 정규화 후 각 군집 중심점과 유클리디안 거리가 가장 가까운
    실제 데이터 포인트를 반환합니다.

    예외 처리:
        - 데이터 0개 → 빈 리스트
        - 데이터 1개 → 그 1개를 리스트로 반환
        - 데이터 2개 → 두 행 그대로 반환 (K-Means 불필요)
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    n = len(df_quad)

    if n == 0:
        return []
    if n <= 2:
        return df_quad.to_dict("records")

    X_raw = df_quad[["ctr", "follows"]].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(X_scaled)

    representatives = []
    used_indices = set()

    for centroid in km.cluster_centers_:
        dists = np.linalg.norm(X_scaled - centroid, axis=1)
        # 이미 선택된 인덱스는 건너뜀 (중복 방지)
        for idx in np.argsort(dists):
            if idx not in used_indices:
                representatives.append(df_quad.iloc[idx].to_dict())
                used_indices.add(idx)
                break

    return representatives


def render_ctr_follows_quadrant_chart(
    scatter_data: list[dict],
    ctr_median: float,
    follows_median: float,
) -> str:
    ctr_mean     = ctr_median
    follows_mean = follows_median
    """
    CTR × 팔로우 4사분면 스캐터 플롯 + K-Means 대표 콘텐츠 썸네일 레이아웃.
    """
    
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    import os

    if not scatter_data:
        return ""

    df = pd.DataFrame(scatter_data)
    df["ctr"]     = pd.to_numeric(df["ctr"],     errors="coerce").fillna(0.0)
    df["follows"] = pd.to_numeric(df["follows"], errors="coerce").fillna(0.0)

    QUAD_COLORS = {
        "Q1": "#2563EB", "Q2": "#16A34A", "Q3": "#9CA3AF", "Q4": "#DC2626",
    }
    QUAD_BG = {
        "Q1": "#EFF6FF", "Q2": "#F0FDF4", "Q3": "#F9FAFB", "Q4": "#FEF2F2",
    }
    QUAD_TITLES = {
        "Q1": "[핵심 성과 콘텐츠]", "Q2": "[특정 타겟 관심도 우수]",
        "Q3": "[소재 개선 필요]", "Q4": "[후킹 위주 (브랜드 연결 필요)]",
    }

    def _quad_label(row):
        high_ctr = row["ctr"]     > ctr_mean      # 기존 >= 에서 > 로 변경
        high_fol = row["follows"] > follows_mean  # 기존 복합 조건에서 단순 > 로 통일
        if   high_ctr and high_fol:      return "Q1"
        elif not high_ctr and high_fol:  return "Q2"
        elif not high_ctr and not high_fol: return "Q3"
        else:                            return "Q4"
    df["quad"] = df.apply(_quad_label, axis=1)

    quad_reps = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        df_q = df[df["quad"] == q].copy()
        quad_reps[q] = _get_quadrant_representatives(df_q, n_clusters=2)

    # 4사분면 동적 축 (X축, Y축) 계산 로직 (양쪽 3개씩, 데이터 0일 때 분기)
    def _add_tick(tick_list, val):
        if not any(abs(t - val) < 1e-7 for t in tick_list):
            tick_list.append(val)

    ctr_max = float(df["ctr"].max())
    follows_max = float(df["follows"].max())

    # --- X축 범위 및 눈금 ---
    x_step = ctr_median / 3.0 if ctr_median > 0 else (ctr_max / 3.0 if ctr_max > 0 else 0.5)
    if x_step <= 0: x_step = 0.5
    x_max = ctr_max + x_step if ctr_max > 0 else 1.0

    x_ticks = []
    for i in range(3, 0, -1):
        v = ctr_median - i * x_step
        if v >= 0: _add_tick(x_ticks, v)
    _add_tick(x_ticks, ctr_median)
    for i in range(1, 4):
        v = ctr_median + i * x_step
        if v <= x_max: _add_tick(x_ticks, v)

    # --- Y축 범위 및 눈금 ---
    if follows_max <= 0:
        # 데이터가 모두 0일 경우 중심을 0으로 맞추고 위아래 동일한 간격(-1 ~ 1) 부여
        y_min, y_max = -1.0, 1.0
        y_ticks = [-1.0, 0.0, 1.0]
    else:
        # 최대값이 0이 아닐 경우: 중앙값 기준으로 3개씩 위아래 분할
        y_min = -1.0
        y_step = follows_median / 3.0 if follows_median > 0 else follows_max / 3.0
        if y_step <= 0: y_step = 1.0
        y_max = follows_max + y_step
        
        y_ticks = [-1.0]
        for i in range(3, 0, -1):
            v = follows_median - i * y_step
            if v >= 0: _add_tick(y_ticks, v)
        _add_tick(y_ticks, follows_median)
        for i in range(1, 4):
            v = follows_median + i * y_step
            if v <= y_max: _add_tick(y_ticks, v)

    #  차트 기본 사이즈
    fig = plt.figure(figsize=(22, 10), facecolor="white")

    outer_gs = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[2.5, 3.5, 2.5],         # 좌우 패널 동일 비율로 수정
        wspace=0.06, left=0.03, right=0.97, top=0.91, bottom=0.09,
    )

    # thumbnail 행 비율 1.3/1.0의 차이를 1.2/1.2로 균등화하여
    # Q2/Q3와 Q1/Q4의 썸네일 높이를 동일하게 유지한다.
    inner_h = [0.18, 1.2, 0.18, 1.2]

    left_inner  = outer_gs[0, 0].subgridspec(4, 2, height_ratios=inner_h, hspace=0.15, wspace=0.08)
    right_inner = outer_gs[0, 2].subgridspec(4, 2, height_ratios=inner_h, hspace=0.15, wspace=0.08)

    ax_scatter = fig.add_subplot(outer_gs[0, 1])

    ax_scatter.axhspan(follows_median, y_max,  xmin=0, xmax=(ctr_median / x_max),      alpha=0.06, color=QUAD_COLORS["Q2"], zorder=0)
    ax_scatter.axhspan(follows_median, y_max,  xmin=(ctr_median / x_max), xmax=1,      alpha=0.06, color=QUAD_COLORS["Q1"], zorder=0)
    ax_scatter.axhspan(y_min, follows_median,  xmin=0, xmax=(ctr_median / x_max),      alpha=0.06, color=QUAD_COLORS["Q3"], zorder=0)
    ax_scatter.axhspan(y_min, follows_median,  xmin=(ctr_median / x_max), xmax=1,      alpha=0.06, color=QUAD_COLORS["Q4"], zorder=0)

    ax_scatter.axvline(x=ctr_median,     color="#888888", linewidth=1.0, linestyle="--", zorder=1)
    ax_scatter.axhline(y=follows_median, color="#888888", linewidth=1.0, linestyle="--", zorder=1)

    for q, color in QUAD_COLORS.items():
        df_q = df[df["quad"] == q]
        if df_q.empty: continue
        ax_scatter.scatter(
            df_q["ctr"], df_q["follows"],
            color=color, alpha=0.6, edgecolors="#aaaaaa", linewidths=0.5, s=60, zorder=2,
        )

    for q, reps in quad_reps.items():
        for rep in reps:
            ax_scatter.scatter(
                rep["ctr"], rep["follows"],
                color=QUAD_COLORS[q], marker="*", s=220, edgecolors="white", linewidths=0.8, zorder=3,
            )

    ax_scatter.set_xlim(0, x_max)
    ax_scatter.set_ylim(y_min, y_max)
    ax_scatter.set_xlabel("CTR",   fontsize=11, labelpad=6, color="#333333")
    ax_scatter.set_ylabel("팔로우", fontsize=11, labelpad=6, color="#333333")

    if x_ticks:
        ax_scatter.set_xticks(x_ticks)
        ax_scatter.set_xticklabels([f"{v:.2f}" for v in x_ticks], fontsize=8, color="#555555")
    if y_ticks:
        ax_scatter.set_yticks(y_ticks)
        ax_scatter.set_yticklabels([f"{int(v)}" if v == int(v) else f"{v:.1f}" for v in y_ticks], fontsize=8, color="#555555")

    ax_scatter.tick_params(axis="both", length=3, color="#cccccc")
    for spine in ax_scatter.spines.values():
        spine.set_edgecolor("#dddddd")
    ax_scatter.grid(False)

    panel_map = [
        (left_inner,  0, 1, "Q2"),
        (left_inner,  2, 3, "Q3"),
        (right_inner, 0, 1, "Q1"),
        (right_inner, 2, 3, "Q4"),
    ]

    for inner, t_row, th_row, q in panel_map:
        color  = QUAD_COLORS[q]
        bg     = QUAD_BG[q]
        reps   = quad_reps[q]
        title  = QUAD_TITLES[q]

        ax_title = fig.add_subplot(inner[t_row, :])
        ax_title.axis("off")
        ax_title.set_facecolor(bg)
        ax_title.text(0.5, 0.5, title, ha="center", va="center", fontsize=9, fontweight="bold", color=color, transform=ax_title.transAxes)
        ax_title.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01", facecolor=bg, edgecolor=color, linewidth=0.8, transform=ax_title.transAxes, clip_on=False))

        for col_idx in range(2):
            ax_th = fig.add_subplot(inner[th_row, col_idx])
            if col_idx < len(reps):
                rep      = reps[col_idx]
                thumb    = str(rep.get("thumbnail") or "").strip()
                img_arr  = _load_thumbnail_array(thumb) if thumb else None
                media_type = str(rep.get("ig_media_type") or "").upper()
                _draw_thumbnail_cell(ax_th, img_arr, label_text=media_type, bg_color=bg)
            else:
                _draw_thumbnail_cell(ax_th, None, label_text="", bg_color="#f5f5f5")

    fig.suptitle("", visible=False)

    out_dir = "static"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_file = os.path.join(out_dir, "quadrant_chart.png")
    fig.savefig(out_file, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    return f"./{out_file}"