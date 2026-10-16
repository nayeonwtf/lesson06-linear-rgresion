import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 마지막 해
MIN_DAYS = 300        # 한 해에 필요한 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")
st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온의 변화를 직선(선형 회귀)으로 나타내고, 원하는 해의 기온을 예상해 봅니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_annual() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    # 날짜 열: 공백/탭 제거 후 변환
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜나 평균기온이 비어 있는 행 제거, 2025년까지만 사용
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]

    # 연도별 평균기온과 관측일 수
    annual = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    # 관측일이 300일 미만인 해 제외
    annual = annual[annual["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return annual


try:
    annual = load_annual()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(annual) < 2:
    st.error("회귀 직선을 만들 데이터가 부족합니다.")
    st.stop()

x = annual["연도"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)

# 회귀 직선과 상관계수
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(annual)
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())

# 최근 20년(끝 연도 기준 20년 구간)만으로 구한 직선
RECENT_SPAN = 20
recent = annual[annual["연도"] > end_year - RECENT_SPAN]
recent_ok = len(recent) >= 3
if recent_ok:
    r_slope, r_intercept = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1,
    )
    recent_start = int(recent["연도"].min())
    recent_end = int(recent["연도"].max())

# ---- 예상 기온 ----
st.subheader("연도별 예상 기온")
year = st.slider("연도를 고르세요", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"<div style='text-align:center; padding:1rem 0;'>"
    f"<div style='font-size:1.2rem; color:gray;'>{year}년 예상 평균기온</div>"
    f"<div style='font-size:4rem; font-weight:700;'>{pred:.2f} °C</div>"
    f"</div>",
    unsafe_allow_html=True,
)
if year < start_year or year > end_year:
    st.caption("※ 직선을 만든 기간 밖의 해라서, 직선을 그대로 늘려서 얻은 값입니다.")

# ---- 분석 요약 ----
st.subheader("분석 요약")
c1, c2, c3 = st.columns(3)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")

st.metric("상관계수 (연도 ↔ 연평균기온)", f"{r:.3f}")

# ---- 기울기: 100년당 변화 ----
st.subheader("100년에 몇 도 오르는가")
st.markdown(
    f"<div style='text-align:center; padding:0.5rem 0 1rem 0;'>"
    f"<div style='font-size:1.2rem; color:gray;'>전체 기간 직선의 기울기 ({start_year}~{end_year}년)</div>"
    f"<div style='font-size:4rem; font-weight:700;'>{slope * 100:+.2f} °C / 100년</div>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("**전체 기간 vs 최근 20년**")
col_all, col_recent = st.columns(2)
with col_all:
    st.markdown(
        f"<div style='text-align:center;'>"
        f"<div style='color:gray;'>전체 기간 ({start_year}~{end_year}년, {n_years}개 해)</div>"
        f"<div style='font-size:2.2rem; font-weight:700;'>{slope * 100:+.2f} °C / 100년</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with col_recent:
    if recent_ok:
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='color:gray;'>최근 20년 ({recent_start}~{recent_end}년, {len(recent)}개 해)</div>"
            f"<div style='font-size:2.2rem; font-weight:700;'>{r_slope * 100:+.2f} °C / 100년</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("최근 20년 구간에 남은 해가 3개 미만이라 기울기를 구할 수 없습니다.")

if recent_ok:
    diff = (r_slope - slope) * 100
    st.caption(
        f"최근 20년의 기울기가 전체 기간보다 {abs(diff):.2f} °C / 100년 "
        f"{'큽니다' if diff > 0 else '작습니다'}. "
        "최근 20년은 해의 수가 적어 값이 해마다의 변동에 크게 흔들리고, "
        "'100년당' 값도 짧은 기간의 추세를 100년으로 늘려 말한 것입니다."
    )

st.caption(
    f"{LAST_YEAR}년까지의 자료만 사용했고, 관측일이 {MIN_DAYS}일 미만인 해는 제외했습니다."
)

# ---- 그래프 ----
st.subheader("산점도와 회귀 직선")
line_x = np.array([min(start_year, year), max(end_year, year)], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8, color="#1f77b4"),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(color="#d62728", width=3),
        hoverinfo="skip",
    )
)
if recent_ok:
    rx = np.array([recent_start, recent_end], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=rx,
            y=r_slope * rx + r_intercept,
            mode="lines",
            name="최근 20년 직선",
            line=dict(color="#ff7f0e", width=3, dash="dash"),
            hoverinfo="skip",
        )
    )
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=14, color="gold", symbol="star", line=dict(width=1, color="black")),
        hovertemplate="%{x}년<br>예상 %{y:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    height=450,
)
st.plotly_chart(fig, use_container_width=True)
