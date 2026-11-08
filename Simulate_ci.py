"""
표본추출 시뮬레이션: 단순임의추출 vs 성별 층화추출 (비례배분)
모집단 = 청소년건강행태조사(2025) 원시자료 응답자 전체 (slim.csv)
관심 모비율 p = 스트레스를 '대단히 많이 + 많이' 느끼는 학생의 비율

[사용법]  slim.csv와 같은 폴더에서
    python simulate_ci.py

[만들어지는 파일]
    sim_results.csv   : 표본 크기·추출 방법별 결과표
    sim_example.txt   : 표본 1회 추출 예시(보고서 '과정 재현'용)
    sim_hist.png      : n별 표본비율 분포 그림 (matplotlib 있을 때)
"""
import numpy as np
import pandas as pd

SEED = 2025          # 고정 시드 -> 누구나 같은 결과를 재현할 수 있음
REPEAT = 10000       # 표본을 뽑는 반복 횟수
SIZES = [50, 100, 400, 1000]
Z = 1.96             # 신뢰도 95%

df = pd.read_csv("slim.csv")
df = df.dropna(subset=["sex", "grade", "stress"]).reset_index(drop=True)

# 스트레스 인지 = 1(대단히 많이) 또는 2(많이) 응답 -> 1, 나머지 -> 0
y = df["stress"].isin([1, 2]).to_numpy().astype(int)
sex = df["sex"].to_numpy()

N = len(y)
p = y.mean()
print(f"모집단 크기 N = {N}")
print(f"모비율 p = {p:.4f}")

# ---------- 편향 근거: 성별·학교급별 모비율 ----------
print("\n[성별 모비율]")
for s, label in ((1, "남"), (2, "여")):
    m = sex == s
    print(f"  {label}학생: 인원 {m.sum()}명 ({m.mean():.3f}), 모비율 {y[m].mean():.4f}")
df["group"] = np.where(df["grade"] <= 3, "중학생", "고등학생")
print("[학교급별 모비율]")
for g in ["중학생", "고등학생"]:
    m = (df["group"] == g).to_numpy()
    print(f"  {g}: 인원 {m.sum()}명, 모비율 {y[m].mean():.4f}")

rng = np.random.default_rng(SEED)
idx_m = np.where(sex == 1)[0]
idx_f = np.where(sex == 2)[0]
share_m = len(idx_m) / N


def srs(n):
    """단순임의추출 (비복원)"""
    return rng.choice(N, size=n, replace=False)


def stratified(n):
    """성별 층화추출: 모집단 성별 비율대로 배분"""
    n_m = int(round(n * share_m))
    n_f = n - n_m
    a = rng.choice(idx_m, size=n_m, replace=False)
    b = rng.choice(idx_f, size=n_f, replace=False)
    return np.concatenate([a, b])


def one_ci(sample_idx):
    n = len(sample_idx)
    ph = y[sample_idx].mean()
    se = np.sqrt(ph * (1 - ph) / n)
    return ph, se, ph - Z * se, ph + Z * se


rows = []
hist_data = {}
for n in SIZES:
    for name, fn in (("단순임의추출", srs), ("성별 층화추출", stratified)):
        phs = np.empty(REPEAT)
        lows = np.empty(REPEAT)
        highs = np.empty(REPEAT)
        for i in range(REPEAT):
            ph, se, lo, hi = one_ci(fn(n))
            phs[i], lows[i], highs[i] = ph, lo, hi
        coverage = np.mean((lows <= p) & (p <= highs))
        rows.append({
            "표본크기 n": n,
            "추출방법": name,
            "표본비율 평균": round(phs.mean(), 4),
            "표본비율 표준편차(실제 흔들림)": round(phs.std(ddof=1), 4),
            "이론 표준오차 sqrt(p(1-p)/n)": round(np.sqrt(p * (1 - p) / n), 4),
            "평균 신뢰구간 폭": round((highs - lows).mean(), 4),
            "p 포함 비율(%)": round(coverage * 100, 2),
        })
        hist_data[(n, name)] = phs

res = pd.DataFrame(rows)
res.to_csv("sim_results.csv", index=False, encoding="utf-8-sig")
print("\n[시뮬레이션 결과]")
print(res.to_string(index=False))

# ---------- 표본 1회 추출 예시 (보고서에 계산 과정 쓰기용) ----------
rng_ex = np.random.default_rng(1)
ex_n = 400
ex_idx = rng_ex.choice(N, size=ex_n, replace=False)
ph, se, lo, hi = one_ci(ex_idx)
x = int(y[ex_idx].sum())
text = (
    f"[표본 1회 예시] 단순임의추출, n={ex_n}\n"
    f"X = {x}, p_hat = X/n = {ph:.4f}\n"
    f"SE = sqrt(p_hat(1-p_hat)/n) = {se:.4f}\n"
    f"95% 신뢰구간 = [{lo:.4f}, {hi:.4f}]  (폭 {hi - lo:.4f})\n"
    f"모비율 p = {p:.4f} 이(가) 구간에 포함되는가? {lo <= p <= hi}\n"
    f"정규근사 조건: n*p_hat = {ex_n * ph:.1f}, n*(1-p_hat) = {ex_n * (1 - ph):.1f}\n"
)
with open("sim_example.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("\n" + text)

# ---------- 그림 (선택) ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, len(SIZES), figsize=(14, 3.2), sharex=True)
    for ax, n in zip(axes, SIZES):
        ax.hist(hist_data[(n, "단순임의추출")], bins=40, alpha=0.6, label="단순임의")
        ax.hist(hist_data[(n, "성별 층화추출")], bins=40, alpha=0.6, label="층화")
        ax.axvline(p, color="k", linestyle="--", linewidth=1)
        ax.set_title(f"n = {n}")
        ax.set_xlabel("표본비율")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig("sim_hist.png", dpi=150)
    print("sim_hist.png 저장 완료")
except Exception as e:
    print("그림 저장은 건너뜀:", e)