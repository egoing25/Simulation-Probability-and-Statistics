"""
청소년건강행태조사 원시자료(.sav)에서 필요한 열만 뽑는 스크립트

[준비]
  pip install pyreadstat pandas

[사용법]
  1단계) 변수 목록 확인 (.sav 파일과 같은 폴더에서 실행)
      python kyrbs_extract.py 파일이름.sav
      -> variables.csv(전체 변수명과 설명)가 만들어지고,
         성별/학년/스트레스/가중치 후보가 화면에 출력됩니다.

  2단계) 필요한 열만 뽑아서 작은 CSV 만들기
      python kyrbs_extract.py 파일이름.sav 성별변수 학년변수 스트레스변수
      예) python kyrbs_extract.py raw.sav SEX GRADE E_S_RCRD
      -> slim.csv(작은 파일)와 응답 분포가 출력됩니다.
"""
import sys
import pandas as pd

try:
    import pyreadstat
except ImportError:
    sys.exit("pyreadstat이 없습니다. 터미널에서: pip install pyreadstat")

if len(sys.argv) < 2:
    sys.exit("사용법: python kyrbs_extract.py 파일이름.sav [성별변수 학년변수 스트레스변수]")

path = sys.argv[1]
# 값 라벨을 적용하지 않고 숫자 코드 그대로 읽는다
df, meta = pyreadstat.read_sav(path, apply_value_formats=False)

names = list(meta.column_names)
labels = list(meta.column_labels)

# ---------- 1단계: 변수 목록 ----------
if len(sys.argv) == 2:
    pd.DataFrame({"변수명": names, "설명": labels}).to_csv(
        "variables.csv", index=False, encoding="utf-8-sig")
    print(f"응답자 수(행): {len(df)}, 변수 수(열): {len(names)}")
    print("전체 변수 목록을 variables.csv로 저장했습니다.\n")

    keywords = ["성별", "학년", "스트레스", "가중", "스마트폰", "수면", "학교급"]
    for kw in keywords:
        print(f"[{kw}] 후보")
        found = False
        for n, l in zip(names, labels):
            if kw in str(l) or kw in str(n):
                print(f"  {n}: {l}")
                found = True
        if not found:
            print("  (없음)")
    sys.exit()

# ---------- 2단계: 필요한 열만 저장 ----------
sex, grade, stress = sys.argv[2], sys.argv[3], sys.argv[4]
for c in (sex, grade, stress):
    if c not in df.columns:
        sys.exit(f"변수 '{c}'가 없습니다. variables.csv에서 이름을 다시 확인하세요.")

slim = df[[sex, grade, stress]].copy()
slim.columns = ["sex", "grade", "stress"]
slim.to_csv("slim.csv", index=False, encoding="utf-8-sig")

print(f"slim.csv 저장 완료 ({len(slim)}행)\n")
for col in ["sex", "grade", "stress"]:
    print(f"[{col}] 값별 응답 수")
    print(slim[col].value_counts(dropna=False).sort_index().to_string())
    print()

# 설문 보기(값 라벨)도 함께 출력해서 코드의 의미를 확인
for orig, new in ((sex, "sex"), (grade, "grade"), (stress, "stress")):
    vl = meta.variable_value_labels.get(orig)
    if vl:
        print(f"[{new}] 코드 의미: {vl}")