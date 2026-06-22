"""
OCV 저전압 불량 분석 & 판정 앱
  탭 1: 데이터 분석 대시보드 (엑셀 업로드 → OCV1~3 vs OCV1~4 비교)
  탭 2: 셀 판정기 (미리 학습된 모델로 셀 몇 개 즉시 판정 + NG 확률)

판정기를 쓰려면 먼저 train_model.py 를 실행해 ocv_model.pkl 을 만들어 두세요.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 한글 폰트
for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    try:
        font_manager.fontManager.addfont(fp)
        rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="OCV 불량 분석 & 판정", layout="wide")
st.title("OCV 저전압 불량 분석 & 판정")

tab1, tab2 = st.tabs(["데이터 분석", "셀 판정기"])


def build_features(d, c1, c2, c3, c4=None):
    f = pd.DataFrame(index=d.index)
    f["OCV1"] = d[c1]; f["OCV2"] = d[c2]; f["OCV3"] = d[c3]
    f["dV_12"] = d[c1] - d[c2]
    f["dV_23"] = d[c2] - d[c3]
    f["accel"] = f["dV_23"] - f["dV_12"]
    f["std_123"] = d[[c1, c2, c3]].std(axis=1)
    if c4 is not None:
        f["OCV4"] = d[c4]
        f["dV_34"] = d[c3] - d[c4]
        f["std_1234"] = d[[c1, c2, c3, c4]].std(axis=1)
    return f


# ───── 탭 1: 분석 ─────
with tab1:
    st.caption("엑셀을 올려 OCV1~3(사전 예측)과 OCV1~4(측정 후)의 분류 성능을 비교합니다.")
    up = st.file_uploader("OCV 엑셀 (.xlsx)", type=["xlsx"], key="analysis_up")
    header_row = st.number_input("헤더 행 번호 (0부터)", 0, 10, 1, key="hr")

    if up is None:
        st.info("엑셀을 업로드하면 분석이 시작됩니다.")
    else:
        df = pd.read_excel(up, header=int(header_row))
        df.columns = [str(c).strip() for c in df.columns]
        st.success(f"{len(df):,}행 x {df.shape[1]}열")
        cols = df.columns.tolist()

        def guess(opts, fb):
            for i, c in enumerate(cols):
                if str(c).upper() in opts:
                    return i
            return min(fb, len(cols) - 1)

        cA, cB, cC = st.columns(3)
        with cA:
            o1 = st.selectbox("OCV1", cols, guess({"OCV1"}, 2), key="o1")
            o2 = st.selectbox("OCV2", cols, guess({"OCV2"}, 3), key="o2")
        with cB:
            o3 = st.selectbox("OCV3", cols, guess({"OCV3"}, 4), key="o3")
            o4 = st.selectbox("OCV4", cols, guess({"OCV4"}, 5), key="o4")
        with cC:
            lc = st.selectbox("판정(OK/NG)", cols, guess({"OK_NG", "판정"}, 6), key="lc")

        is_ng = (df[lc].astype(str).str.upper().str.strip() == "NG").astype(int)
        if is_ng.sum() == 0:
            st.error("NG가 인식되지 않았어요. 판정 컬럼을 확인하세요.")
            st.stop()

        m1, m2, m3 = st.columns(3)
        m1.metric("전체", f"{len(df):,}")
        m2.metric("불량(NG)", f"{int(is_ng.sum()):,}")
        m3.metric("불량 비율", f"{is_ng.mean()*100:.2f}%")

        scenario = st.radio("입력 변수 시나리오",
                            ["OCV1~3 (사전 예측)", "OCV1~4 (측정 후)"], key="sc")
        use4 = scenario.startswith("OCV1~4")

        if st.button("분석 실행", type="primary", key="run_an"):
            X = build_features(df, o1, o2, o3, o4 if use4 else None)
            Xtr, Xte, ytr, yte = train_test_split(
                X, is_ng, test_size=0.2, random_state=42, stratify=is_ng)
            n0, n1 = (ytr == 0).sum(), (ytr == 1).sum()
            model = GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
            model.fit(Xtr, ytr, sample_weight=np.where(ytr == 1, n0 / n1, 1.0))
            pred = model.predict(Xte)
            proba = model.predict_proba(Xte)[:, 1]

            tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
            rec = tp / (tp + fn) if (tp + fn) else 0
            prec = tp / (tp + fp) if (tp + fp) else 0

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("오검(FN)", fn)
            r2.metric("과검(FP)", fp)
            r3.metric("recall", f"{rec:.3f}")
            r4.metric("precision", f"{prec:.3f}")

            # ── 비전공자용 상세 설명 ──
            n_test = tn + fp + fn + tp
            with st.expander("📖 이 숫자들이 어떻게 나왔나요? (클릭해서 펼치기)"):
                st.markdown(f"""
**먼저, 테스트 방법부터.**
전체 데이터 중 **{int(n_test):,}개 셀**을 '시험 문제'로 따로 빼두었습니다.
모델에게 이 셀들의 정답(OK/NG)을 가린 채 맞춰보게 한 뒤,
실제 정답과 비교한 결과가 아래입니다.

모델의 판정은 4가지 경우로 나뉩니다:

| | 실제 OK인데 | 실제 NG인데 |
|---|---|---|
| **모델이 OK라 함** | ✅ 맞음 (정상 통과) = **{int(tn):,}개** | ❌ **오검** (불량을 놓침) = **{int(fn)}개** |
| **모델이 NG라 함** | ❌ **과검** (양품을 버림) = **{int(fp)}개** | ✅ 맞음 (불량 검출) = **{int(tp)}개** |

이 4개 숫자로 아래 지표들을 계산합니다.
""")

                st.markdown(f"""
---
**① 오검 (불량을 놓침) = {int(fn)}개**
실제로는 불량(NG)인데 모델이 "괜찮다(OK)"고 통과시킨 개수입니다.
→ 불량이 그대로 출하되니 **가장 위험**합니다.

**② 과검 (양품을 버림) = {int(fp)}개**
실제로는 멀쩡한(OK) 셀인데 모델이 "불량(NG)"이라고 잘못 버린 개수입니다.
→ 쓸 수 있는 셀을 폐기하니 **수율(생산성) 손실**입니다.
""")

                rec_pct = rec * 100
                st.markdown(f"""
---
**③ recall (불량 검출률) = {rec:.3f}**

*"실제 불량 중에서, 모델이 몇 %나 잡아냈는가?"*

- 실제 불량 총 개수 = 잡은 것 + 놓친 것 = {int(tp)} + {int(fn)} = **{int(tp+fn)}개**
- 그중 모델이 잡은 것 = **{int(tp)}개**

$$ recall = \\frac{{잡은\\ 불량}}{{전체\\ 불량}} = \\frac{{{int(tp)}}}{{{int(tp)}+{int(fn)}}} = \\frac{{{int(tp)}}}{{{int(tp+fn)}}} = {rec:.3f} $$

즉 **실제 불량의 {rec_pct:.1f}%를 잡아냈다**는 뜻입니다.
recall은 **오검(놓침)과 직접 관련**됩니다 — 놓치는 게 적을수록 recall이 높아집니다.
""")

                prec_pct = prec * 100
                st.markdown(f"""
---
**④ precision (정밀도) = {prec:.3f}**

*"모델이 '불량'이라고 한 것 중에서, 진짜 불량은 몇 %였는가?"*

- 모델이 불량이라 한 총 개수 = 맞은 것 + 틀린 것 = {int(tp)} + {int(fp)} = **{int(tp+fp)}개**
- 그중 진짜 불량 = **{int(tp)}개**

$$ precision = \\frac{{진짜\\ 불량}}{{불량이라\\ 한\\ 전체}} = \\frac{{{int(tp)}}}{{{int(tp)}+{int(fp)}}} = \\frac{{{int(tp)}}}{{{int(tp+fp)}}} = {prec:.3f} $$

즉 **불량이라 판정한 것의 {prec_pct:.1f}%가 진짜 불량**이었다는 뜻입니다.
precision은 **과검(헛버림)과 직접 관련**됩니다 — 헛다리가 적을수록 precision이 높아집니다.
""")

                st.info("**한 줄 요약** — recall은 '불량을 안 놓치는 능력'(오검 관련), "
                        "precision은 '불량이라 할 때 정확한 정도'(과검 관련)입니다. "
                        "보통 둘은 한쪽을 올리면 다른 쪽이 내려가는 관계라, 공정에서 "
                        "무엇이 더 중요한지에 따라 균형점을 정합니다.")

            if use4:
                st.success("OCV4까지 본 결과입니다. 검출률은 높지만 OCV4는 마지막 측정값이라 "
                           "'사전 예측'이 아닌 '사후 확인'에 가깝습니다.")
            else:
                st.warning("OCV1~3 사전 예측 결과입니다. 급락형 불량(OCV4 급락)은 이 단계에선 "
                           "단서가 없어 놓칠 수 있습니다. 이게 진짜 '예측' 성능입니다.")

            st.subheader("OCV3 vs OCV4 - OK/NG 분포")
            fig, ax = plt.subplots(figsize=(6, 5))
            okm = is_ng == 0
            ax.scatter(df.loc[okm, o3], df.loc[okm, o4], s=6, alpha=0.2, c="#3b82f6", label="OK")
            ax.scatter(df.loc[~okm, o3], df.loc[~okm, o4], s=22, alpha=0.8, c="#ef4444", label="NG")
            ax.set_xlabel("OCV3"); ax.set_ylabel("OCV4"); ax.legend()
            fig.tight_layout(); st.pyplot(fig)

            st.subheader("임계값에 따른 오검 - 과검")
            rows = []
            for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
                pn = proba >= t
                a, b, c, d2 = confusion_matrix(yte, pn).ravel()
                rows.append({"임계값": t, "오검(FN)": int(c), "과검(FP)": int(b),
                             "recall": round(d2/(d2+c), 3) if (d2+c) else 0})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("**임계값**은 'NG확률이 몇 % 이상이면 불량으로 볼지'의 기준선입니다. "
                       "기준을 낮추면(예: 0.5→0.2) 불량을 더 적극적으로 잡아 오검(놓침)이 줄지만, "
                       "멀쩡한 셀도 더 버려서 과검이 늡니다. 공정에서 '불량 출하'와 '양품 폐기' 중 "
                       "무엇이 더 비싼지에 따라 기준을 정하세요.")


# ───── 탭 2: 판정기 ─────
with tab2:
    st.caption("OCV1, 2, 3 값을 입력하면 학습된 모델이 OK/NG와 NG 확률을 알려줍니다.")

    if not os.path.exists("ocv_model.pkl"):
        st.error("학습된 모델(ocv_model.pkl)이 없습니다. "
                 "먼저 터미널에서 `python train_model.py 파일명.xlsx` 를 실행하세요.")
    else:
        import joblib
        bundle = joblib.load("ocv_model.pkl")
        clf, reg, feats = bundle["clf"], bundle["reg"], bundle["feats"]
        mt = bundle.get("metrics", {})

        if mt:
            st.info(
                f"**검증된 성능** - 검출률(recall) {mt.get('test_recall','?')}, "
                f"정밀도(precision) {mt.get('test_precision','?')} "
                f"(오검 {mt.get('test_fn','?')}, 과검 {mt.get('test_fp','?')}). "
                f"OCV1~3 기반이라 급락형 불량은 놓칠 수 있습니다. "
                f"OCV4 회귀 R2={mt.get('reg_r2','?')} (음수면 OCV4 예측 신뢰 불가, 참고용)."
            )

            with st.expander("📖 'NG확률'과 성능 숫자가 무슨 뜻인가요? (클릭)"):
                st.markdown(f"""
**NG확률이란?**
모델이 각 셀에 매기는 "이 셀이 불량(NG)일 가능성"입니다.
- **90% 이상** → 거의 확실히 불량
- **10% 이하** → 거의 확실히 정상
- **30~70%** → 애매함 (이 앱은 이런 셀에 '추가 검사 권장'을 표시합니다)

판정은 보통 50%를 기준으로 나눕니다 — 50% 넘으면 NG, 아니면 OK.

---
**위 '검증된 성능' 숫자는 어떻게 나왔나요?**

이 모델을 만들 때, 정답(OK/NG)을 아는 과거 데이터 일부를 '시험 문제'로 빼서
모델에게 맞춰보게 한 결과입니다.

- **검출률(recall) {mt.get('test_recall','?')}** =
  실제 불량 중 모델이 잡아낸 비율.
  예: recall 0.75 → 실제 불량 100개 중 75개를 잡고 25개는 놓침(오검).

- **정밀도(precision) {mt.get('test_precision','?')}** =
  모델이 "불량"이라 한 것 중 진짜 불량인 비율.
  예: precision 0.80 → 불량이라 한 것 100개 중 80개는 진짜, 20개는 헛다리(과검).

쉽게: **recall = 불량을 안 놓치는 능력**, **precision = 불량이라 할 때 맞는 정도**.
""")

        def make_feats(d):
            f = pd.DataFrame(index=d.index)
            f["OCV1"] = d["OCV1"]; f["OCV2"] = d["OCV2"]; f["OCV3"] = d["OCV3"]
            f["dV_12"] = d["OCV1"] - d["OCV2"]
            f["dV_23"] = d["OCV2"] - d["OCV3"]
            f["accel"] = f["dV_23"] - f["dV_12"]
            f["std_123"] = d[["OCV1", "OCV2", "OCV3"]].std(axis=1)
            return f[feats]

        mode = st.radio("입력 방식", ["직접 입력", "엑셀 업로드"], key="pmode")
        d = None
        if mode == "직접 입력":
            n = st.number_input("셀 개수", 1, 20, 3, key="pn")
            rows = []
            for i in range(int(n)):
                c1, c2, c3 = st.columns(3)
                v1 = c1.number_input(f"#{i+1} OCV1", value=35.50, step=0.01, format="%.2f", key=f"p1_{i}")
                v2 = c2.number_input(f"#{i+1} OCV2", value=35.50, step=0.01, format="%.2f", key=f"p2_{i}")
                v3 = c3.number_input(f"#{i+1} OCV3", value=35.50, step=0.01, format="%.2f", key=f"p3_{i}")
                rows.append({"OCV1": v1, "OCV2": v2, "OCV3": v3})
            d = pd.DataFrame(rows)
        else:
            pf = st.file_uploader("OCV1~3이 있는 엑셀", type=["xlsx"], key="pup")
            phr = st.number_input("헤더 행 번호 (0부터)", 0, 10, 0, key="phr")
            if pf is not None:
                d = pd.read_excel(pf, header=int(phr))
                d.columns = [str(c).strip() for c in d.columns]
                if not {"OCV1", "OCV2", "OCV3"}.issubset(set(d.columns)):
                    st.warning("엑셀에 OCV1, OCV2, OCV3 컬럼이 필요해요. 현재: " + ", ".join(map(str, d.columns)))
                    d = None

        if d is not None and st.button("판정 실행", type="primary", key="run_pred"):
            Xn = make_feats(d)
            proba = clf.predict_proba(Xn)[:, 1]
            pred = clf.predict(Xn)
            pred_o4 = reg.predict(Xn)

            out = d[["OCV1", "OCV2", "OCV3"]].copy()
            out["판정"] = np.where(pred == 1, "NG", "OK")
            out["NG확률"] = (proba * 100).round(1).astype(str) + "%"
            out["예측OCV4(참고)"] = pred_o4.round(2)

            def color(v):
                return "background-color: #fee2e2" if v == "NG" else "background-color: #dcfce7"
            st.dataframe(out.style.map(color, subset=["판정"]),
                         use_container_width=True, hide_index=True)

            amb = [(i + 1, p) for i, p in enumerate(proba) if 0.3 < p < 0.7]
            if amb:
                txt = ", ".join([f"#{i}({p*100:.0f}%)" for i, p in amb])
                st.warning(f"**추가 검사 권장** - NG확률이 애매한(30~70%) 셀: {txt}")
            else:
                st.success("애매한(경계) 셀은 없습니다.")

            st.caption("예측OCV4는 OCV1~3 기반 추정치로 신뢰도가 낮습니다(R2 음수). 참고용으로만 보세요.")
