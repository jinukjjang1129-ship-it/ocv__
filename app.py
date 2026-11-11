"""Local OCV analysis and early cell scoring using the same tested pipeline."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from battery_ocv.experiment import experiment
from battery_ocv.features import build_features

st.set_page_config(page_title="OCV 사전 불량 분류", layout="wide")
st.title("OCV1~3 기반 사전 불량 분류")
st.caption("생산일을 분리해 평가하고, OCV4 측정 전 사용할 수 있는 입력만 학습합니다.")
analysis, scoring = st.tabs(["데이터 분석", "셀 판정"])
with analysis:
    upload = st.file_uploader("OCV Excel", type=["xlsx"])
    header = st.number_input("헤더 행 (0부터)", min_value=0, max_value=10, value=1)
    split = st.selectbox(
        "평가 방식",
        ["temporal", "random"],
        format_func=lambda x: (
            "생산일 순서 분리" if x == "temporal" else "무작위 분할 민감도 검사"
        ),
    )
    identity = (
        (hashlib.sha256(upload.getvalue()).hexdigest(), header, split)
        if upload
        else None
    )
    if upload and st.button("분석 실행", type="primary"):
        try:
            frame = pd.read_excel(upload, header=int(header))
            frame.columns = frame.columns.astype(str).str.strip()
            with st.spinner("학습·검증·평가를 분리해 4개 모델을 비교합니다."):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    result = experiment(frame, output, strategy=split)
                    result["images"] = {
                        name: (output / name).read_bytes()
                        for name in ["precision_recall.png", "confusion_matrix.png"]
                    }
            st.session_state["experiment"] = result
            st.session_state["identity"] = identity
        except (ValueError, KeyError, OSError) as exc:
            st.error(f"분석을 진행할 수 없습니다: {exc}")
    current = (
        st.session_state.get("experiment")
        if st.session_state.get("identity") == identity
        else None
    )
    if current:
        report = current["report"]
        st.write(f"검증셋 AP로 선택한 모델: {report['selected_model']}")
        st.dataframe(pd.DataFrame(report["partitions"]).T)
        rows = [
            {"model": name, **values["test_selected_threshold"]}
            for name, values in report["models"].items()
        ]
        st.dataframe(
            pd.DataFrame(rows)[
                [
                    "model",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "average_precision",
                    "fn",
                    "fp",
                ]
            ],
            hide_index=True,
        )
        st.image(current["images"]["precision_recall.png"])
        st.image(current["images"]["confusion_matrix.png"])
        st.caption(
            "임곗값은 검증셋에서 선택했습니다. 표의 평가셋 결과로 모델을 다시 선택하면 별도 미래 평가가 필요합니다."
        )
    else:
        st.info("생산일자, OCV1~3, OK_NG 컬럼이 있는 파일을 올린 뒤 분석을 실행하세요.")
with scoring:
    if (
        not st.session_state.get("experiment")
        or st.session_state.get("identity") != identity
    ):
        st.info("현재 파일의 데이터 분석을 먼저 실행하세요.")
    else:
        current = st.session_state["experiment"]
        values = [
            st.number_input(column, value=35.5, format="%.4f")
            for column in ["OCV1", "OCV2", "OCV3"]
        ]
        if st.button("셀 판정"):
            features = build_features(
                pd.DataFrame([values], columns=["OCV1", "OCV2", "OCV3"])
            )
            score = float(current["model"].predict_proba(features)[0, 1])
            st.write("판정: " + ("NG" if score >= current["threshold"] else "OK"))
            st.metric("모델 NG 점수", f"{score:.3f}")
            st.caption(
                "이 점수는 보정된 불량 확률이 아닙니다. 실제 공정 적용에는 독립 생산일·로트 검증이 필요합니다."
            )
