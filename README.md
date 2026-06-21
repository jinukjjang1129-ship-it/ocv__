# OCV 저전압 불량 분석 & 판정 앱

OCV 측정 데이터로 배터리 셀의 OK/NG를 분석하고 판정하는 Streamlit 앱입니다.

## 두 개의 탭
1. **데이터 분석** — 엑셀 전체를 올려 OCV1~3(사전 예측)과 OCV1~4(측정 후)의
   분류 성능을 비교합니다. 산점도, 임계값별 오검/과검 표를 보여줍니다.
2. **셀 판정기** — 미리 학습해둔 모델에 셀 몇 개(OCV1~3)만 입력하면
   즉시 OK/NG와 **NG 확률**을 알려줍니다. 애매한(30~70%) 셀은 따로 표시합니다.

## 핵심 발견 (이 앱이 보여주는 것)
- OCV1~3만으로는 불량의 **약 75%**를 사전 예측할 수 있습니다(점진형·불안정형).
- 나머지 약 25%(급락형)는 OCV4에서야 드러나며, OCV1~3에 전조가 없어
  **사전 예측이 원천적으로 불가능**합니다. (OCV4 회귀 R2가 음수로 확인됨)
- 따라서 OCV1~4를 다 넣으면 성능이 높지만, 그건 '예측'이 아니라 '사후 확인'입니다.

## 사용 순서

### 1) 모델 학습 (한 번만)
    pip install -r requirements.txt
    python train_model.py "삼성SDI OCV머신러닝_샘플데이타_OCV.xlsx"

GridSearchCV로 최적 하이퍼파라미터를 자동으로 찾아 ocv_model.pkl 을 만듭니다.
(1~2분 소요. 데이터가 바뀌면 다시 실행하세요.)

### 2) 앱 실행
    streamlit run app.py

http://localhost:8501 에서 열립니다 (본인 PC에서만 보임).

## 인터넷 링크로 공유 (Streamlit Community Cloud, 무료)
1. app.py, train_model.py, ocv_model.pkl, requirements.txt, packages.txt, README.md 를
   **모두** GitHub 저장소에 올립니다.
   - 중요: 학습된 ocv_model.pkl 도 꼭 같이 올려야 판정 탭이 작동합니다.
2. https://share.streamlit.io 접속 → GitHub 로그인 → New app
3. 저장소/브랜치 선택, Main file = app.py → Deploy
4. 생성된 https://<앱이름>.streamlit.app 링크를 카톡으로 공유

## 참고
- 모델: sklearn GradientBoostingClassifier (배포 안정성 때문에 xgboost 대신 사용).
- 판정 탭의 성능 숫자(recall/precision)는 학습 시 검증셋에서 측정한 '진짜 실력'입니다.
  학습에 쓴 데이터를 다시 판정하면 더 높게 나오지만 그건 외운 것이니 참고하지 마세요.
- 한글 폰트는 packages.txt 의 fonts-nanum 으로 클라우드에 자동 설치됩니다.
