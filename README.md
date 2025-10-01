# 🍽 식음업장 메뉴 수요 예측 AI 해커톤 (LG Aimers 7기)

## 📝 프로젝트 개요
리조트 내 식음업장은 **계절, 요일, 투숙객 수, 행사 일정** 등 다양한 요인에 따라 수요가 크게 변동하는 환경에 놓여 있습니다.  
특히 휴양지 리조트는 단기간에 집중되는 고객 수요와 예측하기 어려운 방문 패턴으로 인해,  
**메뉴별 식자재 준비, 인력 배치, 재고 관리에 있어 높은 운영 난이도**를 보입니다.

이러한 복잡한 운영 환경 속에서 **정확한 메뉴 수요 예측**은 비용 절감과 고객 만족도 향상에 핵심적인 역할을 합니다.  
최근에는 AI 기술을 활용한 수요 예측이 식음 서비스 운영의 새로운 해법으로 주목받고 있으며,  
**정형화된 과거 매출 데이터와 외부 요인을 함께 분석**하는 방식이 빠르게 확산되고 있습니다.

이번 해커톤은 실제 식음업장에서 수집된 판매 데이터를 활용해  
**메뉴별 1주일 수요를 예측하는 AI 모델을 개발**하는 것을 목표로 합니다.  
이를 통해 데이터 기반 의사결정이 리조트 운영에 어떤 가치를 더할 수 있는지를 직접 경험할 수 있습니다.

---

## 📌 주제
리조트 내 식음업장 메뉴별 **1주일 수요 예측 AI 모델 개발**

---

## 🧮 평가 방식

- 평가지표: 식음업장별 가중치가 있는 **SMAPE**
- '담하'와 '미라시아' 업장은 다른 업장보다 높은 가중치로 반영
- 업장별 가중치 값은 비공개이며, 실제 매출 수량이 0인 경우에는 평가 산식에서 제외됨

<p align="center">
  <img width="433" height="60" alt="evaluation_formula_1" src="https://github.com/user-attachments/assets/faf4c249-731e-4110-b71e-468b177c76b9" />
</p>

<p align="center">
  <img width="358" height="173" alt="evaluation_formula_2" src="https://github.com/user-attachments/assets/1112fbdc-a42b-4290-a22c-dcdb76e318fc" />
</p>

- **Public Score**: 전체 테스트 데이터 샘플 중 사전 샘플링된 50%  
- **Private Score**: 전체 테스트 데이터 샘플 100%

---

## 📂 데이터 구성

### 📁 train/
- 기간: 2023.01.01 ~ 2024.06.15  
- 영업장명_메뉴명별 매출 수량 정보 포함  
- 주요 컬럼
  - `영업일자`
  - `영업장명_메뉴명`
  - `매출수량`

---

### 📁 test/
- 파일: `TEST_00.csv` ~ `TEST_09.csv`  
- 기간: 2025년 특정 시점(28일)  
- 영업장명_메뉴명별 매출 수량 정보 포함  
- 주요 컬럼
  - `영업일자`
  - `영업장명_메뉴명`
  - `매출수량`

---

### 📄 sample_submission.csv
- 제출 양식 파일  
- 각 영업장명_메뉴명의 TEST 파일별 **+1일~+7일 매출 수량 예측 결과** 포함  
- 예시: TEST_00+1일, TEST_00+2일, ..., TEST_00+7일

---

## 🏁 점수 산정 방식 요약

| 항목 | 설명 |
|------|------|
| Public Score | 전체 테스트 데이터 샘플 중 50%로 평가 |
| Private Score | 전체 테스트 데이터 100%로 최종 평가 |

---

## 🧠 모델 개요

본 프로젝트에서는 시계열 예측을 위해 **Temporal Convolutional Network (TCN)** 기반 모델을 적용했습니다.  
TCN은 dilated causal convolution을 활용해 긴 시계열 패턴을 효율적으로 학습할 수 있으며,  
RNN 계열 모델보다 병렬 연산이 가능하고 안정적인 학습이 가능합니다.

### Global TCN 구조
- 메뉴(Item) 임베딩과 수치형·피처 입력을 결합해 학습
- dilated convolution을 반복 적용한 multi-level TCN 구조
- 마지막 시점의 hidden state를 기반으로 향후 7일 수요를 예측

---

## ⚙️ 학습 방식

### 1. H=1(Recursive) 학습 + 7-step Unroll 보조 손실
- 1-step 예측 손실에 더해, **7일 동안 예측을 반복(unroll)** 하며 보조 손실을 부여하여 장기 예측 안정성을 확보
- Scheduled Sampling을 적용해 학습 시 실제값과 예측값을 혼합, 학습-추론 간 분포 차이를 완화

---

### 2. MIMO + Recursive 앙상블 (α-calibration)
- **MIMO 모델**: 7일치를 한 번에 예측
- **Recursive 모델**: 하루씩 순차 예측
- 백테스트 데이터로 shop-weighted SMAPE를 최소화하는 **α 값을 보정**하여  
  과적합을 방지하고 일반화 성능을 향상

---

### 3. 학습 안정화 기법
- Optimizer: AdamW  
- Scheduler: Cosine Annealing  
- Early Stopping 적용 (Patience=10)  
- Gradient Clipping (Norm=1.0)  
- Mixed Precision 학습(torch.cuda.amp)으로 학습 안정성 및 속도 개선

---

## 🧪 모델 요약

| 항목 | 내용 |
|------|------|
| 모델 | Temporal Convolutional Network (TCN) |
| 예측 방식 | MIMO + Recursive 앙상블 |
| 보조 손실 | 7-step Unroll + Scheduled Sampling |
| 튜닝 방식 | α-calibration (shop-weighted SMAPE 최소화) |
| 학습 안정화 | Early stopping, Grad clipping, AMP |

---

## 📌 참고
- LG Aimers 공식 대회 페이지: [https://www.lgaimers.ai](https://www.lgaimers.ai)  
- 곤지암 리조트: [https://www.konjiamresort.co.kr](https://www.konjiamresort.co.kr)

