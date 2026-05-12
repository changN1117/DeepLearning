# IMBK_DeepLearning_Competition
금융 데이터 딥러닝 학습 모델

## 1. 프로젝트명 
고객 등급 분류 딥러닝 모델 학습

## 2. 기간 
2026/05/12

## 3. 기술스택 
  - 데이터 전처리: Pandas, NumPy, Torch

  - 모델링/최적화: TabTransFormer

  - 시각화/해석: Matplotlib

## 4. 데이터 출처 : 캐글 Bank Customer Churn Dataset (row: 10000, col:12)raw데이터-(Credit score classification)

## 5. 데이터 전처리

   - 불필요한 피처 제거 : 'ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan'
   - 범주형 데이터 인코딩 : 'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'
   - 데이터 분할 : X = data2[cat_cols + num_cols], y = data2[target_col]
                   train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
   - 피처 스케일링 : StandardScaler
   - 로그 변환 : log_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt', 'Total_EMI_per_month',
                            'Amount_invested_monthly', 'Monthly_Balance']
                 for col in log_cols:
                  data2[col] = np.log1p(data2[col])
  - 결측치 제거 : data2 = data2.dropna(subset=['Credit_Score'])
                  data2 = data2.dropna(subset=['Monthly_Balance'])

## 6. EDA 
  - 정답값으로 쓰려는 Credit_Score에 결측치가 하나 존재했는데 Monthly_Balance역시 결측치가 존재해서 
결측치는 제거하고 보았습니다.
 - 값들중에서 로그변환이 필요해보이는 값을이 존재했는데 분류학습을 더 잘 학습하기위해
로그변환을 씌우고 진행했습니다. <img width="836" height="626" alt="딥러닝 로그변환 이유" src="https://github.com/user-attachments/assets/ed123875-e893-4e7c-a11c-11f3bc5e7cd5" />

## 7. FeatureSelection 및 모델 선정기준
  - 정답값과 크게 상관을 보이는 컬럼들이 없어서 모든 컬럼을 X값으로 넣었습니다.
  <img width="585" height="616" alt="상관계수" src="https://github.com/user-attachments/assets/eea6420e-be3f-422b-801a-ec333c7ae956" />
  - 파생변수 추가
  - 연속형과 범주형 컬럼들이 공존해서 tabtransformer사용
  - 파생변수 세 개를 추가 했습니다
  - 'Debt_to_Income' : 소득 대비 빚, 소득에 비해 빚이 얼마나 많은지를 보여주는 상대적 지표
  -'EMI_to_Salary' : 월급 대비 대출 상환 부담, 재정 압박도를 나타내며 월급에 비해 대출 상환이 얼마나 많은지를 나타내는 상대적 지표
  -'Delay_per_Loan' : 대출 1개당 평균 연체 횟수, 대출 개수 대비 얼마나 자주 늦었는지에 대한 지표
  - 정답값과 다른 컬럼들의 상관계수 확인 결과 유의미한 상관계수를 나타내는 컬럼이 없어 앞서 전처리에서 제외한 컬럼들을 제외하고 파생변수를 추가한 모든 컬럼을 사용했습니다

## 8. 모델링 및 학습
  - 성능 향상을 위해서 tabtranformer 모델을 사용했습니다.
  - 이 때 학습률을 0.001을 주고  또한 특정 neuron의 의존 방지를 위해 dropout을 0.1로 설정했습니다
  - 모델이 무거워 배치사이즈를 1024로 설정하고 에폭을 50으로 설정했습니다.

## 9. valid score 출력
 - valid score는 0.76으로 나쁘지않은 성능을 보였지만 추후 개선의 여지는 분명합니다.
   <img width="901" height="593" alt="valid score 출력 1~26" src="https://github.com/user-attachments/assets/438e8783-9e63-4d7b-ae10-35e4dd163bb2" />
   <img width="898" height="540" alt="valid score 27~50" src="https://github.com/user-attachments/assets/fb5be657-a630-4b80-94ff-b46d313aba2c" />

## 10. 개선사항 
  - 학습용 스코어와 평가용 스코어의 차이가 약 9%로 과적합 시작 초기상태일수도 있습니다
  - 조기에 학습을 끝내는 early stopping 방식을 사용해봐도 좋을 것 같습니다.  
  - lr(learning rate)을 줄이거나 dropout을 약간 증가시키는 등 과적합 방지를 조심해야할 것 같습니다
  - weight=class_weights를 설정하는등 클래스 가중치 설정도 해볼만할 것 같습니다
  - 여러방면에서의 개선작업을 시행하느라 시간을 많이 써서 추후에 추가적인 개선을 할 여지가 있습니다. 

