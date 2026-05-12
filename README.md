# IMBK_DeepLearning_Competition
금융 데이터 딥러닝 학습 모델

## 1. 프로젝트명 
고객 등급 분류 딥러닝 모델 학습

## 2. 기간 
2026/05/12

## 3. 기술스택 
  - 데이터 전처리: Pandas, NumPy, Scikit-learn

  - 모델링/최적화: PyCaret, Optuna, LightGBM, CatBoost, AdaBoost

  - 시각화/해석: SHAP, Matplotlib

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
  정답값으로 쓰려는 Credit_Score에 결측치가 하나 존재했는데 Monthly_Balance역시 결측치가 존재해서 
결측치는 제거하고 보았습니다.
 값들중에서 로그변환이 필요해보이는 값을이 존재했는데 분류학습을 더 잘 학습하기위해
로그변환을 씌우고 진행했습니다. <img width="836" height="626" alt="딥러닝 로그변환 이유" src="https://github.com/user-attachments/assets/ed123875-e893-4e7c-a11c-11f3bc5e7cd5" />

## 7. FeatureSelection 및 모델 선정기준
  정답값과 크게 상관을 보이는 컬럼들이 없어서 모든 컬럼을 X값으로 넣었습니다.
  <img width="585" height="616" alt="상관계수" src="https://github.com/user-attachments/assets/eea6420e-be3f-422b-801a-ec333c7ae956" />


## 8. 모델링 및 학습

## 9. valid score 출력

## 10. 개선사항 
  

