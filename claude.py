import pandas as pd
import statsmodels.api as sm

raw_data = {
    "EMP_NO": [1, 2, 3, 4, 5, 6],
    "NAME": ["Aurora", "Bill", "Charlie", "Dragon", "Entropy", "False"],
    "AGE": [20, 24, 22, 30, 52, 14],
    "SALARY": [1000, 3000, 2000, 5000, 10000, 100],
    "DEPARTMENT": ["IT", "IT", "HR", "IT", "CEO", "HR"]
}

# 데이터프레임 생성
df = pd.DataFrame(raw_data)

# 독립변수(X)와 종속변수(y) 설정
X = df['AGE']
y = df['SALARY']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())



# 모델 요약 출력
print("=== 회귀분석 결과 ===")


# 예측값과 실제값의 비교
print("\n=== 예측값과 실제값 비교 ===")
predictions = model.predict(X)
comparison = pd.DataFrame({
    '실제 급여': y,
    '예측 급여': predictions,
    '차이': y - predictions
})
print(comparison)

# 모델 성능 지표
print("\n=== 모델 성능 지표 ===")
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")