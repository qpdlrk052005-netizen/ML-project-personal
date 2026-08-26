# 형벌 유형 분류: Baseline A·B 코드 및 출력

> **문서 상태:** Baseline A·B 통합 분석 기록  
> **분석 자료:** USSC FY2021–FY2023, `SOURCES == 1`  
> **평가 방법:** 5-fold Stratified Cross-Validation의 Out-of-Fold 예측  
> **주요 입력:** Baseline A는 `GLMIN`, `GLMAX`; Baseline B는 `GLMIN`, `GLMAX`, `ZONE`

이 문서는 데이터 구성, 반응변수 생성, 피처 엔지니어링, 전처리 파이프라인과 Baseline A·B 분류모형의 실행 코드 및 출력을 순서대로 기록한다. 기존 Python 코드와 실행 결과는 변경하지 않고 문서 구조와 출력 형식만 통합하였다.

## 목차

1. [데이터 준비](#1-데이터-준비)
2. [피처 엔지니어링 및 검증 설정](#2-피처-엔지니어링-및-검증-설정)
3. [Baseline A 모형 평가](#3-baseline-a-모형-평가)
4. [Baseline B 모형 평가](#4-baseline-b-모형-평가)
5. [Baseline A·B 성능 비교](#5-baseline-ab-성능-비교)

---

## 1. 데이터 준비

### 1.1 연도별 열 이름 추출

```python
# 년도별 열 이름 추출 코드
import csv
import os

file_path = r"opafy25nid.csv"

print("파일 존재 여부:", os.path.exists(file_path))
print("파일 경로:", file_path)

with open(file_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    columns = next(reader)

with open("columns25년도.txt", "w", encoding="utf-8") as f:
    for col in columns:
        f.write(col + "\n")
```

### 1.2 분석 변수 선택 및 연도별 자료 결합

```python
# 데이터 불러오기

col=[
    'ZONE',
    'GLMIN',
    'GLMAX',
    'XMINSOR',
    'XMAXSOR',
    'XFOLSOR',
    'XCRHISSR',
    'GDLINEHI',
    'NOCOMP',
    'AMENDYR',
    'SOURCES',
    'SENTIMP',
    'TOTPRISN'
]
df1=pd.read_csv('opafy21nid.csv', usecols=col, low_memory=False)
df2=pd.read_csv('opafy22nid.csv', usecols=col, low_memory=False)
df3=pd.read_csv('opafy23nid.csv', usecols=col, low_memory=False)

df=pd.concat([df1,df2,df3], axis=0, ignore_index=True)
df.to_csv('회귀및분류모델.csv', index=False)
```

### 1.3 결측치 확인 및 반응변수 생성

```python
# 결측치 확인 및 반응변수 'Class' 생성

df=pd.read_csv('회귀및분류모델.csv')
df=df[df['SOURCES']==1]

print(
    df.isna().sum()
)

# SENSPLT0가 결측이어도 Class를 구분할 수 있으므로 제거하지 않음

df['Class'] = np.select(
    [
        ((df['TOTPRISN']==9996) | (df['TOTPRISN']==9998)),
        df['SENTIMP'] == 0,
        df['SENTIMP'] == 4
    ],
    [
        0,
        3,
        2
    ],
    default=1
)

print(
    df['Class'].value_counts()
)

print(
    pd.crosstab(
        df['SENTIMP'],
        df['Class'],
        margins=True,
        dropna=False
    )
)
```

#### 실행 결과

```text
ZONE        0
SENSPLT0    3
ALTDUM      0
ALTMO       0
SENTIMP     0
SOURCES     0
GLMIN       0
GLMAX       0
TOTPRISN    0
dtype: int64

Class
1    165461
2      9712
0       349
3       185
Name: count, dtype: int64
```

| SENTIMP | Class 0<br>(Life) | Class 1<br>(기간형) | Class 2<br>(Probation Only) | Class 3<br>(Fine Only) | All |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 185 | 185 |
| 1 | 347 | 157,596 | 0 | 0 | 157,943 |
| 2 | 2 | 5,158 | 0 | 0 | 5,160 |
| 3 | 0 | 2,707 | 0 | 0 | 2,707 |
| 4 | 0 | 0 | 9,712 | 0 | 9,712 |
| **All** | **349** | **165,461** | **9,712** | **185** | **175,707** |

### 1.4 반응변수 정의 불일치 사례 확인

```python
# 불일치 2건 분석
print(df[(df['Class']==0) & (df['SENTIMP']==2)])
```

#### 실행 결과

| Index | ZONE | SENSPLT0 | ALTDUM | ALTMO | SENTIMP | SOURCES | GLMIN | GLMAX | TOTPRISN | Class |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 156926 | D | 470.0 | 1 | 0 | 2 | 1 | 300.0 | 327.0 | 9996 | 0 |
| 169563 | D | 470.0 | 1 | 0 | 2 | 1 | 9996.0 | 9996.0 | 9996 | 0 |

---

## 2. 피처 엔지니어링 및 검증 설정

### 2.1 GL 파생변수 생성

```python
# 파생변수를 만드는 함수

X=df[['GLMIN','GLMAX','ZONE']]
y=df['Class']

def add_GL_features(X):
    result=X.copy()

    result['GLMIN_MONTH'] = result['GLMIN'].mask(
        result['GLMIN'].eq(9996),
        0
    )

    result['GLMAX_MONTH'] = result['GLMAX'].mask(
        result['GLMAX'].eq(9996),
        0
    )

    result['GLMIN_ZERO'] = result['GLMIN'].eq(0).astype('int')

    life=result['GLMIN'].eq(9996)
    finite_to_life=result['GLMAX'].eq(9996)

    result['GL_RANGE_TYPE'] = np.select(
        [
            life & finite_to_life,
            ~life & finite_to_life
        ],
        [
            'life',
            'finite_to_life'
        ],
        default='finite'
    )

    return result
```

### 2.2 Stratified 5-fold Cross-Validation 설정

```python
# CV - 5 FOLD (StratifiedKFold) 설정

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

### 2.3 Baseline A 전처리기 구성

```python
# baseline A 전처리기 생성

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor_A = ColumnTransformer(
    transformers=[
        (
            'pass',
            'passthrough',
            ['GLMIN_MONTH','GLMAX_MONTH','GLMIN_ZERO']
        ),
        (
            'Encoder',
            OneHotEncoder(sparse_output=False),
            ['GL_RANGE_TYPE']
        )
    ],
    remainder='drop'
)
```

---

## 3. Baseline A 모형 평가

### 3.1 Dummy Classifier

#### 실행 코드

```python
# dummy 모델 생성 및 평가

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

dummy_pipeline = Pipeline([
    ('add',FunctionTransformer(add_GL_features)),
    ('prepro',preprocessor_A),
    ('model',DummyClassifier(strategy='most_frequent'))
])

y_pred = cross_val_predict(
    dummy_pipeline,
    X,
    y,
    cv=cv
)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0 | 349 | 0 | 0 | 349 |
| **Class 1: 기간형** | 0 | 165,461 | 0 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 9,712 | 0 | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체** | 0 | 175,707 | 0 | 0 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.00 | 0.00 | 0.00 | 349 |
| **Class 1: 기간형** | 0.94 | 1.00 | 0.97 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.00 | 0.00 | 0.00 | 9,712 |
| **Class 3: 벌금 Only** | 0.00 | 0.00 | 0.00 | 185 |
| **Accuracy** | — | — | **0.94** | 175,707 |
| **Macro average** | 0.24 | 0.25 | 0.24 | 175,707 |
| **Weighted average** | 0.89 | 0.94 | 0.91 | 175,707 |

---

### 3.2 Logistic Regression — None (`class_weight=None`)

#### 실행 코드

```python
# baseline A 모델 - Logistic ( class_weight=None )

Logistic=LogisticRegression(max_iter=10000)

modelA2=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('logistic',Logistic)
])

y_pred=cross_val_predict(modelA2,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0 | 349 | 0 | 0 | 349 |
| **Class 1: 기간형** | 0 | **165,461** | 0 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 9,712 | 0 | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체 예측 건수** | 0 | 175,707 | 0 | 0 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.00 | 0.00 | 0.00 | 349 |
| **Class 1: 기간형** | 0.94 | 1.00 | 0.97 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.00 | 0.00 | 0.00 | 9,712 |
| **Class 3: 벌금 Only** | 0.00 | 0.00 | 0.00 | 185 |
| **Accuracy** | — | — | **0.94** | 175,707 |
| **Macro average** | 0.24 | 0.25 | 0.24 | 175,707 |
| **Weighted average** | 0.89 | 0.94 | 0.91 | 175,707 |

---

### 3.3 Logistic Regression — Balanced (`class_weight='balanced'`)

#### 실행 코드

```python
# baseline A 모델 - Logistic ( class_weight='balanced' )

Logisticbalanced=LogisticRegression(
    class_weight='balanced',
    max_iter=10000
)

modelA1=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('logistic',Logisticbalanced)
])

y_pred=cross_val_predict(modelA1,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **328** | 21 | 0 | 0 | 349 |
| **Class 1: 기간형** | 3,210 | **98,392** | 53,062 | 10,797 | 165,461 |
| **Class 2: 보호관찰 Only** | 1 | 1,547 | **5,347** | 2,817 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 9 | 25 | **151** | 185 |
| **전체 예측 건수** | 3,539 | 99,969 | 58,434 | 13,765 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.09 | 0.94 | 0.17 | 349 |
| **Class 1: 기간형** | 0.98 | 0.59 | 0.74 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.09 | 0.55 | 0.16 | 9,712 |
| **Class 3: 벌금 Only** | 0.01 | 0.82 | 0.02 | 185 |
| **Accuracy** | — | — | **0.59** | 175,707 |
| **Macro average** | 0.29 | 0.73 | 0.27 | 175,707 |
| **Weighted average** | 0.93 | 0.59 | 0.71 | 175,707 |

---

### 3.4 Decision Tree — None (`class_weight=None`)

#### 실행 코드

```python
# baseline A 모델 - Decision Tree ( class_weight=None )

from sklearn.tree import DecisionTreeClassifier

Decision1=DecisionTreeClassifier(
    criterion='gini',
    random_state=42
)

modelA3=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('tree',Decision1)
])

y_pred=cross_val_predict(modelA3,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0 | 349 | 0 | 0 | 349 |
| **Class 1: 기간형** | 0 | **165,461** | 0 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 9,712 | 0 | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체 예측 건수** | 0 | 175,707 | 0 | 0 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.00 | 0.00 | 0.00 | 349 |
| **Class 1: 기간형** | 0.94 | 1.00 | 0.97 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.00 | 0.00 | 0.00 | 9,712 |
| **Class 3: 벌금 Only** | 0.00 | 0.00 | 0.00 | 185 |
| **Accuracy** | — | — | **0.94** | 175,707 |
| **Macro average** | 0.24 | 0.25 | 0.24 | 175,707 |
| **Weighted average** | 0.89 | 0.94 | 0.91 | 175,707 |

---

### 3.5 Decision Tree — Balanced (`class_weight='balanced'`)

#### 실행 코드

```python
# baseline A 모델 - Decision Tree ( class_weight='balanced' )

Decision2=DecisionTreeClassifier(
    class_weight='balanced',
    criterion='gini',
    random_state=42
)

modelA4=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('tree',Decision2)
])

y_pred=cross_val_predict(modelA4,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **332** | 17 | 0 | 0 | 349 |
| **Class 1: 기간형** | 4,615 | **117,578** | 31,161 | 12,107 | 165,461 |
| **Class 2: 보호관찰 Only** | 1 | 2,557 | **4,281** | 2,873 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 14 | 20 | **151** | 185 |
| **전체 예측 건수** | 4,948 | 120,166 | 35,462 | 15,131 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.07 | 0.95 | 0.13 | 349 |
| **Class 1: 기간형** | 0.98 | 0.71 | 0.82 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.12 | 0.44 | 0.19 | 9,712 |
| **Class 3: 벌금 Only** | 0.01 | 0.82 | 0.02 | 185 |
| **Accuracy** | — | — | **0.70** | 175,707 |
| **Macro average** | 0.29 | 0.73 | 0.29 | 175,707 |
| **Weighted average** | 0.93 | 0.70 | 0.79 | 175,707 |

---

### 3.6 Random Forest — None (`class_weight=None`)

#### 실행 코드

```python
# baseline A 모델 - Randomforest ( class_weight=None )

from sklearn.ensemble import RandomForestClassifier

RandomForestModel=RandomForestClassifier(
    n_estimators=300,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

modelA_RF=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('randomforest',RandomForestModel)
])

y_pred=cross_val_predict(modelA_RF,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0 | 349 | 0 | 0 | 349 |
| **Class 1: 기간형** | 0 | **165,450** | 11 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 9,703 | **9** | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체 예측 건수** | 0 | 175,687 | 20 | 0 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.0000 | 0.0000 | 0.0000 | 349 |
| **Class 1: 기간형** | 0.9417 | 0.9999 | 0.9700 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.4500 | 0.0009 | 0.0018 | 9,712 |
| **Class 3: 벌금 Only** | 0.0000 | 0.0000 | 0.0000 | 185 |
| **Accuracy** | — | — | **0.9417** | 175,707 |
| **Macro average** | 0.3479 | 0.2502 | 0.2430 | 175,707 |
| **Weighted average** | 0.9117 | 0.9417 | 0.9135 | 175,707 |

---

### 3.7 Random Forest — Balanced (`class_weight='balanced_subsample'`)

#### 실행 코드

```python
# baseline A 모델 - Randomforest ( class_weight='balanced_subsample' )

RandomForestBalanced=RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

modelA_RF_balanced=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('randomforest',RandomForestBalanced)
])

y_pred=cross_val_predict(modelA_RF_balanced,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **328** | 21 | 0 | 0 | 349 |
| **Class 1: 기간형** | 3,103 | **119,783** | 31,455 | 11,120 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 2,570 | **4,316** | 2,826 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 14 | 20 | **151** | 185 |
| **전체 예측 건수** | 3,431 | 122,388 | 35,791 | 14,097 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.0956 | 0.9398 | 0.1735 | 349 |
| **Class 1: 기간형** | 0.9787 | 0.7239 | 0.8323 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.1206 | 0.4444 | 0.1897 | 9,712 |
| **Class 3: 벌금 Only** | 0.0107 | 0.8162 | 0.0211 | 185 |
| **Accuracy** | — | — | **0.7090** | 175,707 |
| **Macro average** | 0.3014 | 0.7311 | 0.3042 | 175,707 |
| **Weighted average** | 0.9285 | 0.7090 | 0.7946 | 175,707 |

---

### 3.8 HistGradientBoosting — None (`class_weight=None`)

#### 실행 코드

```python
# baseline A 모델 - HistGradientBoosting ( class_weight=None )

from sklearn.ensemble import HistGradientBoostingClassifier

HistGradientModel=HistGradientBoostingClassifier(
    class_weight=None,
    learning_rate=0.1,
    max_iter=100,
    random_state=42
)

modelA_HGB=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('histgradient',HistGradientModel)
])

y_pred=cross_val_predict(modelA_HGB,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **188** | 161 | 0 | 0 | 349 |
| **Class 1: 기간형** | 760 | **164,676** | 25 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 9,712 | 0 | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체 예측 건수** | 948 | 174,734 | 25 | 0 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.1983 | 0.5387 | 0.2899 | 349 |
| **Class 1: 기간형** | 0.9424 | 0.9953 | 0.9681 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.0000 | 0.0000 | 0.0000 | 9,712 |
| **Class 3: 벌금 Only** | 0.0000 | 0.0000 | 0.0000 | 185 |
| **Accuracy** | — | — | **0.9383** | 175,707 |
| **Macro average** | 0.2852 | 0.3835 | 0.3145 | 175,707 |
| **Weighted average** | 0.8879 | 0.9383 | 0.9122 | 175,707 |

---

### 3.9 HistGradientBoosting — Balanced (`class_weight='balanced'`)

#### 실행 코드

```python
# baseline A 모델 - HistGradientBoosting ( class_weight='balanced' )

HistGradientBalanced=HistGradientBoostingClassifier(
    class_weight='balanced',
    learning_rate=0.1,
    max_iter=100,
    random_state=42
)

modelA_HGB_balanced=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('histgradient',HistGradientBalanced)
])

y_pred=cross_val_predict(modelA_HGB_balanced,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **333** | 16 | 0 | 0 | 349 |
| **Class 1: 기간형** | 5,772 | **116,483** | 31,087 | 12,119 | 165,461 |
| **Class 2: 보호관찰 Only** | 1 | 2,565 | **4,273** | 2,873 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 14 | 20 | **151** | 185 |
| **전체 예측 건수** | 6,106 | 119,078 | 35,380 | 15,143 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.0545 | 0.9542 | 0.1032 | 349 |
| **Class 1: 기간형** | 0.9782 | 0.7040 | 0.8187 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.1208 | 0.4400 | 0.1895 | 9,712 |
| **Class 3: 벌금 Only** | 0.0100 | 0.8162 | 0.0197 | 185 |
| **Accuracy** | — | — | **0.6900** | 175,707 |
| **Macro average** | 0.2909 | 0.7286 | 0.2828 | 175,707 |
| **Weighted average** | 0.9280 | 0.6900 | 0.7817 | 175,707 |

---

## 4. Baseline B 모형 평가

### 4.1 입력 변수 및 전처리기

#### 입력 변수 구성

| 구분 | 변수 | 처리 방식 |
|---|---|---|
| 가이드라인 하한 | `GLMIN_MONTH` | `GLMIN == 9996`이면 0으로 변환한 뒤 수치형으로 통과 |
| 가이드라인 상한 | `GLMAX_MONTH` | `GLMAX == 9996`이면 0으로 변환한 뒤 수치형으로 통과 |
| 하한 0 여부 | `GLMIN_ZERO` | `GLMIN == 0`이면 1, 아니면 0 |
| 가이드라인 범위 유형 | `GL_RANGE_TYPE` | `finite`, `finite_to_life`, `life`를 원-핫 인코딩 |
| 양형구간 | `ZONE` | `A`, `B`, `C`, `D`를 원-핫 인코딩 |

#### 전처리기 생성 코드

```python
# baseline B 전처리기 생성

preprocessor_B = ColumnTransformer(
    transformers=[
        (
            'pass',
            'passthrough',
            ['GLMIN_MONTH','GLMAX_MONTH','GLMIN_ZERO']
        ),
        (
            'Encoder1',
            OneHotEncoder(sparse_output=False),
            ['GL_RANGE_TYPE']
        ),
        (
            'Encoder2',
            OneHotEncoder(sparse_output=False),
            ['ZONE']
        )
    ],
    remainder='drop'
)
```

---

### 4.2 Logistic Regression — None (`class_weight=None`)

#### 실행 코드

```python
# baseline B 모델 - Logistic ( class_weight=None )

Logistic=LogisticRegression(max_iter=10000)

modelB2=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_B),
    ('logistic',Logistic)
])

y_pred=cross_val_predict(modelB2,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0 | 349 | 0 | 0 | 349 |
| **Class 1: 기간형** | 0 | **165,461** | 0 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 9,712 | 0 | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체 예측 건수** | 0 | 175,707 | 0 | 0 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.0000 | 0.0000 | 0.0000 | 349 |
| **Class 1: 기간형** | 0.9417 | 1.0000 | 0.9700 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.0000 | 0.0000 | 0.0000 | 9,712 |
| **Class 3: 벌금 Only** | 0.0000 | 0.0000 | 0.0000 | 185 |
| **Accuracy** | — | — | **0.9417** | 175,707 |
| **Macro average** | 0.2354 | 0.2500 | 0.2425 | 175,707 |
| **Weighted average** | 0.8868 | 0.9417 | 0.9134 | 175,707 |

---

### 4.3 Logistic Regression — Balanced (`class_weight='balanced'`)

#### 실행 코드

```python
# baseline B 모델 - Logistic ( class_weight='balanced' )

Logisticbalanced=LogisticRegression(
    class_weight='balanced',
    max_iter=10000
)

modelB1=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_B),
    ('logistic',Logisticbalanced)
])

y_pred=cross_val_predict(modelB1,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **328** | 21 | 0 | 0 | 349 |
| **Class 1: 기간형** | 3,033 | **102,453** | 49,061 | 10,914 | 165,461 |
| **Class 2: 보호관찰 Only** | 1 | 1,800 | **5,091** | 2,820 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 10 | 24 | **151** | 185 |
| **전체 예측 건수** | 3,362 | 104,284 | 54,176 | 13,885 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.0976 | 0.9398 | 0.1768 | 349 |
| **Class 1: 기간형** | 0.9824 | 0.6192 | 0.7596 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.0940 | 0.5242 | 0.1594 | 9,712 |
| **Class 3: 벌금 Only** | 0.0109 | 0.8162 | 0.0215 | 185 |
| **Accuracy** | — | — | **0.6148** | 175,707 |
| **Macro average** | 0.2962 | 0.7249 | 0.2793 | 175,707 |
| **Weighted average** | 0.9306 | 0.6148 | 0.7245 | 175,707 |

---

### 4.4 Decision Tree — None (`class_weight=None`)

#### 실행 코드

```python
# baseline B 모델 - Decision Tree ( class_weight=None )

Decision1=DecisionTreeClassifier(
    criterion='gini',
    random_state=42
)

modelB3=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_B),
    ('tree',Decision1)
])

y_pred=cross_val_predict(modelB3,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0 | 349 | 0 | 0 | 349 |
| **Class 1: 기간형** | 0 | **165,461** | 0 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 9,712 | 0 | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체 예측 건수** | 0 | 175,707 | 0 | 0 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.0000 | 0.0000 | 0.0000 | 349 |
| **Class 1: 기간형** | 0.9417 | 1.0000 | 0.9700 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.0000 | 0.0000 | 0.0000 | 9,712 |
| **Class 3: 벌금 Only** | 0.0000 | 0.0000 | 0.0000 | 185 |
| **Accuracy** | — | — | **0.9417** | 175,707 |
| **Macro average** | 0.2354 | 0.2500 | 0.2425 | 175,707 |
| **Weighted average** | 0.8868 | 0.9417 | 0.9134 | 175,707 |

---

### 4.5 Decision Tree — Balanced (`class_weight='balanced'`)

#### 실행 코드

```python
# baseline B 모델 - Decision Tree ( class_weight='balanced' )

Decision2=DecisionTreeClassifier(
    class_weight='balanced',
    criterion='gini',
    random_state=42
)

modelB4=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_B),
    ('tree',Decision2)
])

y_pred=cross_val_predict(modelB4,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **332** | 17 | 0 | 0 | 349 |
| **Class 1: 기간형** | 4,615 | **117,661** | 31,022 | 12,163 | 165,461 |
| **Class 2: 보호관찰 Only** | 1 | 2,557 | **4,281** | 2,873 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 15 | 19 | **151** | 185 |
| **전체 예측 건수** | 4,948 | 120,250 | 35,322 | 15,187 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.0671 | 0.9513 | 0.1254 | 349 |
| **Class 1: 기간형** | 0.9785 | 0.7111 | 0.8236 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.1212 | 0.4408 | 0.1901 | 9,712 |
| **Class 3: 벌금 Only** | 0.0099 | 0.8162 | 0.0196 | 185 |
| **Accuracy** | — | — | **0.6968** | 175,707 |
| **Macro average** | 0.2942 | 0.7299 | 0.2897 | 175,707 |
| **Weighted average** | 0.9283 | 0.6968 | 0.7864 | 175,707 |

---

### 4.6 Random Forest — None (`class_weight=None`)

#### 실행 코드

```python
# baseline B 모델 - Randomforest ( class_weight=None )

RandomForestModel=RandomForestClassifier(
    n_estimators=300,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

modelB_RF=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_B),
    ('randomforest',RandomForestModel)
])

y_pred=cross_val_predict(modelB_RF,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0 | 349 | 0 | 0 | 349 |
| **Class 1: 기간형** | 0 | **165,450** | 11 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 9,703 | **9** | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체 예측 건수** | 0 | 175,687 | 20 | 0 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.0000 | 0.0000 | 0.0000 | 349 |
| **Class 1: 기간형** | 0.9417 | 0.9999 | 0.9700 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.4500 | 0.0009 | 0.0018 | 9,712 |
| **Class 3: 벌금 Only** | 0.0000 | 0.0000 | 0.0000 | 185 |
| **Accuracy** | — | — | **0.9417** | 175,707 |
| **Macro average** | 0.3479 | 0.2502 | 0.2430 | 175,707 |
| **Weighted average** | 0.9117 | 0.9417 | 0.9135 | 175,707 |

---

### 4.7 Random Forest — Balanced (`class_weight='balanced_subsample'`)

#### 실행 코드

```python
# baseline B 모델 - Randomforest ( class_weight='balanced_subsample' )

RandomForestBalanced=RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

modelB_RF_balanced=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_B),
    ('randomforest',RandomForestBalanced)
])

y_pred=cross_val_predict(
    modelB_RF_balanced,
    X,
    y,
    cv=cv
)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **328** | 21 | 0 | 0 | 349 |
| **Class 1: 기간형** | 3,103 | **119,872** | 31,310 | 11,176 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 2,574 | **4,312** | 2,826 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 15 | 19 | **151** | 185 |
| **전체 예측 건수** | 3,431 | 122,482 | 35,641 | 14,153 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.0956 | 0.9398 | 0.1735 | 349 |
| **Class 1: 기간형** | 0.9787 | 0.7245 | 0.8326 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.1210 | 0.4440 | 0.1902 | 9,712 |
| **Class 3: 벌금 Only** | 0.0107 | 0.8162 | 0.0211 | 185 |
| **Accuracy** | — | — | **0.7095** | 175,707 |
| **Macro average** | 0.3015 | 0.7311 | 0.3043 | 175,707 |
| **Weighted average** | 0.9285 | 0.7095 | 0.7949 | 175,707 |

---

### 4.8 HistGradientBoosting — None (`class_weight=None`)

#### 실행 코드

```python
# baseline B 모델 - HistGradientBoosting ( class_weight=None )

HistGradientModel=HistGradientBoostingClassifier(
    class_weight=None,
    learning_rate=0.1,
    max_iter=100,
    random_state=42
)

modelB_HGB=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_B),
    ('histgradient',HistGradientModel)
])

y_pred=cross_val_predict(modelB_HGB,X,y,cv=cv)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **119** | 230 | 0 | 0 | 349 |
| **Class 1: 기간형** | 714 | **164,709** | 38 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 16 | 9,668 | **28** | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체 예측 건수** | 849 | 174,792 | 66 | 0 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.1402 | 0.3410 | 0.1987 | 349 |
| **Class 1: 기간형** | 0.9423 | 0.9955 | 0.9682 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.4242 | 0.0029 | 0.0057 | 9,712 |
| **Class 3: 벌금 Only** | 0.0000 | 0.0000 | 0.0000 | 185 |
| **Accuracy** | — | — | **0.9382** | 175,707 |
| **Macro average** | 0.3767 | 0.3348 | 0.2931 | 175,707 |
| **Weighted average** | 0.9111 | 0.9382 | 0.9124 | 175,707 |

---

### 4.9 HistGradientBoosting — Balanced (`class_weight='balanced'`)

#### 실행 코드

```python
# baseline B 모델 - HistGradientBoosting ( class_weight='balanced' )

HistGradientBalanced=HistGradientBoostingClassifier(
    class_weight='balanced',
    learning_rate=0.1,
    max_iter=100,
    random_state=42
)

modelB_HGB_balanced=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_B),
    ('histgradient',HistGradientBalanced)
])

y_pred=cross_val_predict(
    modelB_HGB_balanced,
    X,
    y,
    cv=cv
)

print('혼동행렬')
print(confusion_matrix(y,y_pred,labels=[0,1,2,3]))
print('결과')
print(
    classification_report(
        y,
        y_pred,
        labels=[0,1,2,3],
        digits=4,
        zero_division=0
    )
)
```

#### 평가 결과

##### 혼동행렬

| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **333** | 16 | 0 | 0 | 349 |
| **Class 1: 기간형** | 5,769 | **116,602** | 30,921 | 12,169 | 165,461 |
| **Class 2: 보호관찰 Only** | 1 | 2,572 | **4,264** | 2,875 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 15 | 19 | **151** | 185 |
| **전체 예측 건수** | 6,103 | 119,205 | 35,204 | 15,195 | 175,707 |

##### 분류 보고서

| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.0546 | 0.9542 | 0.1032 | 349 |
| **Class 1: 기간형** | 0.9782 | 0.7047 | 0.8192 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.1211 | 0.4390 | 0.1899 | 9,712 |
| **Class 3: 벌금 Only** | 0.0099 | 0.8162 | 0.0196 | 185 |
| **Accuracy** | — | — | **0.6906** | 175,707 |
| **Macro average** | 0.2909 | 0.7285 | 0.2830 | 175,707 |
| **Weighted average** | 0.9280 | 0.6906 | 0.7824 | 175,707 |

---

## 5. Baseline A·B 성능 비교

Baseline A와 Baseline B는 동일한 자료, 반응변수, 5-fold Stratified Cross-Validation 및 분류모형을 사용한다. 두 Baseline의 차이는 입력 변수에 `ZONE`을 포함하는지 여부이다.

| Baseline | 입력 정보 |
|---|---|
| **Baseline A** | `GLMIN`, `GLMAX` 및 가이드라인 범위 파생변수 |
| **Baseline B** | Baseline A 입력 정보 + `ZONE` |

다음 값은 `cross_val_predict()`로 생성한 전체 Out-of-Fold 예측을 기준으로 계산한 결과이다.

| Baseline | 모형 | 클래스 가중치 | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---:|---:|---:|---:|
| A | Dummy Classifier | Most Frequent | 0.9400 | 0.2400 | 0.2500 | 0.2400 |
| A | Logistic Regression | None | 0.9400 | 0.2400 | 0.2500 | 0.2400 |
| B | Logistic Regression | None | 0.9417 | 0.2354 | 0.2500 | 0.2425 |
| A | Logistic Regression | Balanced | 0.5900 | 0.2900 | 0.7300 | 0.2700 |
| B | Logistic Regression | Balanced | 0.6148 | 0.2962 | 0.7249 | 0.2793 |
| A | Decision Tree | None | 0.9400 | 0.2400 | 0.2500 | 0.2400 |
| B | Decision Tree | None | 0.9417 | 0.2354 | 0.2500 | 0.2425 |
| A | Decision Tree | Balanced | 0.7000 | 0.2900 | 0.7300 | 0.2900 |
| B | Decision Tree | Balanced | 0.6968 | 0.2942 | 0.7299 | 0.2897 |
| A | Random Forest | None | 0.9417 | 0.3479 | 0.2502 | 0.2430 |
| B | Random Forest | None | 0.9417 | 0.3479 | 0.2502 | 0.2430 |
| A | Random Forest | Balanced (`balanced_subsample`) | 0.7090 | 0.3014 | 0.7311 | 0.3042 |
| B | Random Forest | Balanced (`balanced_subsample`) | 0.7095 | 0.3015 | 0.7311 | 0.3043 |
| A | HistGradientBoosting | None | 0.9383 | 0.2852 | 0.3835 | 0.3145 |
| B | HistGradientBoosting | None | 0.9382 | 0.3767 | 0.3348 | 0.2931 |
| A | HistGradientBoosting | Balanced | 0.6900 | 0.2909 | 0.7286 | 0.2828 |
| B | HistGradientBoosting | Balanced | 0.6906 | 0.2909 | 0.7285 | 0.2830 |

> Baseline A의 Logistic Regression과 Decision Tree 결과는 기존 실행 출력의 소수점 둘째 자리까지 기록된 값을 그대로 사용하였다. 세부적인 클래스별 성능과 혼동행렬은 각 모형의 평가 결과에 제시하였다.
