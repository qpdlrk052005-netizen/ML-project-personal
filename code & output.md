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


# 데이터 불러오기

col1=[
    # 가이드라인 관련 변수
    'ZONE',
    'GLMIN',
    'GLMAX', 
    # 연구 범위 한정 변수
    'SOURCES',
    # 실제로 부과된 형벌의 형태
    'SENTIMP', # 0 이면 벌금만, 4면 보호관찰만
    # 대체구금 개월 수
    'ALTMO', # 97이면 결측치.
    'ALTDUM', # 대체구금 선고 여부 0- 없음 / 1- 있음
    # 총 선고 기간
    'SENSPLT0', # 자유 제한 기간. 470이면 종신형. 
    # 무기징역/사형 구분
    'TOTPRISN' # 9996이면 종신형, 9997은 결측치 , 9998은 사형 **이걸 종신형의 지표로 사용해야 함.
]

df1=pd.read_csv('opafy21nid.csv',usecols=col1)
df2=pd.read_csv('opafy22nid.csv',usecols=col1)
df3=pd.read_csv('opafy23nid.csv',usecols=col1)

df=pd.concat([df1,df2,df3],axis=0)
df.to_csv('회귀및분류모델.csv', index=False)

# 결측치 확인 및 반응변수 'Class' 생성

df=pd.read_csv('회귀및분류모델.csv')
df=df[df['SOURCES']==1]
print(
    df.isna().sum()
)
'''df.dropna(subset=['SENSPLT0'],inplace=True)''' -> 없어도 Class를 구분할 수 있으므로, 제거할 필요 없음.    

df['Class'] = np.select(
    [
        ( (df['TOTPRISN']==9996) | (df['TOTPRISN']==9998)),
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

print(pd.crosstab(df['SENTIMP'], df['Class'], margins=True, dropna=False))

# 결과

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

| SENTIMP | Class 0<br>(Life) | Class 1<br>(기간형) | Class 2<br>(Probation Only) | Class 3<br>(Fine Only) | All |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 185 | 185 |
| 1 | 347 | 157,596 | 0 | 0 | 157,943 |
| 2 | 2 | 5,158 | 0 | 0 | 5,160 |
| 3 | 0 | 2,707 | 0 | 0 | 2,707 |
| 4 | 0 | 0 | 9,712 | 0 | 9,712 |
| **All** | **349** | **165,461** | **9,712** | **185** | **175,707** |

# 불일치 2건 분석
print(df[(df['Class']==0) & (df['SENTIMP']==2)])

# 결과
| Index | ZONE | SENSPLT0 | ALTDUM | ALTMO | SENTIMP | SOURCES | GLMIN | GLMAX | TOTPRISN | Class |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 156926 | D | 470.0 | 1 | 0 | 2 | 1 | 300.0 | 327.0 | 9996 | 0 |
| 169563 | D | 470.0 | 1 | 0 | 2 | 1 | 9996.0 | 9996.0 | 9996 | 0 |

# 파생변수를 만드는 함수
X=df[['GLMIN','GLMAX','ZONE']]
y=df['Class']

def add_GL_features(X) :
    result=X.copy()
    result['GLMIN_MONTH'] = result['GLMIN'].mask(result['GLMIN'].eq(9996),0)
    result['GLMAX_MONTH'] = result['GLMAX'].mask(result['GLMAX'].eq(9996),0)
    result['GLMIN_ZERO'] = result['GLMIN'].eq(0).astype('int')
    life=result['GLMIN'].eq(9996)
    finite_to_life=result['GLMAX'].eq(9996)
    result['GL_RANGE_TYPE'] = np.select([life & finite_to_life , ~life & finite_to_life],
                                        ['life','finite_to_life']
                                        ,default='finite')
    return result

# CV - 5 FOLD (StratifiedKFold) 설정
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

# baseline A 전처리기 생성
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
preprocessor_A = ColumnTransformer(
    transformers=[
        ('pass','passthrough',['GLMIN_MONTH','GLMAX_MONTH','GLMIN_ZERO']),
        ('Encoder',OneHotEncoder(sparse_output=False),['GL_RANGE_TYPE'])
    ],
    remainder='drop')

# dummy 모델 생성 및 평가
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
dummy_pipeline = Pipeline([('add',FunctionTransformer(add_GL_features)),
                        ('prepro',preprocessor_A),
                        ('model',DummyClassifier(strategy='most_frequent'))])
y_pred = cross_val_predict(dummy_pipeline,X,y,cv=cv)
print('혼동행렬')
print(confusion_matrix(y,y_pred))
print('결과')
print(classification_report(y,y_pred))

# 평가 결과
혼동행렬
| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0** | 0 | 349 | 0 | 0 | 349 |
| **Class 1** | 0 | 165,461 | 0 | 0 | 165,461 |
| **Class 2** | 0 | 9,712 | 0 | 0 | 9,712 |
| **Class 3** | 0 | 185 | 0 | 0 | 185 |
| **전체** | 0 | 175,707 | 0 | 0 | 175,707 |

결과
| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.00 | 0.00 | 0.00 | 349 |
| **Class 1: 기간형** | 0.94 | 1.00 | 0.97 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.00 | 0.00 | 0.00 | 9,712 |
| **Class 3: 벌금 Only** | 0.00 | 0.00 | 0.00 | 185 |
| **Accuracy** | — | — | **0.94** | 175,707 |
| **Macro average** | 0.24 | 0.25 | 0.24 | 175,707 |
| **Weighted average** | 0.89 | 0.94 | 0.91 | 175,707 |

# basline A 모델 - Logistic ( class_weight='balanced' )
Logisticbalanced=LogisticRegression(class_weight='balanced',max_iter=10000)
modelA1=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('logistic',Logisticbalanced)
])

y_pred=cross_val_predict(modelA1,X,y,cv=cv)
print('혼동행렬')
print(confusion_matrix(y,y_pred))
print('결과')
print(classification_report(y,y_pred))

# 평가 결과
혼동행렬
| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **328** | 21 | 0 | 0 | 349 |
| **Class 1: 기간형** | 3,210 | **98,392** | 53,062 | 10,797 | 165,461 |
| **Class 2: 보호관찰 Only** | 1 | 1,547 | **5,347** | 2,817 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 9 | 25 | **151** | 185 |
| **전체 예측 건수** | 3,539 | 99,969 | 58,434 | 13,765 | 175,707 |

결과
| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.09 | 0.94 | 0.17 | 349 |
| **Class 1: 기간형** | 0.98 | 0.59 | 0.74 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.09 | 0.55 | 0.16 | 9,712 |
| **Class 3: 벌금 Only** | 0.01 | 0.82 | 0.02 | 185 |
| **Accuracy** | — | — | **0.59** | 175,707 |
| **Macro average** | 0.29 | 0.73 | 0.27 | 175,707 |
| **Weighted average** | 0.93 | 0.59 | 0.71 | 175,707 |

# baseline A 모델 - Logistic ( class_weight=None )
Logistic=LogisticRegression(max_iter=10000)
modelA2=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('logistic',Logistic)
])

y_pred=cross_val_predict(modelA2,X,y,cv=cv)
print('혼동행렬')
print(confusion_matrix(y,y_pred))
print('결과')
print(classification_report(y,y_pred))

# 평과 결과
혼동행렬
| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0 | 349 | 0 | 0 | 349 |
| **Class 1: 기간형** | 0 | **165,461** | 0 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 9,712 | 0 | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체 예측 건수** | 0 | 175,707 | 0 | 0 | 175,707 |

결과
| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.00 | 0.00 | 0.00 | 349 |
| **Class 1: 기간형** | 0.94 | 1.00 | 0.97 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.00 | 0.00 | 0.00 | 9,712 |
| **Class 3: 벌금 Only** | 0.00 | 0.00 | 0.00 | 185 |
| **Accuracy** | — | — | **0.94** | 175,707 |
| **Macro average** | 0.24 | 0.25 | 0.24 | 175,707 |
| **Weighted average** | 0.89 | 0.94 | 0.91 | 175,707 |

# baseline A 모델 - Decision Tree ( class_weight='balanced' )
from sklearn.tree import DecisionTreeClassifier
Decision2 = DecisionTreeClassifier(class_weight='balanced',criterion='gini',random_state=42)
modelA4=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('tree',Decision2)
])

y_pred=cross_val_predict(modelA4,X,y,cv=cv)
print('혼동행렬')
print(confusion_matrix(y,y_pred))
print('결과')
print(classification_report(y,y_pred,zero_division=0))

# 평가 결과
혼동행렬
| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | **332** | 17 | 0 | 0 | 349 |
| **Class 1: 기간형** | 4,615 | **117,578** | 31,161 | 12,107 | 165,461 |
| **Class 2: 보호관찰 Only** | 1 | 2,557 | **4,281** | 2,873 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 14 | 20 | **151** | 185 |
| **전체 예측 건수** | 4,948 | 120,166 | 35,462 | 15,131 | 175,707 |

결과
| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.07 | 0.95 | 0.13 | 349 |
| **Class 1: 기간형** | 0.98 | 0.71 | 0.82 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.12 | 0.44 | 0.19 | 9,712 |
| **Class 3: 벌금 Only** | 0.01 | 0.82 | 0.02 | 185 |
| **Accuracy** | — | — | **0.70** | 175,707 |
| **Macro average** | 0.29 | 0.73 | 0.29 | 175,707 |
| **Weighted average** | 0.93 | 0.70 | 0.79 | 175,707 |

# baseline A 모델 - Decision Tree ( class_weight=None )
from sklearn.tree import DecisionTreeClassifier
Decision1 = DecisionTreeClassifier(criterion='gini',random_state=42)
modelA3=Pipeline([
    ('add',FunctionTransformer(add_GL_features,validate=False)),
    ('prepro',preprocessor_A),
    ('tree',Decision1)
])

y_pred=cross_val_predict(modelA3,X,y,cv=cv)
print('혼동행렬')
print(confusion_matrix(y,y_pred))
print('결과')
print(classification_report(y,y_pred,zero_division=0))

# 평가 결과
혼동행렬
| 실제 \ 예측 | Class 0 | Class 1 | Class 2 | Class 3 | 전체 |
|---|---:|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0 | 349 | 0 | 0 | 349 |
| **Class 1: 기간형** | 0 | **165,461** | 0 | 0 | 165,461 |
| **Class 2: 보호관찰 Only** | 0 | 9,712 | 0 | 0 | 9,712 |
| **Class 3: 벌금 Only** | 0 | 185 | 0 | 0 | 185 |
| **전체 예측 건수** | 0 | 175,707 | 0 | 0 | 175,707 |

결과
| 구분 | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| **Class 0: 무기징역·사형** | 0.00 | 0.00 | 0.00 | 349 |
| **Class 1: 기간형** | 0.94 | 1.00 | 0.97 | 165,461 |
| **Class 2: 보호관찰 Only** | 0.00 | 0.00 | 0.00 | 9,712 |
| **Class 3: 벌금 Only** | 0.00 | 0.00 | 0.00 | 185 |
| **Accuracy** | — | — | **0.94** | 175,707 |
| **Macro average** | 0.24 | 0.25 | 0.24 | 175,707 |
| **Weighted average** | 0.89 | 0.94 | 0.91 | 175,707 |



