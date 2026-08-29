# 데이터 전처리

```python
# Baseline Full 예측변수
feature_cols = [
    'AMENDYR',
    'GDLINEHI',
    'NOCOMP',
    'XFOLSOR',
    'XCRHISSR',
    'XMINSOR',
    'XMAXSOR',
    'STATMIN',
    'STATMAX',
    'GLMIN',
    'GLMAX',
    'ZONE',
    'FINEMIN',
    'FINEMAX'
]

# 표본 제한과 Class 생성에만 사용하는 변수
auxiliary_cols = [
    'SOURCES',
    'SENTIMP',
    'TOTPRISN'
]

cols = feature_cols + auxiliary_cols

df22=pd.read_csv('opafy22nid.csv',usecols=cols)

df22=df22[df22['SOURCES']==1]

# 사형이 있는가?
print(
    '사형으로 판정된 사건의 개수 = ',
    (df22['TOTPRISN']==9998).sum()
)

df22['Class']=np.select(
    [
        df22['TOTPRISN']==9996,
        df22['SENTIMP']==4,
        df22['SENTIMP']==0
    ],
    [
        0,
        2,
        3,
    ],
    default=1
)

# 결측치 분석
print(
    '결측치 개수\n',
    df22.isna().sum().sort_values()
)

df22.to_csv('baselinedata2022.csv',index=False)

df=pd.read_csv('baselinedata2022.csv')

print(
    'Class별 분포\n',
    df['Class'].value_counts().sort_index()
)

```

# 결과

```text
사형으로 판정된 사건의 개수 =  0

결측치 개수
ZONE           0
XCRHISSR       0
SOURCES        0
SENTIMP        0
XFOLSOR        0
GLMAX          0
GLMIN          0
AMENDYR        0
XMINSOR        0
XMAXSOR        0
TOTPRISN       0
Class          0
GDLINEHI      22
NOCOMP        22
STATMAX      126
STATMIN      127
FINEMAX     2069
FINEMIN     2088
dtype: int64

Class별 분포
 Class
0      111
1    57488
2     3347
3       60
Name: count, dtype: int64
```
