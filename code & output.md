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

```

# 분석에서 제외할 변수 탐구
```python
print(
    '분석 제외할 대상(사형 선고)\n',
    (df['STATMIN']==9997).sum(),
    (df['STATMAX']==9997).sum(),
)
```

# 결과
```text
분석 제외할 대상(사형 선고)
0 0
```

# 변수 특성 탐구
```python
print(
    df.info()
)
```

# 결과
```text
RangeIndex: 61006 entries, 0 to 61005
Data columns (total 18 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   ZONE      61006 non-null  str    
 1   FINEMIN   58918 non-null  float64
 2   FINEMAX   58937 non-null  float64
 3   NOCOMP    60984 non-null  float64
 4   SENTIMP   61006 non-null  int64  
 5   SOURCES   61006 non-null  int64  
 6   XCRHISSR  61006 non-null  float64
 7   XFOLSOR   61006 non-null  float64
 8   AMENDYR   61006 non-null  float64
 9   GLMIN     61006 non-null  float64
 10  GLMAX     61006 non-null  float64
 11  STATMAX   60880 non-null  float64
 12  STATMIN   60879 non-null  float64
 13  TOTPRISN  61006 non-null  int64  
 14  XMAXSOR   61006 non-null  float64
 15  XMINSOR   61006 non-null  float64
 16  GDLINEHI  60984 non-null  str    
 17  Class     61006 non-null  int64  
dtypes: float64(12), int64(4), str(2)
```


# 범주형 변수 분석
```python
print(
    'Class별 분포\n',
    df['Class'].value_counts().sort_index()
)
categorical=[
    'AMENDYR',
    'GDLINEHI',
    'XCRHISSR',
    'ZONE'
]

for i in categorical :
    print(
        f'{i}의 빈도'
    )
    print(
        df[i].value_counts()
    )
```

# 결과
```text
Class별 분포
 Class
0      111
1    57488
2     3347
3       60
Name: count, dtype: int64

AMENDYR의 빈도
AMENDYR
2018.0000    60806
2016.0000      155
2015.0000       14
2014.0000        7
2012.0000        3
2008.0000        3
2013.0000        2
1998.0000        2
1999.0000        2
1997.0000        2
2001.0000        2
2002.0000        1
2004.0000        1
2006.0000        1
2010.0000        1
1994.0000        1
2003.0000        1
1992.0000        1
2000.0000        1
Name: count, dtype: int64

GDLINEHI의 빈도
GDLINEHI
2D1.1     19377
2L1.2     11931
2K2.1      8803
2B1.1      5148
2L1.1      4006
2G2.2      1422
2B3.1      1402
2S1.1       978
2G2.1       738
2L2.2       656
2A2.2       518
2G1.3       472
2A3.5       306
2A1.1       278
2P1.1       277
2T1.1       261
2A2.4       242
2X4.1       210
2C1.1       210
2X5.2       203
2D1.2       203
2A6.1       184
2S1.3       181
2A2.1       179
2T1.4       139
2A3.1       135
2P1.2       125
2D2.1       124
2A4.1       120
2M5.2       118
2B4.1       116
2J1.2       111
2K1.4       101
2X3.1        94
2B5.1        86
2L2.1        84
2A1.2        80
2A6.2        78
2T1.6        69
2A2.3        68
2B2.1        68
2G1.1        67
2Q2.1        67
2H1.1        62
2D2.2        51
2Q1.2        49
2A3.4        48
2E1.1        48
2A3.2        45
2A1.3        40
2A1.4        39
2B2.3        39
2E3.1        38
2G3.1        33
2A5.2        27
2J1.6        27
2A1.5        27
2B5.3        23
2B3.2        23
2D1.8        22
2N2.1        20
2M5.3        19
2M5.1        17
2B1.4        17
2C1.2        15
2X1.1        14
2N1.1        13
2J1.3        13
2Q1.3        12
2E4.1        12
2T3.1        12
2K2.5        12
2H3.3        10
2E1.4         9
2H4.1         9
2D1.5         8
2R1.1         8
2J1.4         8
2T1.9         7
2B1.5         7
2C1.8         7
2A3.3         6
2K1.3         6
2B3.3         5
2E2.1         4
2H3.1         4
2C1.3         4
2H2.1         3
2D1.12        3
2B6.1         3
2Q1.1         2
2P1.3         2
2E1.2         2
2A4.2         2
2K1.1         1
2D3.1         1
2T2.1         1
2N1.3         1
2E5.1         1
2G2.6         1
2Q1.4         1
2D2.3         1
2K2.6         1
2T2.2         1
2D1.10        1
2D1.11        1
2T1.3         1
Name: count, dtype: int64

XCRHISSR의 빈도
XCRHISSR
1.0000    24852
3.0000    10487
2.0000     8457
6.0000     6706
4.0000     6383
5.0000     4121
Name: count, dtype: int64

ZONE의 빈도
ZONE
D    46763
B     5182
A     4677
C     4384
Name: count, dtype: int64
```

# 결측치 처리 함수
```python
def Handling_missing(df) :
    X = df.copy()
    # GDLINEHI / NOCOMP 결측치 처리
    gdlmiss = X['GDLINEHI'].isna()
    nomiss = X['NOCOMP'].isna()
    X['OTHER_CAL'] = ( gdlmiss & nomiss ).astype('int')
    X['GDLINEHI'] = (
        X['GDLINEHI']
        .astype('string')
        .fillna('MISSING')
    )

    # STATMIN,STATMAX 결측치 처리
    X['STATMIN_MISSING'] = X['STATMIN'].isna().astype('int')
    X['STATMAX_MISSING'] = X['STATMAX'].isna().astype('int')
    X['MIN_TRUMP_STATUS'] = np.select(
        [
            X['XMINSOR'] == X['GLMIN'],
            X['XMINSOR'] < X['GLMIN'],
            X['XMINSOR'] > X['GLMIN']
        ],
        [
            'SAME',
            'RAISED',
            'LOWERED'
        ],
        default="UNKNOWN"
    )
    X['MAX_TRUMP_STATUS'] = np.select(
        [
            X['XMAXSOR'] == X['GLMAX'],
            X['XMAXSOR'] < X['GLMAX'],
            X['XMAXSOR'] > X['GLMAX']
        ],
        [
            'SAME',
            'RAISED',
            'LOWERED'
        ],
        default="UNKNOWN"
    )
    return X
```

# 특이값 처리 함수 (희소 범주 그대로)
```python
def preprocess(df) : 
    X = df.copy()
    # 특이값 처리
    X['XMINSOR_LIFE'] = (X['XMINSOR']==9996).astype('int')
    X['XMINSOR_ZERO'] = (X['XMINSOR']==0).astype('int')
    X['XMINSOR_MONTH'] = X['XMINSOR'].mask(
         X['XMINSOR']==9996,
        0
    )
    X['XMAXSOR_LIFE'] = (X['XMAXSOR']==9996).astype('int')
    X['XMAXSOR_MONTH'] = X['XMAXSOR'].mask(
        X['XMAXSOR']==9996,
        0
    )
    X['STATMIN_LIFE'] = (X['STATMIN']==9996).astype('int')
    X['STATMIN_ZERO'] = (X['STATMIN']==0).astype('int')
    X['STATMIN_MONTH'] = X['STATMIN'].mask(
         X['STATMIN']==9996,
        0
    )
    X['STATMAX_LIFE'] = (X['STATMAX']==9996).astype('int')
    X['STATMAX_MONTH'] = X['STATMAX'].mask(
        X['STATMAX']==9996,
        0
    )
    X['GLMIN_LIFE'] = (X['GLMIN']==9996).astype('int')
    X['GLMIN_ZERO'] = (X['GLMIN']==0).astype('int')
    X['GLMIN_MONTH'] = X['GLMIN'].mask(
         X['GLMIN']==9996,
        0
    )
    X['GLMAX_LIFE'] = (X['GLMAX']==9996).astype('int')
    X['GLMAX_MONTH'] = X['GLMAX'].mask(
        X['GLMAX']==9996,
        0
    ) 
    # 수치형 -> 범주형 전환
    X['XCRHISSR'] = X['XCRHISSR'].astype('Int64').astype('string')
    X['AMENDYR'] = X['AMENDYR'].astype('Int64').astype('string')
    return X
```


