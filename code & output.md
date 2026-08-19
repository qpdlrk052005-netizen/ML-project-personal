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
df2=pd.read_csv('opafy21nid.csv',usecols=col1)
df3=pd.read_csv('opafy21nid.csv',usecols=col1)

df=pd.concat([df1,df2,df3],axis=0)
df.to_csv('회귀및분류모델.csv', index=False)

# 결측치 제거 및 반응변수 'Class' 생성
df=pd.read_csv('회귀및분류모델.csv')
df=df[df['SOURCES']==1]
print(
    df.isna().sum()
)
df.dropna(subset=['SENSPLT0'],inplace=True)

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
1    151791
2      9168
3       198
0       177
Name: count, dtype: int64

Class      0       1     2    3     All
SENTIMP                                
0          0       0     0  198     198
1        177  143967     0    0  144144
2          0    4830     0    0    4830
3          0    2994     0    0    2994
4          0       0  9168    0    9168
All      177  151791  9168  198  161334

