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


# 결측치 처리 -> SENSPCAP은 결측치가 1개씩이니까 지웠다.
df21_step1=pd.read_csv('opafy21nid.csv',usecols=['SENSPCAP','GLMIN','GLMAX','SOURCES']).dropna(subset=['SENSPCAP'])
df22_step1=pd.read_csv('opafy22nid.csv',usecols=['SENSPCAP','GLMIN','GLMAX','SOURCES']).dropna(subset=['SENSPCAP'])
df23_step1=pd.read_csv('opafy23nid.csv',usecols=['SENSPCAP','GLMIN','GLMAX','SOURCES']).dropna(subset=['SENSPCAP'])
df_step1=pd.concat([df21_step1,df22_step1,df23_step1],axis=0)

# SOURCES가 들어왔으니 새로운 결측치 확인
print(
    df_step1.isna().sum()
)

# output
SENSPCAP       0
SOURCES        0
GLMIN       1373
GLMAX       1373
dtype: int64

# SOURCES 변수가 6,7,8,9인 경우만 결측 처리가 되었는가?
missing=df_step1[df_step1.isna().any(axis=1)]
print('가이드라인이 없거나 확정하기 어려운 사건에 얼마나 몰려있는가?\n' , missing['SOURCES'].value_counts().sort_index(ascending=False))
print('총 개수 = ',missing['SOURCES'].value_counts().sum())

# output
가이드라인이 없거나 확정하기 어려운 사건에 얼마나 몰려있는가?
 SOURCES
9    864
8    102
7     83
6    324
Name: count, dtype: int64
총 개수 =  1373
