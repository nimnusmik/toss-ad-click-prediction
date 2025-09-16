# EDA

## Data 설명

* train.parquet [파일] :
    * 총 10,704,179개 샘플
    * 총 119개 ('clicked' Target 컬럼 포함) 컬럼 존재
        * gender : 성별
        * age_group : 연령 그룹
        * inventory_id : 지면 ID
        * day_of_week : 주번호
        * hour : 시간
        * seq : 유저 서버 로그 시퀀스
        * l_feat_* : 속성 정보 피처 (l_feat_14는 Ads set)
        * feat_e_* : 정보영역 e 피처
        * feat_d_* : 정보영역 d 피처
        * feat_c_* : 정보영역 c 피처
        * feat_b_* : 정보영역 b 피처
        * feat_a_* : 정보영역 a 피처
        * history_a_* : 과거 인기도 피처 a
        * history_b_* : 과거 피쳐 b
        * clicked : 클릭 여부 (Label)

* test.parquet [파일] :
    * 총 1,527,298개 샘플
    * 총 118개 ('ID' 식별자 컬럼 포함) 컬럼 존재
        * ID : 샘플 식별자
        * gender : 성별
        * age_group : 연령 그룹
        * inventory_id : 지면 ID
        * day_of_week : 주번호
        * hour : 시간
        * seq : 유저 서버 로그 시퀀스
        * l_feat_* : 속성 정보 피처 (l_feat_14는 Ads set)
        * feat_e_* : 정보영역 e 피처
        * feat_d_* : 정보영역 d 피처
        * feat_c_* : 정보영역 c 피처
        * feat_b_* : 정보영역 b 피처
        * feat_a_* : 정보영역 a 피처
        * history_a_* : 과거 인기도 피처
        * history_b_* : 과거 피쳐 b

* sample_submission.csv [파일] - 제출 양식  
    * 총 1,527,298개 샘플
        * ID : 샘플 식별자
        * clicked : 광고를 클릭할 확률 (0 ~ 1)

## EDA by Column 

### Clicked
![alt text](img/image-5.png)

### Gender
![alt text](img/image.png)

![alt text](img/image-6.png)

### Age group
![alt text](img/image-1.png)

![alt text](img/image-7.png)

### Inventory id
![alt text](img/image-2.png)

![alt text](img/image-8.png)

### Day of week
![alt text](img/image-3.png)

![alt text](img/image-9.png)

### Hour
![alt text](img/image-4.png)

![alt text](img/image-10.png)

### Seq


#### seq의 구조 확인

```
shape: (10, 2)
┌─────────────────────────────────┬─────────┐
│ seq                             ┆ clicked │
│ ---                             ┆ ---     │
│ str                             ┆ i32     │
╞═════════════════════════════════╪═════════╡
│ 9,18,269,516,57,97,527,74,317,… ┆ 0       │
│ 9,144,269,57,516,97,527,74,315… ┆ 0       │
│ 269,516,57,97,165,527,74,77,31… ┆ 0       │
│ 269,57,516,21,214,269,561,214,… ┆ 0       │
│ 144,269,57,516,35,479,57,516,5… ┆ 0       │
│ 9,516,57,527,74,77,532,101,132… ┆ 0       │
│ 9,516,57,221,97,63,520,426,221… ┆ 0       │
│ 57,516,338,416,74,527,77,318,4… ┆ 0       │
│ 57,516,97,165,74,527,317,269,3… ┆ 0       │
│ 9,269,57,516,74,207,452,452,24… ┆ 0       │
└─────────────────────────────────┴─────────┘
```

#### seq의 길이 분포 확인

```
shape: (3_672, 2)
┌─────────┬───────┐
│ seq_len ┆ len   │
│ ---     ┆ ---   │
│ u32     ┆ u32   │
╞═════════╪═══════╡
│ 1       ┆ 59300 │
│ 2       ┆ 16240 │
│ 3       ┆ 16665 │
│ 4       ┆ 17581 │
│ 5       ┆ 18667 │
│ …       ┆ …     │
│ 10103   ┆ 2     │
│ 10340   ┆ 1     │
│ 10560   ┆ 3     │
│ 12935   ┆ 1     │
│ 14441   ┆ 1     │
└─────────┴───────┘
```

#### seq 길이에 따른 클릭률 확인

![alt text](img/image-12.png)

![alt text](img/image-13.png)

#### 첫 10개의 seq와 마지막 10개의 seq 확인

```
shape: (10, 21)
┌───────┬───────┬───────┬───────┬───┬────────┬────────┬─────────┬─────────┐
│ seq_1 ┆ seq_2 ┆ seq_3 ┆ seq_4 ┆ … ┆ seq_-8 ┆ seq_-9 ┆ seq_-10 ┆ clicked │
│ ---   ┆ ---   ┆ ---   ┆ ---   ┆   ┆ ---    ┆ ---    ┆ ---     ┆ ---     │
│ i32   ┆ i32   ┆ i32   ┆ i32   ┆   ┆ i32    ┆ i32    ┆ i32     ┆ i32     │
╞═══════╪═══════╪═══════╪═══════╪═══╪════════╪════════╪═════════╪═════════╡
│ 9     ┆ 18    ┆ 269   ┆ 516   ┆ … ┆ 311    ┆ 269    ┆ 317     ┆ 0       │
│ 9     ┆ 144   ┆ 269   ┆ 57    ┆ … ┆ 311    ┆ 269    ┆ 315     ┆ 0       │
│ 269   ┆ 516   ┆ 57    ┆ 97    ┆ … ┆ 452    ┆ 77     ┆ 527     ┆ 0       │
│ 269   ┆ 57    ┆ 516   ┆ 21    ┆ … ┆ 479    ┆ 469    ┆ 317     ┆ 0       │
│ 144   ┆ 269   ┆ 57    ┆ 516   ┆ … ┆ 317    ┆ 77     ┆ 74      ┆ 0       │
│ 9     ┆ 516   ┆ 57    ┆ 527   ┆ … ┆ 77     ┆ 74     ┆ 101     ┆ 0       │
│ 9     ┆ 516   ┆ 57    ┆ 221   ┆ … ┆ 227    ┆ 193    ┆ 463     ┆ 0       │
│ 57    ┆ 516   ┆ 338   ┆ 416   ┆ … ┆ 416    ┆ 57     ┆ 479     ┆ 0       │
│ 57    ┆ 516   ┆ 97    ┆ 165   ┆ … ┆ 511    ┆ 408    ┆ 108     ┆ 0       │
│ 9     ┆ 269   ┆ 57    ┆ 516   ┆ … ┆ 479    ┆ 408    ┆ 342     ┆ 0       │
└───────┴───────┴───────┴───────┴───┴────────┴────────┴─────────┴─────────┘
```

![alt text](img/image-14.png)
![alt text](img/image-35.png)
 
 <br>

![alt text](img/image-15.png)
![alt text](img/image-36.png)

 <br>

![alt text](img/image-16.png)
![alt text](img/image-44.png)

 <br>
 
![alt text](img/image-17.png)
![alt text](img/image-38.png)

 <br>
 
![alt text](img/image-19.png)
![alt text](img/image-39.png)

 <br>
 
![alt text](img/image-43.png)
![alt text](img/image-40.png)

 <br>
 
![alt text](img/image-21.png)
![alt text](img/image-41.png)

 <br>
 
![alt text](img/image-22.png)
![alt text](img/image-42.png)

 <br>
 
![alt text](img/image-23.png)
![alt text](img/image-45.png)

 <br>
 
![alt text](img/image-24.png)
![alt text](img/image-46.png)

 <br>
 
![alt text](img/image-25.png)
![alt text](img/image-47.png)

 <br>
 
![alt text](img/image-26.png)
![alt text](img/image-48.png)

 <br>
 
![alt text](img/image-27.png)
![alt text](img/image-49.png)

 <br>
 
![alt text](img/image-28.png)
![alt text](img/image-50.png)

 <br>
 
![alt text](img/image-29.png)
![alt text](img/image-51.png)

 <br>
 
![alt text](img/image-30.png)
![alt text](img/image-52.png)

 <br>
 
![alt text](img/image-31.png)
![alt text](img/image-53.png)

 <br>
 
![alt text](img/image-32.png)
![alt text](img/image-54.png)

 <br>
 
![alt text](img/image-33.png)
![alt text](img/image-55.png)

 <br>
 
![alt text](img/image-34.png)
![alt text](img/image-56.png)

 <br>
 
 ### l_feat_*

 ![alt text](img/image-57.png)

 ### feat_a_*

![alt text](img/image-58.png)

 ### feat_b_*

![alt text](img/image-59.png)

 ### feat_c_*

![alt text](img/image-60.png)

 ### feat_d_*

![alt text](img/image-61.png)

 ### feat_e_*

![alt text](img/image-62.png)

 ### history_a_*

![alt text](img/image-63.png)

 ### history_b_*

![alt text](img/image-64.png)


## 결측치 처리

TBA

## Feature Engineering

TBA