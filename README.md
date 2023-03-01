# PNU_Tensorflow
[K-DIGITAL 부산대] AI 활용 빅데이터분석 풀스택웹서비스 SW 개발자 양성과정

### Chapter01 인공지능, 딥러닝, 텐서플로 설치
- 인공지능과 딥러닝
- 텐서플로 설치

### Chapter02 텐서플로 기초
- 즉시 실행 모드와 텐서 생성
- 텐서플로 연산

### Chapter03 회귀(Regression)
- 평균 제곱 오차 손실함수
- 넘파이 단순 선형 회귀
- 자동 미분 계산
- 텐서플로 단순 선형 회귀
- 다변수 선형 회귀
- tf.keras.optimizers를 이용한 학습
- 다항식 회귀

### Chapter04 tf.keras를 사용한 회귀
- 순차형(Sequential) 모델
- 함수형 Model
- 모델 저장 및 로드

### Chapter05 완전 연결 신경망 분류(Classfication)
- 원-핫 인코딩과 교차 엔트로피 오차
- 활성화 함수
- 분류 성능평가
- 1-Dense 층(1뉴런) AND, OR 분류
- 1-Dense 층(2뉴런) AND, OR 분류
- 2층 신경망: XOR 이진 분류
- 2D 정규분포 데이터 생성 및 분류
- IRIS 데이터 분류

### Chapter06 데이터 셋: tf.keras.datasets
- Boston_housing 데이터 셋
- IMDB 데이터 셋
- Reuters 데이터 셋
- MNIST 데이터 셋
- Fashion_MNIST 데이터 셋
- CIFAR - 10 데이터 셋
- CIFAR - 100 데이터 셋

### Chapter07 콜백: 학습 모니터링
- 콜백
- 텐서보드

### Chapter08 그래디언트 소실과 과적합
- 그래디언트 소실과 가중치 초기화
- 배치정규화
- 과적합, 가중치 규제, 드롭아웃
- 드롭아웃

### Chapter09 합성곱 신경망(CNN)
- 패딩
- 1차원 풀링
- 1차원 합성곱
- 1차원 합성곱 신경망(CNN) 분류
- 2차원 풀링
- 2차원 합성곱
- 2차원 합성곱 신경망(CNN)

### Chapter10 함수형 API
- tf.keras.layers 층
- 함수형 API 합성곱 신경망(CNN)

### Chapter11 사전학습 모델: tf.keras.applications
- VGG 모델
- ResNet 모델
- Inception, GoogLeNet

### Chapter13 업 샘플링, 전치 합성곱, 오토 인코더, GAN
- 업 샘플링
- 전치 합성곱
- 오토 인코더
- 절대적 생성모델(GAN) 모델

### Chapter14 영상 분할, 검출, CoLab
- Oxford-IIIT Pet Dataset
- Oxford-IIIT Pet Dataset 분류
- U-Net 영상분할
- 물체 위치검출 및 분류
- Colab 



<a href="https://blog.naver.com/kjh920411/221889422562/" target="_blank">
<img src = "https://blogfiles.pstatic.net/MjAyMDA0MDNfMTA2/MDAxNTg1OTExMDQ1NTI5.P17sN8p2a9-VLxBj925pnc5VObM_bjGZXMF9b_dW_zcg.B3qmjqddpLwMzuH02GGhkM79Nxut3pmCGhVe-PGx6dIg.PNG.kjh920411/image.png"> <a/> <br/>


# 머신러닝이란?
명시적으로 프로그램을 작성하지 않고, 컴퓨터가 스스로 규칙을 학습하는 연구분야

+ 지도학습
+ 비지도학습
+ 강화학습

#### > 학습 순서
1. 모듈 임포트: import
2. 데이터 전처리
    + 데이터 불러오기 read
    + 데이터 나누기 iloc, train_test_split
3. 모델링: 객체 생성
4. 학습: fit
5. 예측: predict
6. 시각화(예측값, 실제값): plot, scatter
7. 평가: score
---  
## 1. 지도학습 : 정답이 있는데이터를 통해 분류 / 결과예측
+ 회귀: 변수들 간의 상관관계를 찾는 것, 연속적인 데이터로부터 결과를 예측.(input, output 데이터 o)
+ 분류: 주어진 데이터를 주어진 범주(category)에 따라 분류. 예측결과가 숫자가 아닐 때(input 데이터만 o)
---    
### 1) 회귀 모델  
### (1) 단순 선형 회귀 모델

y = mx +b   
- X: 독립변수(원인) = 입력변수 = feature
- Y: 종속변수(결과) = 출력변수 = targer, label
- m: 기울기
- b: Y절편

<img src= "https://user-images.githubusercontent.com/51871037/211140756-dafc7368-3d3d-43d5-92a1-094b416da2ee.PNG"> 		
> 데이터는 좌표평면 위에 순서쌍 (X, Y)로 표현할 수 있다. <br> 모든 점을 지나는 직선이 가장 이상적인 선형회귀 모델이지만 현실적으로 불가능하므로, 가능한 범위에서 가장 좋은 직선을 찾아야 한다. <br/>
    
```
** 선형회귀모델 최적의 직선 찾기 **
- RSS: 잔차 제곱의 합
- OLS: 최소제곱법 (노이즈(이상치)에 취약하다)
```  
    
### (2) 다중 선형 회귀
y = mx1 + nx2 + kx3 +b <br/>
<img src= "https://user-images.githubusercontent.com/51871037/211141939-7525a292-d357-4d55-a0e8-4024bd295d42.PNG" width = "40%">

### (3) 다항회귀
y = b + m1x + m2x^2 + m3x^3 + ... + mnx^n <br/>
<img src= "https://user-images.githubusercontent.com/51871037/211142296-58c53119-4d99-4d6a-9b43-22fbda15ecbe.png">
+ 참고사이트: https://arachnoid.com/polysolve/
```   
** 원 핫 인코딩 **
- 표현하고 싶은 값만 1로, 나머지는 모두 0으로 
```
<img src= "https://user-images.githubusercontent.com/51871037/211142177-5b1aeb21-c010-4bf0-872e-cc5532019d77.PNG">

```
** 다중 공선성**
독립 변수들 간에 서로 강한 상관관계를 가지면서 회귀계수 추정의 오류가 나타나는 문제
=> 하나의 피처가 다른 피처에 영향을 미치는 것
ex) Home + Library + cafe = 1
        Home = 1 - (Library + cafe)
=> Dummy Column이 n 개면 n-1개만 사용해서 다중 공선성 문제를 해결할수있다.
```
    
### (4) 회귀 모델 평가 지표
- 평균 절대 오차 (MAE): 실제 값과 예측 값의 차이를 절댓값으로 변환해 평균한 것
- 평균 제곱 오차(MSE): 실제 값과 예측 값의 차이를 제곱해 평균한 것. 회귀분석 손실함수로 자주 사용
- 평균 제곱 오차의 제곱근 (RMSE): MSE가 큰값은 더 크게, 작은값은 더 작게 만드는 단점이 있어 이를 보완.
- 결정계수(R²):  분산 기반으로 예측 성능을 평가. 1에 가까울수록 예측 정확도가 높음  
---
### 2) 분류 모델    
### (1) 로지스틱 회귀
- 분류모델의 대표 알고리즘
- 선형 회귀 방식을 분류에 적용한 알고리즘.
- 데이터가 어떤 범주에 속할 확률을 0~1사이의 값으로 예측, 더 높은 범주에 속하는 쪽으로 분류
  
#### > 시그모이드 함수
<img src="https://user-images.githubusercontent.com/51871037/211141156-bc39c2db-e4ad-42a7-9e8b-ff3b46d690fd.PNG">
<img src="https://user-images.githubusercontent.com/51871037/211141163-7406b99b-ea9e-440f-ae43-21155171b564.PNG">
  
#### > 혼동행렬
<img src="https://user-images.githubusercontent.com/51871037/211141174-b1c0cd5f-d76d-4bdd-baff-0461fbcc4610.PNG">
  
 + True Positive (TP):  실제 참을 참이라고 예측
 + True Negative (NP): 실제 거짓을 거짓이라고 예측
 + False Positive (FP): 실제 거짓을 참이라고 예측
 + False Negative (NP): 실제 참을 거짓이라고 예측 <br/>
 
#### > 지표, 계산법
+ 민감도 (Sensitivity):	(TP / (TP + FN)) 양성 중 맞춘 양성의 수
+ 특이도 (Specificity):	(TN / (FP + TN)) 음성 중 맞춘 음성의 수
+ 정밀도 (Precision):	(TP / (TP + FP)) 양성이라고 판정 한 것 중에 실제 양성 수
+ 재현율 (Recall):	(TP / (TP + FN)) 전체 양성 수에서 검출 양성 수
+ 정확도 (accuracy):	((TP + TN) / (TP + FN + FP + TN)) 전체 개수 중에서 양성과 음성을 맞춘 수
+ F1 score: Precision과 Recall의 조화평균(역수의 평균). 데이터 label이 불균형 구조일 때, 모델의 성능을 정확하게 평가할 수 있으며, 성능을 하나의 숫자로 표현할 수 있다

---
## 2. 비지도 학습
+ 정답이 없는 데이터를 통해
+ 데이터의 유의미한 패턴/ 구조 발견 <br/>
 
  
### 1) Clustering
+ 유사한 특징을 가지는 데이터들을 그룹화

### (1) K-Means
+ 데이터를 K개의 그룹(클러스터)로 군집화하는 알고리즘.
+ 각 데이터로부터 이들이 속한 클러스터의 중심점까지의 평균 거리를 계산
  
### 1-1) K-Means 동작 순서
1. K값 설정(데이터가 많을 때 몇개의 그룹으로 나눌지 고민해야함)
2. 지정된 K개 만큼의 랜덤 좌표 설정
3. 모든 데이터로부터 가장 가까운 중심점 선택
4. 데이터들의 평균 중심으로 중심점 이동
5. 중심점이 더 이상 이동되지 않을 때 까지 반복
  
### 1-2) '2. 지정된 K개 만큼의 랜덤 좌표 설정' 의 문제점
+ Random Initialization Trap: 랜덤으로 좌표가 설정되다보니 매번 결과가 달라짐(원치않는 결과가 나올 수 있다.) 
+ 각 센터들의 거리가 짧으면 클러스터링이 제대로 되지 않을 수 있다. <br/>

### 1-3) Elbow Method
최적의 k 판단 방법
1. K 변화에 따른 중심점까지의 평균 거리 비교
2. 경사가 완만해지는 지점의 K 선정: 너무 많지않은 클러스터 갯수, 평균거리의 값은 작은 지점

### (2) K-Means++ 
k-means의 문제점 개선
1. 데이터 중에서 랜덤으로 1개를 중심점으로 선택
2. 나머지 데이터로부터 중심점까지의 거리 계산
3. 중심점과 가장 먼 지점의 데이터를 다음 중심점으로 선택
4. 중심점이 K개가 될 때까지 반복
5. K-Means 전통적인 방식으로 진행
  
#### > Euclidean Distance 유클리드 거리
+ 두 지점의 거리를 잴 때 직선으로 잇는 것<br/>
<img src="https://user-images.githubusercontent.com/51871037/211141221-756d53e6-8c1d-48e4-936c-5dce465fd426.PNG">
  
#### > Manhattan Distance 맨해튼 거리
+ 두 지점을 계단식으로 잇는 것 <br/>
<img src="https://user-images.githubusercontent.com/51871037/211141231-1b0dceb0-f3ff-4ee4-9300-63dec1e6552c.PNG">
  
#### > Cosine Similarity 코사인 유사도
+ 지점들간의 각도. 각도가 작을 수록 유사도가 높다고 본다. <br/>
<img src="https://user-images.githubusercontent.com/51871037/211141239-9d620477-3d19-4672-845f-30532c8048c0.png">
