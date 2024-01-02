# 데이터 불러오기
diabetes <- read.csv('~/R과제/diabetes_prediction_dataset.csv')

# 데이터 확인 - 
#gender (성별): 관측 대상의 성별을 나타내는 문자열 변수입니다.
# age (나이): 관측 대상의 나이를 나타내는 숫자 변수입니다.
# hypertension (고혈압): 관측 대상의 고혈압 여부를 나타내는 이진 변수입니다. 0은 고혈압이 없음을, 1은 고혈압이 있음을 나타냅니다.
# heart_disease (심장 질환): 관측 대상의 심장 질환 여부를 나타내는 이진 변수입니다. 0은 심장 질환 없음을, 1은 심장 질환 있음을 나타냅니다.
# smoking_history (흡연 이력): 관측 대상의 흡연 이력을 나타내는 문자열 변수입니다.
# bmi (체질량 지수): 관측 대상의 체질량 지수(Body Mass Index)를 나타내는 숫자 변수입니다.
# HbA1c_level (HbA1c 수치): 관측 대상의 HbA1c 수치를 나타내는 숫자 변수입니다.
# blood_glucose_level (혈당 수치): 관측 대상의 혈당 수치를 나타내는 숫자 변수입니다.
# diabetes (당뇨병): 관측 대상의 당뇨병 여부를 나타내는 이진 변수입니다. 0은 당뇨병이 없음을, 1은 당뇨병이 있음을 나타냅니다.
head(diabetes)

# null 값 확인
sum(is.null(diabetes))


# 데이터의 컬럼 확인
names(diabetes)

# 데이터의 형식 확인
str(diabetes)

# 데이터 요약
summary(diabetes)

# 컬럼 내용 확인
table(diabetes$smoking_history)
table(diabetes$hypertension)
table(diabetes$heart_disease)
table(diabetes$diabetes)

# 범주형으로 변경
diabetes$gender <- as.factor(diabetes$gender)
diabetes$hypertension <- as.factor(diabetes$hypertension)
diabetes$heart_disease <- as.factor(diabetes$heart_disease)
diabetes$smoking_history <- as.factor(diabetes$smoking_history)
diabetes$diabetes <- as.factor(diabetes$diabetes)
str(diabetes)


cont_dia <- diabetes[c('age', 'bmi', 'HbA1c_level', 'blood_glucose_level')]

cor(cont_dia)

# 랜덤한 훈련-의사결정 나무
set.seed(1)
train_sample <- sample(100000,90000)
train_sample

# 테스트 데이터셋 생성
dia_train <- diabetes[train_sample, ]
dia_test <- diabetes[-train_sample, ]

round(prop.table(table(dia_train$diabetes)), 2)
round(prop.table(table(dia_test$diabetes)), 2)

# 모델학습 : C5.0 패키지 로딩
install.packages('C50')
library(C50)

# 모델학습 : C5.0을 이용한 당뇨병 예측 식별 모델 학습
# 정확도 = 100 - 2.7 = 97.3%
# 오분류율 = (86+2364)/90000 = 2.7%
dia_model <- C5.0(dia_train[-9], dia_train$diabetes)
dia_model

summary(dia_model)

# 모델 성능 평가 
# 결과 정확도 = (9116+597)/10000 = 97.13%
# 오분류율 = (20+267)/10000 = 2.87%
# 민감도 = 9116/(9116+20) = 99.78%
# 특이도 = 597/(597+267) = 69.1%
# 정밀도 = 9116/(9116+267) = 97.15%
# 재현율 = 9116/(9116+20) = 99.78%
# F-척도 = (9116*2)/(2*9116+267+20) = 0.98

install.packages('gmodels')
library(gmodels)

dia_pred <- predict(dia_model, dia_test[-9])
CrossTable(dia_test$diabetes, dia_pred,
           prop.chisq = TRUE, prop.c = TRUE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# 추정된 예측 확률 - 확률 값의 핪은 1
dia_pred <- predict(dia_model, 
                       dia_test[-9], type = "class")
head(dia_pred)

# 카파 통계량 활용하기
# caret 패키지
# kappa 값 = 0.7912
install.packages("caret")
library(caret)
confusionMatrix(dia_pred,
                dia_test$diabetes, positive = "1")

# irr 패키지
install.packages('irr')
library(irr)
kappa2(data.frame(dia_test$diabetes, dia_pred))

# ROC와 AUC 활용하기
install.packages("pROC")
library(pROC)


# 모델 예측 확률 구하기
dia_pred_prob <- predict(dia_model, dia_test[-9], type = "prob")

# ROC 곡선 및 AUC 계산
dia_roc <- roc(dia_test$diabetes, dia_pred_prob[, "1"])  # '1' 클래스에 대한 예측 확률 사용
dia_roc

# AUC 값 출력
auc(dia_roc)

# 시각화
plot(dia_roc, main = "ROC Curve",
     print.auc = TRUE, col = "blue", lwd = 2)

# 미래의 성능예측 - 교차 검증
library(caret)

set.seed(2)
folds <- createFolds(diabetes$diabetes, k = 10)
str(folds)

# 교차 검증 적용하기
library(C50)
library(irr)

cv_results <- lapply(folds, function(x){
  dia_train <- diabetes[-x, ]
  dia_test <- diabetes[x, ]
  dia_model <- C5.0(diabetes ~., data = dia_train)
  dia_pred <- predict(dia_model, dia_test)
  dia_actual <- dia_test$diabetes
  kappa <- kappa2(data.frame(dia_actual, dia_pred))$value
  return(kappa)})

str(cv_results)

mean(unlist(cv_results))

