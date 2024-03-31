library(ggplot2)
library(caret)
library(lattice)
library(nortest)
player33<-read.csv("all_player.csv")
player<-read.csv("player.csv")

ggplot(data = player, aes(x = position)) +
  geom_bar()

ggplot(data = player, aes(x = cards_red)) +
  geom_bar()

ggplot(data = player, aes(x = cards_yellow)) +
  geom_bar()

install.packages(c("readxl", "ggplot"))     
plot(density(player$age),
            pch = 16,
            main = "Density Plot",
            xlab = "age",
            ylab = "density")
nortest::lillie.test(player$age)
lillie.test(player$age)
fivenum(player$age)
       
plot(density(player$goals_assists_per90),
            pch = 16,
            main = "Density Plot",
            xlab = "goals_assists_per90",
            ylab = "density")
nortest::lillie.test(player$goals_assists_per90)
lillie.test(player$goals_assists_per90)
fivenum(player$goals_assists_per90)



player2<-read.csv("real_player_data.csv")

#linear model

lm1 = lm(sca~tackles_mid_3rd+passes_received+touches_att_pen_area+dribbles_completed+progressive_passes_received, data = player2)
summary(lm1)

lm2 = glm(sca~tackles_mid_3rd+passes_received+touches_att_pen_area+dribbles_completed+progressive_passes_received, data = player2)
summary(lm2)




set.seed(22)
train.index <- sample(x=1:nrow(Prostate), size=ceiling(0.8*nrow(Prostate) ))

train = Prostate[train.index, ]
test = Prostate[-train.index, ]

install.packages("pheatmap")
if(!require(pheatmap)){
  install.packages("pheatmap")
  library(pheatmap)
}

library(pheatmap)
library(DMwR2)
library(mice)
library(RANN)

playerA <- preProcess(player, method = "knnImpute")
player <- predict(playerA, player)
head(model.matrix(sca ~ . -player, data = player))

pheatmap(player33)

#-----------------------------------------------------------
library(glmnet)
library(Matrix)
library(dplyr)
library(ggplot2)

player33<-read.csv("player2.csv")
data(player33)
#檢查離群值
str(player33)
#隨機抽樣
n <- nrow(player33)
set.seed(123)
subplayer33 <- sample(seq_len(n), size = round(0.7 * n))
traindata <- player33[subplayer33,]%>% as.matrix()
testdata <- player33[ - subplayer33,]%>% as.matrix()
trainx <- traindata[,-c(1)]
trainy <- traindata[,c(1)]
testx <- testdata[,-c(1)]
testy <- testdata[,c(1)]

#調參 lamda
ridge <- cv.glmnet(x = trainx,y = trainy,alpha = 0)
#交叉驗證 預設k=10，alpha = 0為ridge, =1為lasso
ridge
#視覺化&選自變量
coef(ridge, s = "lambda.min") %>% 
  as.matrix() %>% 
  as.data.frame() %>% 
  add_rownames(var = "var") %>% 
  `colnames<-`(c("var","coef")) %>%
  filter(var != "(Intercept)") %>%  #剔除截距項
  top_n(10, wt = coef) %>% 
  ggplot(aes(coef, reorder(var, coef))) +
  geom_bar(stat = "identity", width=0.2,
           color="blue", fill=rgb(0.1,0.4,0.5,0.7))+
  xlab("Coefficient") +
  ylab(NULL)

#預測
future <- predict(ridge,newx = testx, s = ridge$lambda.min)
future <- as.data.frame(future)
final <- cbind(future,testy) %>% data.frame()
final <- mutate(final,mape=abs(s1-(testy)/(testy)))
mean(final$mape)





# Installing Packages
install.packages("data.table")
install.packages("dplyr")
install.packages("glmnet")
install.packages("ggplot2")
install.packages("caret")
install.packages("xgboost")
install.packages("e1071")
install.packages("cowplot")

# load packages
library(data.table) # used for reading and manipulation of data
library(dplyr)     # used for data manipulation and joining
library(glmnet)     # used for regression
library(ggplot2) # used for ploting
library(caret)     # used for modeling
library(xgboost) # used for building XGBoost model
library(e1071)     # used for skewness
library(cowplot) # used for combining multiple plots

set.seed(123)
control = trainControl(method ="cv", number = 5)
Grid_ri_reg = expand.grid(alpha = 0, lambda = seq(0.001, 0.1,
                                                  by = 0.0002))
Ridge_model = train(x = trainx,
                    y = trainy,
                    method = "glmnet",
                    trControl = control,
                    tuneGrid = Grid_reg
                    
)
Ridge_model

# mean validation score
mean(Ridge_model$resample$RMSE)

# Plot
plot(Ridge_model, main="Ridge Regression")
