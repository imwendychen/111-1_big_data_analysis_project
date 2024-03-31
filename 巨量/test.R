library(readr)
library(dplyr)
library(caret)
library(lattice)
library(ggplot2)
library(sqldf)
library(xgboost)
data <- read.csv("player.csv")
data <- na.omit(data)
data1 <- read.csv("player_shooting.csv")
data2 <- read.csv("player_passing_types.csv")

new_data <- sqldf("SELECT b.player, b.position, b.team, b.age , goals_assists_per90, cards_red, cards_yellow, sca, passes_completed, average_shot_distance
                  FROM data AS a, data1 AS b, data2 AS c
                  WHERE a.player = b.player AND a.player = c. player 
                  ")
new_data <- na.omit(new_data)

#data <-predict(preProcess(data, method = c("center", "scale")), data)  

ggplot(new_data, aes(x = cards_yellow, y = sca)) + geom_bar(stat = "identity")
cards_yellow_aov <- aov(cards_yellow ~ sca, data=new_data)
summary(cards_yellow_aov)  #p-value = <2e-16(<0.05), reject H0

ggplot(new_data, aes(x = cards_red, y = sca)) + geom_bar(stat = "identity")
cards_red_aov <- aov(cards_red ~ sca, data=new_data)
summary(cards_red_aov)  #p-value = <2e-16(<0.05), reject H0

summary(year_aov)  #p-value = <2e-16(<0.05), reject H0

qplot(Genre, Global_Sales, data = rm_data, geom = "boxplot",xlab = "Genre")
genre_aov <- aov(Global_Sales ~ Genre, data=rm_data)
summary(genre_sales_aov)  #p-value = <2e-16(<0.05), reject H0

qplot(Publisher, Global_Sales, data = rm_data, geom = "boxplot",xlab = "Publisher")
publisher_aov <- aov(Global_Sales ~ Publisher, data=rm_data)
summary(publisher_aov)  #p-value = <2e-16(<0.05), reject H0

cor.test(new_data$age, new_data$sca, method = "pearson") 
ggplot(new_data, aes(x = age, y = sca)) + geom_point() 

cor.test(new_data$passes_completed, new_data$sca, method = "pearson")
ggplot(new_data, aes(x = passes_completed, y = sca)) + geom_point() 

cor.test(new_data$passes_completed, new_data$sca, method = "pearson")
ggplot(new_data, aes(x = passes_completed, y = sca)) + geom_point() 

cor.test(new_data$average_shot_distance, new_data$sca, method = "pearson") 
ggplot(new_data, aes(x = average_shot_distance, y = sca)) + geom_point() 

cor.test(new_data$goals_assists_per90, new_data$sca, method = "pearson")  
ggplot(new_data, aes(x = goals_assists_per90, y = sca)) + geom_point() 

set.seed(1)
train_idx <- sample(1:nrow(data), nrow(data) * 0.7)
train <- data[train_idx,]
trainx <- as.matrix(train[,-1])
trainy <- train$sca
test <- data[setdiff(1:nrow(data), train_idx),]
testx <- as.matrix(test[,-1])
testy <- test$sca
model1 <- lm(sca ~ ., data = train)
summary(model1)
mod_abs_rank <- data.frame(sort(abs(model1$coefficients[-1]), decreasing = T))
mod_abs_rank

LOOCV_lm_rmse = function(f, d){
  errs = sapply(1:nrow(d), FUN = function(k){
    reponse_var = all.vars(f)[1]; # Name of the response variable
    m = lm(f, d[- k,], na.action = na.omit)
    return((d[[reponse_var]][k]) - predict(m, newdata = d[k,]))
  })
  return(round(sqrt(mean(errs ^ 2)),4))
}
LOOCV_lm_rmse(sca ~ goals_assists_per90, new_data)

Map(function(x) LOOCV_lm_rmse(x, data),
    c(sca ~ tackles_def_3rd, sca ~ passes_received, sca ~ touches_att_pen_area, sca ~ dribbles_completed, sca ~ tackles_mid_3rd, sca ~ progressive_passes_received ))
LOOCV_lm_rmse(sca ~ progressive_passes_received + touches_att_pen_area + dribbles_completed + passes_received, data )
LOOCV_lm_rmse(sca ~ ., data )

logmod <- glm(sca ~ ., data = train)
training_y_hat <- predict(logmod)
RMSE(train$sca, training_y_hat)
testing_y_hat <- predict(logmod,test)
RMSE(test$sca, testing_y_hat)

lmmod <- lm(sca ~ , data = train)
training1_y_hat <- predict(lmmod)
RMSE(train$sca, training1_y_hat)
testing1_y_hat <- predict(lmmod, test)
RMSE(test$sca, testing1_y_hat)

xgbmodel <- xgboost(data = trainx,
                    label = trainy,
                    nrounds = 531, eta = 0.1,
                    min_child_weight=3, max_depth=1,
                    subsample=1, colsample_bytree=0.9,
                    objective = "reg:squarederror", verbose = 0)

importance <- xgb.importance(model = xgbmodel)
par(mfrow = c(1,1))
xgb.plot.importance(importance, top_n = 10, measure = "Gain")


a = 'SELECT sca, average_shot_distance, assists, dispossessed, minutes_per_sub, minutes_pct, assisted_shots FROM data'
a = 'SELECT sca, average_shot_distance, assists, dispossessed FROM data'
data <- sqldf(a)

set.seed(1)
train_idx <- sample(1:nrow(data), nrow(data) * 0.7)
train <- data[train_idx,]
trainx <- as.matrix(train[,-1])
trainy <- train$sca
test <- data[setdiff(1:nrow(data), train_idx),]
testx <- as.matrix(test[,-1])
testy <- test$sca

hyper_grid <- expand.grid(eta = c( .05, .1),
                          max_depth = c( 5, 7),
                          min_child_weight = c(3, 5, 7),
                          subsample = c(.65, .8),
                          colsample_bytree = c(.8, .9, 1),
                          nrounds = 0,
                          RMSE = 0)
nrow(hyper_grid)
for(i in 1:nrow(hyper_grid)){
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth =hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )
  set.seed(123)
  xgb.tune <- xgb.cv(
    params = params,
    data = trainx,
    label = trainy,
    nrounds = 500,
    nfold = 5,
    objective = "reg:squarederror",
    verbose = 0,
    early_stopping_rounds = 10
  )
  
  hyper_grid$nrounds[i] <- which.min(xgb.tune$evaluation_log$test_rmse_mean)
  hyper_grid$RMSE[i] <- min(xgb.tune$evaluation_log$test_rmse_mean)
}
hyper_grid |> dplyr::arrange(RMSE) |> head(10)


set.seed(5)

xgbmodel <- xgboost(data = trainx,
                    label = trainy,
                    nrounds = 501, eta = 0.05,
                    min_child_weight=7, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8,
                    objctive = "reg:linear", verbose = 0)

xgbmodel <- xgboost(data = trainx,
                    label = trainy,
                    nrounds = 531,
                    objctive = "reg:linear")


future <- predict(xgbmodel, testx)
future <- as.data.frame(future)
final <- cbind(future, testy, testx)
final <- mutate(final, rmse=sqrt(mean((future-testy)^2)))
mean(final$rmse)
