library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(sqldf)
library(xgboost)
library(pheatmap)
library(reshape2)
library(data.table)
library(RANN)
library(ISLR)
library(rpart)
library(rpart.plot)
library(partykit)
library(varImp)

#load original data
player <- read.csv("player2.csv")

#data preprocessing
playerA <- preProcess(player, method = "knnImpute")
player <- predict(playerA, player)
head(model.matrix(sca ~ . -player, data = player))
dummies <- dummyVars(sca ~ . -player, data = player)
player <- predict(dummies, newdata = player)
player <- as.data.frame(player)
find <- findCorrelation(player, cutoff = 0.999)
player <- player[,find]
dec <- cor(player)
summary(dec[upper.tri(dec)])
player <- as.data.frame(player)

#write csv
write.csv(player, file = "C:\\Users\\staff\\Documents\\大三\\巨量資料分析導論\\player2.csv")

#load new data
player <- read.csv("player2.csv")
lm_original_model <- lm(sca~., data=player)
summary(lm_original_model)

#linear model
lm_model <- lm(sca~positionMF + tackles_def_3rd + tackles_mid_3rd + assisted_shots + 
                 on_goals_for + touches_att_pen_area + dribbles_completed +
                 passes_received + progressive_passes_received + shots_free_kicks +
                 positionDF + positionFW, data = player)
summary(lm_model)

lm_model_2 <- lm(sca~positionMF + tackles_def_3rd + tackles_mid_3rd + assisted_shots + 
                   touches_att_pen_area + dribbles_completed +
                   passes_received + progressive_passes_received + shots_free_kicks, data = encode_data)
summary(lm_model_2)

# make prediction
predict_train <- predict(lm_model_2, newdata = encode_data[train_idx,]);
predict_test <- predict(lm_model_2, newdata = encode_data[test_idx,]);

#rmse function
rmse = function(actual, predicted) {
  x = actual - predicted
  return (sqrt(mean(x^2)))
}

#training and testing RMSEs(round up to the fourth decimal digits)
rmse(encode_data[train_idx,"sca"], predict_train) 
rmse(encode_data[test_idx,"sca"], predict_test) 

#importance

lm_importance <-data.frame(
  names = c("positionMF", "tackles_def_3rd", "tackles_mid_3rd", "assisted_shots"
            , "touches_att_pen_area", "dribbles_completed"
            , "passes_received", "progressive_passes_received", "shots_free_kicks"),
  feature_importance = c(4.596e-02, 5.208e-02, 4.075e-02, 5.702e-01, 1.138e-01, 1.460e-01, 1.474e-01, 8.126e-02, 1.151e-01)
)
ggplot(data=lm_importance, aes(x=feature_importance, y=names)) +
  geom_bar(stat="identity")

#----------------------------------------------------------------------------------------------------------
#load real data
real_data <- read.csv("C:/Users/staff/Downloads/all_player.csv")

#select all important variables but assisted_shots
a = 'SELECT sca, position, tackles_def_3rd, tackles_mid_3rd, touches_att_pen_area, dribbles_completed, passes_received, progressive_passes_received, shots_free_kicks FROM real_data'
real_data <- sqldf(a)

# Split dataset into training and testing
library(ISLR)
real_data = na.omit(real_data)
numOfRows = nrow(real_data) 
set.seed(1) 
train_idx = sample(1:numOfRows,size = numOfRows * 0.7) # 70%  as training
test_idx = setdiff(1:numOfRows, train_idx) # 30% as testing

# Fully-grown regression Trees 
player_RT <- rpart(sca ~ ., data=real_data[train_idx,], control = rpart.control())
player_RT

# Plot regression tree
plot.party(as.party(player_RT))
printcp(player_RT)

# To get an CP table with "real" errors by multipling root node error
realCPTable = data.frame(printcp(player_RT))
Map(function(x) realCPTable[,x] <<- realCPTable[,x] * 14.529, c(1,3,4,5))
realCPTable

# Tree with 5 splits is better
player_RT_5split = prune(player_RT,cp = 0.04)
player_RT_5split
printcp(player_RT_5split)
plotcp(player_RT_5split)

# Plot the tree
plot.party(as.party(player_RT_5split))

#importance
argPlot <- prop.table(player_RT$variable.importance)
argPlot <- data.frame(argPlot)

rt_importance <-data.frame(
  names = c("progressive_passes_received", "touches_att_pen_area", "passes_received", "dribbles_completed"
            , "position", "tackles_mid_3rd"
            , "tackles_mid_3rd", "shots_free_kicks"),
  feature_importance = c(0.402315098,0.278455057, 0.134280087, 0.090091192, 0.033208564, 0.030027857, 0.024289021, 0.007333124)
)
ggplot(data=rt_importance, aes(x=feature_importance, y=names)) +
  geom_bar(stat="identity")

#rmse function
rmse = function(actual, predicted) {
  x = actual - predicted
  return (sqrt(mean(x^2)))
}

# make prediction
predict_train <- predict(player_RT_5split, newdata = real_data[train_idx,]);
predict_test <- predict(player_RT_5split, newdata = real_data[test_idx,]);

#training and testing RMSEs(round up to the fourth decimal digits)
rmse(real_data[train_idx,"sca"], predict_train) 
rmse(real_data[test_idx,"sca"], predict_test)    

#----------- Random Forest -----------#

#load encoded data
encode_data <- read.csv("C:\\Users\\staff\\Documents\\大三\\巨量資料分析導論\\player2.csv")

#select all important variables but assisted_shots
a = 'SELECT sca, position, tackles_def_3rd, tackles_mid_3rd, touches_att_pen_area, dribbles_completed, passes_received, progressive_passes_received, shots_free_kicks FROM encode_data'
encode_data <- sqldf(a)

# Split dataset into training and testing
library(ISLR); library(randomForest)
encode_data = na.omit(encode_data)
numOfRows = nrow(encode_data) 
set.seed(1); train_idx = sample(1:numOfRows,size = numOfRows * 0.7) # 70%  as training
test_idx = setdiff(1:numOfRows, train_idx) # 30% as testing

# Build a random forest with 500 trees (by default)
set.seed(1)
player_RF = randomForest(sca ~ ., encode_data[train_idx,],importance = TRUE, type=classification)
player_RF
p <- prop.table(randomForest::importance(player_RF))
data.frame(p)

player_RF$err.rate[,1]

ggplot(data=p, aes(x=feature_importance, y=names)) +
  geom_bar(stat="identity")

p <- data.frame(
  names = c("position", "tackles_def_3rd", "tackles_mid_3rd", "touches_att_pen_area"
            , "dribbles_completed", "passes_received"
            , "progressive_passes_received", "shots_free_kicks"),
  feature_importance = c(0.02561441,0.05146327, 0.04411799, 0.19437067, 0.14053382, 0.22674898, 0.26940015, 0.04775071)
)

x.data.frame
# Plot the "Errors vs. # of trees". Or just enter "plot(hitSalary_RF)" 
qplot(1:500, player_RF$mse, geom = "line") + labs(x = "number of trees", y = "MSE")

# Get RMSEs for 7 different models
train_player_RMSE = Map(function(f){ 
  set.seed(1)
  fit = f(sca ~ ., encode_data)
  pred_response = predict(fit, newdata = encode_data[train_idx,]);
  return(sqrt(mean((pred_response - encode_data[train_idx,"sca"])^2)));
},c("RF_50" = function(f, d) randomForest(formula = f, data = encode_data, ntree=50),
    "RF_150" = function(f, d) randomForest(formula = f, data = encode_data, ntree=150),
    "RF_200" = function(f, d) randomForest(formula = f, data = encode_data, ntree=200),
    "RF_300" = function(f, d) randomForest(formula = f, data = encode_data, ntree=300),
    "RF_350" = function(f, d) randomForest(formula = f, data = encode_data, ntree=350),
    "RF_400" = function(f, d) randomForest(formula = f, data = encode_data, ntree=400),
    "RT" = rpart, "LM" = lm))
train_player_RMSE

# Get RMSEs for 7 different models
test_player_RMSE = Map(function(f){ 
  set.seed(1)
  fit = f(sca ~ ., encode_data);
  pred_response = predict(fit, newdata = encode_data[test_idx,]);
  return(sqrt(mean((pred_response - encode_data[test_idx,"sca"])^2)));
},c("RF_50" = function(f, d) randomForest(formula = f, data = encode_data, ntree=50),
    "RF_80" = function(f, d) randomForest(formula = f, data = encode_data, ntree=80),
    "RF_100" = function(f, d) randomForest(formula = f, data = encode_data, ntree=100),
    "RF_200" = function(f, d) randomForest(formula = f, data = encode_data, ntree=200),
    "RF_300" = function(f, d) randomForest(formula = f, data = encode_data, ntree=300),
    "RF_350" = function(f, d) randomForest(formula = f, data = encode_data, ntree=350),
    "RF_400" = function(f, d) randomForest(formula = f, data = encode_data, ntree=400),
    "RT" = rpart,"LM" = lm))
test_player_RMSE


#xgboost
set.seed(1)
train_idx <- sample(1:nrow(player), nrow(player) * 0.7)
train <- player[train_idx,]
trainx <- train |> select(-sca) |> as.matrix()
trainy <- train$sca
test <- player[setdiff(1:nrow(player), train_idx),]
testx <- test |> select(-sca) |> as.matrix()
testy <- test$sca

xgbmodel <- xgboost(data = trainx[,-c(2)],
                    label = trainy,
                    nrounds = 531, eta = 0.1,
                    min_child_weight=3, max_depth=1,
                    subsample=1, colsample_bytree=0.9,
                    objctive = "reg:squarederror", verbose = 0)

importance <- xgb.importance(model = xgbmodel)
par(mfrow = c(1,1))
xgb.plot.importance(importance, top_n = 10, measure = "Gain")


a = 'SELECT sca, assisted_shots, tackles_att_3rd, shots, Progressive_passes, pass_xa, through_balls, miscontrols,  touches_att_pen_area FROM player'
player <- sqldf(a)

set.seed(1)
train_idx <- sample(1:nrow(player), nrow(player) * 0.7)
train <- player[train_idx,]
trainx <- as.matrix(train[,-1])
trainy <- train$sca
test <- player[setdiff(1:nrow(player), train_idx),]
testx <- as.matrix(test[,-1])
testy <- test$sca

hyper_grid <- expand.grid(eta = c(.01, .05, .1, .3),
                          max_depth = c(1, 3, 5, 7),
                          min_child_weight = c(1, 3, 5, 7),
                          subsample = c(.65, .8, 1),
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
                    nrounds = 38, eta = 0.1,
                    min_child_weight=5, max_depth=5,
                    subsample=0.65, colsample_bytree=0.8,
                    objctive = "reg:squarederror", verbose = 0)

future <- predict(xgbmodel, trainx)
future <- as.data.frame(future)
final <- cbind(future, trainy, trainx)
final <- mutate(final, rmse=sqrt(mean((future-trainy)^2)))
mean(final$rmse)


future <- predict(xgbmodel, testx)
future <- as.data.frame(future)
final <- cbind(future, testy, testx)
final <- mutate(final, rmse=sqrt(mean((future-testy)^2)))
mean(final$rmse)
