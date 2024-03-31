library(readr)
library(sqldf)
library(data.table)
library(caret)
library(RANN)

player <- read.csv("C:\\Users\\user\\Desktop\\R\\final\\player.csv")
player <- data.frame(player[rowSums(is.na(player)) == 0,])
#------ passed completed normaltest
# Lilliefors normality test
library(nortest)


nortest::lillie.test(player$passes_completed)

#summary
summary(player$passes_completed)

#plot
library(ggplot2)
hist(player$passes_completed,col= rgb(0,0.3,0,0.3), xlab = "passes_completed", main = "passes_completed", freq = F)
lines(density(player$passes_completed), col = "purple")
ggplot(player, aes(y = passes_completed)) + geom_boxplot() + scale_x_discrete() + ylab("density") + xlab("passes_completed")     


# ------sca normaltest
# Lilliefors normality test
nortest::lillie.test(player$sca)

#summary
summary(player$sca)

#plot
hist(player$sca, col = rgb(0,0.3,0,0.3), xlab = "sca", main = "sca", freq = F)
lines(density(player$sca), col = "purple")
ggplot(player, aes(y = sca)) + geom_boxplot() + scale_x_discrete() + ylab("density") + xlab("sca")

#-------average shot distance normal test
# Lilliefors normality test
nortest::lillie.test(player$average_shot_distance)

#summary
summary(player$average_shot_distance)

#plot
hist(player$average_shot_distance, col =rgb(0,0.3,0,0.3), xlab = "density", main = "average_shot_distance", freq = F)
lines(density(player$average_shot_distance), col = "purple")
ggplot(player, aes(y = average_shot_distance)) + geom_boxplot() + scale_x_discrete() + ylab("density") + xlab("averge_shot_distance")
####################
player33<-read.csv("all_player.csv")

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
##########################
new_data <- read.csv('player2.csv')

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
