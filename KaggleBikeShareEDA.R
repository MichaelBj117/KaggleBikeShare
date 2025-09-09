#Loading the data
library(vroom)
library(patchwork)
Train_data <- vroom("GitHub/KaggleBikeShare/train.csv")
Train_data$season <- as.factor(Train_data$season)
Train_data$holiday <- as.factor(Train_data$holiday)
Train_data$workingday <- as.factor(Train_data$workingday)
Train_data$weather <- as.factor(Train_data$weather)
wexwi <- ggplot(data = Train_data, mapping = aes(x = weather, y = windspeed)) +
  geom_boxplot()  #Shows how windspeed stays about the same throughout weather.
weatherbar <- ggplot(data = Train_data, mapping = aes(x = weather)) +
  geom_bar()  #barplot of weather as a factor. Looks exponential.
atempxtemp <- ggplot(data = Train_data, mapping = aes(x = temp, y = atemp)) +
  geom_point()  #this shows the colinearity of atmep with temp.
tempxcount <- ggplot(data = Train_data, mapping = aes(x = temp, y = count)) +
  geom_point()+
  geom_smooth(se = FALSE)  
    #in general it seems the temperature is positively correlated with bike use
EDA <- (wexwi + weatherbar)/(atempxtemp + tempxcount) 
