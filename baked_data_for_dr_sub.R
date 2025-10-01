library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(glmnet)
library(rpart)

test_data <- vroom("~/GitHub/KaggleBikeShare/test.csv")
train_data <- vroom("~/GitHub/KaggleBikeShare/train.csv")
train_data_clean <- train_data %>% 
  select(-registered, -casual) %>% 
  mutate(count = log(count))

bike_recipe <-recipe(count ~ ., data = train_data_clean) %>% 
  step_time(datetime, features="hour") %>%
  step_date(datetime, features= "dow") %>% 
  step_mutate(hour_of_week = (((as.numeric(datetime_dow) + 6) %% 7) * 24 + 
                                as.numeric(datetime_hour))) %>% 
  step_rm(c(datetime, datetime_hour, datetime_dow)) %>% 
  step_mutate(weather = factor(ifelse(weather == 4, 3, weather))) %>% 
  step_mutate(season = factor(season)) %>% 
  step_mutate(workingday = factor(workingday)) %>% 
  step_mutate(holiday = factor(holiday))

  
prepped_recipe <- prep(bike_recipe)
train_baked_data <- bake(prepped_recipe, new_data=train_data_clean)
test_baked_data <- bake(prepped_recipe, new_data=test_data)

vroom_write(x=baked_data, 
            file="GitHub/KaggleBikeShare/baked_data_for_dr.csv", delim=",")
vroom_write(x=test_baked_data,
            file="GitHub/KaggleBikeShare/test_bd_for_dr.csv", delim=",")
pred <- vroom("C:/Users/mikey/Downloads/result-68dcadc9a86998b8bf0bb651.csv")

pred <- pred %>% 
  mutate(count_PREDICTION = exp(count_PREDICTION))  

kaggle_submission <- pred %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, count_PREDICTION) %>% #Just keep datetime and prediction variables
  rename(count=count_PREDICTION) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #right format to Kaggle  

vroom_write(x=kaggle_submission, 
            file="GitHub/KaggleBikeShare/data_robotBTRkagglesub2.csv", delim=",")
