library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(glmnet)

test_data <- vroom("~/GitHub/KaggleBikeShare/test.csv")
train_data <- vroom("~/GitHub/KaggleBikeShare/train.csv")
train_data_clean <- train_data %>% 
  select(-registered, -casual) %>% 
  mutate(count = log(count))

bike_recipe <-recipe(count ~ ., data = train_data_clean) %>% 
  step_time(datetime, features="hour") %>% 
  step_rm(datetime) %>%
  step_mutate(weather3lvl = factor(ifelse(weather == 4, 3, weather))) %>% 
  step_mutate(season = factor(season)) %>% 
  step_zv(all_predictors()) %>% 
  step_poly(c(humidity, temp, atemp, windspeed), degree = 10) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data=train_data_clean)

preg_model <- linear_reg(penalty=0.01, mixture=0) %>% 
  set_engine("glmnet") 
#improved1 is penalty 0 mixture doesn't matter at that point
#improved2 is penalty 0.1, mixture 1
#improved3 is penalty 0.1, mixture 0
#improved4 is penalty 0.01, mixture 1
#improved5 is penalty 0.01, mixture 0

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model) %>%
  fit(data=train_data_clean)

lin_preds <- predict(bike_workflow, new_data = test_data) %>% 
  mutate(.pred = exp(.pred))

kaggle_submission <- lin_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #right format to Kaggle

vroom_write(x=kaggle_submission, 
            file="GitHub/KaggleBikeShare/LMpenpredimproved5.csv", delim=",")
