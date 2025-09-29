library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(glmnet)
library(rpart)
library(agua)

test_data <- vroom("~/GitHub/KaggleBikeShare/test.csv")
train_data <- vroom("~/GitHub/KaggleBikeShare/train.csv")
train_data_clean <- train_data %>% 
  select(-registered, -casual) %>% 
  mutate(count = log(count))

bike_recipe <-recipe(count ~ ., data = train_data_clean) %>% 
  step_time(datetime, features="hour") %>%
  step_date(datetime, features= "dow") %>% 
  step_mutate(datetime_hour = factor(datetime_hour)) %>% 
  step_mutate(datetime_dow = factor(datetime_dow)) %>% 
  step_rm(datetime) %>% 
  step_mutate(weather = factor(ifelse(weather == 4, 3, weather))) %>% 
  step_dummy(all_nominal_predictors()) 

prepped_recipe <- prep(bike_recipe)
baked_data <- bake(prepped_recipe, new_data=train_data_clean)

Sys.setenv(JAVA_HOME="C:/Program Files/Eclipse Adoptium/jdk-25.0.0.36-hotspot")
h2o::h2o.init()

auto_mod <- auto_ml() %>%
  set_engine("h2o", max_runtime_secs= 1000, max_models=10) %>%
  set_mode("regression")

automl_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(auto_mod) %>% 
  fit(data=train_data_clean)

lin_preds <- automl_wf %>% 
  predict(new_data = test_data) %>% 
  mutate(.pred = exp(.pred))

kaggle_submission <- lin_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #right format to Kaggle

vroom_write(x=kaggle_submission, 
            file="GitHub/KaggleBikeShare/stacking10h2osub.csv", delim=",")

