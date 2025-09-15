library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)

test_data <- vroom("~/GitHub/KaggleBikeShare/test.csv")
train_data <- vroom("~/GitHub/KaggleBikeShare/train.csv")
train_data_clean <- train_data %>% 
  select(-registered, -casual) %>% 
  mutate(count = log(count))

bike_recipe <-recipe(count ~ ., data = train_data_clean) %>% 
  step_time(datetime, features="hour") %>% 
  step_mutate(weather3lvl = factor(ifelse(weather == 4, 3, weather))) %>% 
  step_mutate(season = factor(season)) %>% 
  step_zv(all_predictors())
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data=train_data_clean)

lin_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model) %>%
  fit(data=train_data_clean)

lin_preds <- predict(bike_workflow, new_data = test_data) %>% 
  mutate(.pred = exp(.pred))

kaggle_submission_2 <- lin_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #right format to Kaggle

vroom_write(x=kaggle_submission_2, 
            file="GitHub/KaggleBikeShare/LMpredimproved.csv", delim=",")
