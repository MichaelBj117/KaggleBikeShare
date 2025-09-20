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
  step_mutate(datetime_hour = factor(datetime_hour)) %>% 
  step_mutate(datetime_dow = factor(datetime_dow)) %>% 
  step_rm(c(datetime, atemp)) %>% 
  step_mutate(weather = factor(ifelse(weather == 4, 3, weather))) %>% 
  step_mutate(season = factor(season)) %>% 
  step_mutate(workingday = factor(workingday)) %>% 
  step_mutate(holiday = factor(holiday))

prepped_recipe <- prep(bike_recipe)
baked_data <- bake(prepped_recipe, new_data=train_data_clean)

tree_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # What R function to use
  set_mode("regression") 

bike_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(tree_mod)

grid_of_tuning_params <- grid_regular(tree_depth(),
                                      cost_complexity(),
                                      min_n(),
                                      levels = 5)

folds <- vfold_cv(train_data_clean, v=5, repeats = 1)

CV_results <- bike_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae))

collect_metrics(CV_results) %>% # Gathers metrics into DF8
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=tree_depth, y=mean, color=factor(cost_complexity))) +
  geom_line()

bestTune <- CV_results %>%
  select_best(metric="rmse")

final_wf <- bike_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data_clean)

lin_preds <- final_wf %>% 
  predict(new_data = test_data) %>% 
  mutate(.pred = exp(.pred))

kaggle_submission <- lin_preds %>%
  bind_cols(., test_data) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #right format to Kaggle

vroom_write(x=kaggle_submission, 
            file="GitHub/KaggleBikeShare/treetunedlvl5.csv", delim=",")
