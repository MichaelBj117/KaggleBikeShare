testData <- vroom("GitHub/KaggleBikeShare/test.csv")
Train_data <- Train_data %>%
  select(-casual, -registered)

my_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula = count ~ ., data = Train_data)

bike_predictions <- predict(my_linear_model,
                            new_data=testData)
kaggle_submission <- bike_predictions %>%
bind_cols(., testData) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
vroom_write(x=kaggle_submission, 
            file="GitHub/KaggleBikeShare/LMpred.csv", delim=",")
