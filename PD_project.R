#loading the packages
  install.packages("dplyr")
  install.packages("ggplot2")
  install.packages("caret")
  library(dplyr)
  library(ggplot2)
  library(caret)
  library(pROC)
  library(readr)
  
#loading data

  install.packages("data.table")
  library(data.table)
  
  df <- fread("accepted_2007_to_2018Q4.csv")
  
  
  
  #df <- read.csv("accepted_2007_to_2018Q4.csv", stringsAsFactors = FALSE)

#selecting relevant columns
  
  df <- df %>% select(loan_amnt, term, int_rate, annual_inc, dti, emp_length,
           home_ownership, purpose, loan_status)
  head(df)

#transforming data to useful numeric/factor form
  
  df <- df %>%
    filter(loan_status %in% c("Fully Paid", "Charged Off","Default")) %>%
    mutate(
      Default = ifelse(loan_status == "Charged Off", 1, ifelse(loan_status=="Default",1,0)),
      int_rate = as.numeric(gsub("%", "", int_rate)),
      term = ifelse(grepl("36", term), 36, 60),
      home_ownership = as.factor(home_ownership),
      purpose = as.factor(purpose)
    )
  
#dividing data into training and test data, selecting half rows randomly
  
  df<- na.omit(df)
  set.seed(7273)
  training.rows<- sample(1:nrow(df), 0.7*nrow(df))
  df_training <- df[training.rows,]
  df_test <- df[-training.rows,]  

#fitting the model
  
  model <- glm(Default ~ loan_amnt *term *int_rate - 
                 loan_amnt:term:int_rate - 
                 loan_amnt:int_rate + annual_inc + dti +
                 home_ownership + purpose,
               data = df_training, family = "binomial")
  summary(model)

  model2 <- glm(Default ~ loan_amnt *term *int_rate - 
                  loan_amnt:term:int_rate - 
                  loan_amnt:int_rate + annual_inc + dti +
                  purpose,
                data = df_training, family = "binomial")
  summary(model2)
  
  model3 <- glm(Default ~ loan_amnt + term + annual_inc + dti + emp_length +
                purpose ,
                data = df_training, family = "binomial")
  
  summary(model3)
  
#getting predictions
  
  df_training$predicted_pd <- predict(model3, type = "response" , newdata = df_training)
  df_test$predicted_pd <- predict(model3, type = "response" , newdata = df_test)

#creating predicted class (with 25% cutoff for default probability)
  
  df_test$pred_class <- ifelse(df_test$predicted_pd > 0.25, 1, 0)
  
#model testing
  
  confusionMatrix(as.factor(df_test$pred_class),
                  as.factor(df_test$Default))  
  
  roc_obj <- roc(df_test$Default,
                 df_test$predicted_pd)
  
  auc(roc_obj)
  
  plot(roc_obj,
       main = "ROC Curve",
       col = "blue")
  