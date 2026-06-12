# ============================================================
# CREDIT RISK MODEL — PROBABILITY OF DEFAULT (PD)
# LendingClub Loan Data 2007–2018
# ============================================================
# This model estimates the probability that a borrower will
# default on their loan, using logistic regression.
# Expected Loss = PD x LGD x EAD  (Basel III framework)
# ============================================================



# ============================================================
# SECTION 1 — LOAD PACKAGES
# ============================================================

install.packages("dplyr")
install.packages("ggplot2")
install.packages("caret")
install.packages("pROC")
install.packages("data.table")

library(data.table)   # fast CSV reading
library(dplyr)        # data manipulation
library(ggplot2)      # plots
library(caret)        # confusion matrix
library(pROC)         # ROC curve and AUC



# ============================================================
# SECTION 2 — LOAD AND SELECT DATA
# ============================================================

df <- fread("accepted_2007_to_2018Q4.csv")

# Keep only the columns relevant to our model
df <- df %>% select(loan_amnt, term, int_rate, annual_inc,
                         dti, emp_length, home_ownership, purpose,
                         revol_util, open_acc, delinq_2yrs,
                         pub_rec, inq_last_6mths,
                         loan_status)



# ============================================================
# SECTION 3 — CLEAN THE DATA
# ============================================================

# Keep only loans that have a clear outcome (remove "Current" etc.)
df <- df %>% filter(loan_status %in% c("Fully Paid", "Charged Off", "Default"))

# Create binary target variable: 1 = defaulted, 0 = paid back
df$Default <- ifelse(df$loan_status %in% c("Charged Off", "Default"), 1, 0)

# Clean interest rate — remove the "%" sign and convert to number
df$int_rate <- as.numeric(gsub("%", "", df$int_rate))

# Clean term — convert "36 months" / "60 months" to numbers 36 / 60
df$term <- ifelse(grepl("36", df$term), 36, 60)

# Clean employment length — convert text to numbers
# "< 1 year" becomes 0, "10+ years" becomes 10, rest extracted as number
df$emp_length <- gsub("\\+ years| years| year|< ", "", df$emp_length)
df$emp_length <- ifelse(df$emp_length == "n/a", NA, df$emp_length)
df$emp_length <- as.numeric(df$emp_length)

# Create log income variable
# annual_inc is heavily right-skewed — a small number of very high earners
# pull the scale. Log transformation compresses this and makes it
# behave better in a linear model.
# We add 1 before taking log to avoid log(0) which is undefined
df$log_annual_inc <- log(df$annual_inc + 1)

# Convert categorical columns to factors (needed for regression)
df$home_ownership <- as.factor(df$home_ownership)
df$purpose        <- as.factor(df$purpose)

# Remove rows with any missing values
df <- na.omit(df)



# ============================================================
# SECTION 4 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Overall default rate in the dataset
cat("Overall Default Rate:", round(mean(df$Default) * 100, 2), "%\n")

# Default rate by loan purpose
default_by_purpose <- df %>%
  group_by(purpose) %>%
  summarise(default_rate = round(mean(Default) * 100, 2),
            count = n()) %>%
  arrange(desc(default_rate))

print(default_by_purpose)

# Default rate by term
default_by_term <- df %>%
  group_by(term) %>%
  summarise(default_rate = round(mean(Default) * 100, 2),
            count = n())

print(default_by_term)

# Plot: Interest rate distribution by default status
ggplot(df, aes(x = int_rate, fill = factor(Default))) +
  geom_histogram(bins = 40, alpha = 0.6, position = "identity") +
  scale_fill_manual(values = c("steelblue", "red"),
                    labels = c("Fully Paid", "Defaulted")) +
  labs(title = "Interest Rate Distribution by Default Status",
       x = "Interest Rate (%)", y = "Count", fill = "Loan Status")
ggsave("interest_rate_distribution.png")

# Plot: Default rate by purpose
ggplot(default_by_purpose, aes(x = reorder(purpose, default_rate),
                               y = default_rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Default Rate by Loan Purpose",
       x = "Purpose", y = "Default Rate (%)")
ggsave("default_rate_by_purpose.png")


# ============================================================
# SECTION 5 — SPLIT INTO TRAINING AND TEST DATA
# ============================================================

set.seed(7273)  # for reproducibility

training_rows <- sample(1:nrow(df), size = 0.7 * nrow(df))

df_train <- df[training_rows, ]
df_test  <- df[-training_rows, ]

cat("Training rows:", nrow(df_train), "\n")
cat("Test rows:    ", nrow(df_test),  "\n")



# ============================================================
# SECTION 5B — BALANCE THE TRAINING DATA (UNDERSAMPLING)
# ============================================================
# The dataset is imbalanced — about 80% fully paid, 20% defaulted
# This causes the model to be biased toward predicting "fully paid"
# because that is statistically the safer guess
#
# Fix: Undersample the majority class (fully paid) so both classes
# are equal in size in the training data
#
# We chose undersampling over oversampling because our dataset is
# large enough (880k rows) that we can afford to discard some rows
# without losing model quality
#
# IMPORTANT: We only balance the TRAINING data
# The test data is kept as-is to reflect real world conditions
# ============================================================

# Separate the two classes
df_default    <- df_train[df_train$Default == 1, ]
df_fully_paid <- df_train[df_train$Default == 0, ]

cat("Defaulters in training data:   ", nrow(df_default), "\n")
cat("Fully paid in training data:   ", nrow(df_fully_paid), "\n")

# Randomly keep only as many fully paid rows as there are defaulters
set.seed(42)
df_fully_paid_reduced <- df_fully_paid[sample(1:nrow(df_fully_paid),
                                              nrow(df_default)), ]

# Combine back into one balanced training dataset
df_train_balanced <- rbind(df_default, df_fully_paid_reduced)

# Confirm the balance
cat("Balanced training set — Default 0:", sum(df_train_balanced$Default == 0), "\n")
cat("Balanced training set — Default 1:", sum(df_train_balanced$Default == 1), "\n")



# ============================================================
# SECTION 6 — FIT THE LOGISTIC REGRESSION MODEL
# ============================================================
# We use logistic regression because:
# 1. Output is naturally a probability between 0 and 1
# 2. It is interpretable — we can explain each coefficient
# 3. It is the industry standard for PD modelling in banks
# ============================================================

model <- glm(Default ~ loan_amnt + term + int_rate +
               log_annual_inc + dti + emp_length + purpose +
               revol_util + open_acc + delinq_2yrs +
               pub_rec + inq_last_6mths,
             data   = df_train_balanced,
             family = "binomial")

summary(model)

# How to read the output:
# Estimate > 0 means the variable increases default probability
# Estimate < 0 means the variable decreases default probability
# Pr(>|z|) < 0.05 means the variable is statistically significant



# ============================================================
# SECTION 7 — GENERATE PREDICTIONS
# ============================================================

# Predicted probability of default for each loan
df_train$predicted_pd <- predict(model, type = "response", newdata = df_train)
df_test$predicted_pd  <- predict(model, type = "response", newdata = df_test)

# Summary of predicted probabilities
summary(df_test$predicted_pd)



# ============================================================
# SECTION 8 — CONVERT PROBABILITY TO CIBIL SCORE
# ============================================================
# CIBIL is India's credit scoring system (range: 300 to 900)
# Higher score = lower risk = less likely to default
# The formula rescales the log-odds (linear predictor from GLM)
# into the 300-900 range
#
# Formula: CIBIL Score = 750 - 80 * log(PD / (1 - PD))
#
# Why 750 and 80?
# - 750 is the midpoint anchor (average risk borrower gets ~750)
# - 80 controls how spread out the scores are across 300-900
# - The minus sign flips direction: higher PD = lower score
#
# CIBIL Score Bands:
# 750 - 900 : Excellent — loan approved, best interest rates
# 700 - 749 : Good      — loan likely approved
# 650 - 699 : Fair      — approved with conditions
# 600 - 649 : Poor      — likely rejected
# 300 - 599 : Very Poor — rejected
# ============================================================

df_test$cibil_score <- 750 - 80 * log(df_test$predicted_pd /
                                        (1 - df_test$predicted_pd))

# Cap scores within the valid CIBIL range of 300 to 900
df_test$cibil_score <- pmin(pmax(df_test$cibil_score, 300), 900)

# Summary of CIBIL scores
cat("CIBIL Score Summary:\n")
print(round(summary(df_test$cibil_score), 0))

# Plot: CIBIL score distribution by default status
ggplot(df_test, aes(x = cibil_score, fill = factor(Default))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("steelblue", "red"),
                    labels = c("Fully Paid", "Defaulted")) +
  labs(title = "CIBIL Score Distribution by Default Status",
       x = "CIBIL Score (300-900)", y = "Density", fill = "Loan Status")
ggsave("cibil_score_distribution.png")


# ============================================================
# SECTION 9 — MODEL VALIDATION
# ============================================================

# --- 9a. Confusion Matrix ---
# Classify as default if predicted probability > 55%
df_test$pred_class <- ifelse(df_test$predicted_pd > 0.55, 1, 0)

confusionMatrix(as.factor(df_test$pred_class),
                as.factor(df_test$Default))

# --- 9b. ROC Curve and AUC ---
roc_obj <- roc(df_test$Default, df_test$predicted_pd)

plot(roc_obj,
     col  = "blue",
     main = "ROC Curve — Logistic Regression PD Model")
ggsave("roc_curve.png")     

cat("AUC:", round(auc(roc_obj), 4), "\n")

# AUC interpretation:
# 0.5 = no better than random guessing
# 0.7 = acceptable
# 0.8 = good
# 0.9 = excellent

# --- 9c. Gini Coefficient ---
# Standard metric used in credit risk (range: 0 to 1, higher is better)
gini <- 2 * auc(roc_obj) - 1
cat("Gini Coefficient:", round(gini, 4), "\n")

# --- 9d. THRESHOLD ANALYSIS ---
# The threshold decides at what predicted PD we classify
# a loan as "likely to default"
# A lower threshold = stricter = catches more defaulters
# but also wrongly flags more good borrowers
#
# Sensitivity = out of all actual defaulters, how many did we catch?
# Specificity = out of all good borrowers, how many did we correctly approve?
# These two always trade off against each other — this is the same
# story the ROC curve tells, just shown as a table
# ============================================================

thresholds <- c(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)

threshold_results <- data.frame()

for (t in thresholds) {
  
  pred_class <- ifelse(df_test$predicted_pd > t, 1, 0)
  
  cm <- confusionMatrix(as.factor(pred_class),
                        as.factor(df_test$Default))
  
  threshold_results <- rbind(threshold_results, data.frame(
    Threshold   = t,
    Accuracy    = round(cm$overall["Accuracy"], 4),
    Sensitivity = round(cm$byClass["Sensitivity"], 4),
    Specificity = round(cm$byClass["Specificity"], 4)
  ))
}

print(threshold_results)

# Plot: Distribution of predicted probabilities by default status
ggplot(df_test, aes(x = predicted_pd, fill = factor(Default))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("steelblue", "red"),
                    labels = c("Fully Paid", "Defaulted")) +
  geom_vline(xintercept = 0.55, linetype = "dashed",
             color = "black", linewidth = 0.8) +
  annotate("text", x = 0.52, y = 2,
           label = "Threshold = 0.55", hjust = 0) +
  labs(title = "Predicted Probability of Default Distribution",
       x = "Predicted PD", y = "Density", fill = "Loan Status")
ggsave("predicted_pd_distribution.png")

# ============================================================
# SECTION 10 — EXPECTED LOSS CALCULATION
# ============================================================
# This is the actuarial section — connecting PD to actual £/$ loss
# Expected Loss (EL) = PD x LGD x EAD
# LGD assumed at 45% (industry standard for unsecured personal loans)
# EAD = loan_amnt (full outstanding balance)
# ============================================================

LGD <- 0.45  # Loss Given Default assumption

df_test$expected_loss <- df_test$predicted_pd * LGD * df_test$loan_amnt

# Total expected loss across the test portfolio
total_EL <- sum(df_test$expected_loss)
total_exposure <- sum(df_test$loan_amnt)

cat("Total Portfolio Exposure:  $", round(total_exposure, 0), "\n")
cat("Total Expected Loss:       $", round(total_EL, 0), "\n")
cat("EL as % of Portfolio:      ",  round(total_EL / total_exposure * 100, 2), "%\n")

# Plot: Distribution of expected loss per loan
ggplot(df_test, aes(x = expected_loss)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Expected Loss per Loan",
       x = "Expected Loss ($)", y = "Count")
ggsave("expected_loss_distribution.png")


# Actual loss based on real outcomes
df_test$actual_loss <- df_test$Default * LGD * df_test$loan_amnt

total_actual_loss <- sum(df_test$actual_loss)

cat("Total Actual Loss:         $", round(total_actual_loss, 0), "\n")
cat("Total Expected Loss:       $", round(total_EL, 0), "\n")
cat("Difference:                $", round(total_EL - total_actual_loss, 0), "\n")
cat("Ratio (EL / Actual):       ",  round(total_EL / total_actual_loss, 4), "\n")