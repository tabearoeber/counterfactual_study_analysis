library(lme4)
library(dplyr)
library(brms)
library(performance)

# hiddenFalse (Table 4 in supplementary material)
df <- read.csv('./processed-data/final/p2_hiddenFalse.csv', header=TRUE)

# hiddenTrue (Table 5 in supplementary material)
df <- read.csv('./processed-data/final/p2_hiddenTrue.csv', header=TRUE)

# set reference group
df$attributedUserExplanationType <- relevel(
  factor(df$attributedUserExplanationType),
  ref = "none"
)
df$gender <- relevel(
  factor(df$gender),
  ref = "female"
)
df$education <- relevel(
  factor(df$education),
  ref = "bachelors"
)
df$dailyAIWork <- relevel(
  factor(df$dailyAIWork),
  ref = "daily"
)
df$houseBuying <- relevel(
  factor(df$houseBuying),
  ref = "no"
)
df$attributedUserExplanationViewMode <- relevel(
  factor(df$attributedUserExplanationViewMode),
  ref = "table"
)


######## MODEL 1 ########
# mixed model
mixed_model_1 <- glmer(
  followAI ~ attributedUserExplanationType + (1 | userID),
  data = df,
  family = binomial(link = "logit")
)

print(summary(mixed_model_1),digits=4)
icc(mixed_model_1)


######## MODEL 2 ########
# mixed model
mixed_model_2 <- glmer(
  followAI ~ attributedUserExplanationType + gender + education + age + (1 | userID),
  data = df,
  family = binomial(link = "logit")
)

print(summary(mixed_model_2), digits=4)
icc(mixed_model_2)

# bayesian model
brms_model_2 <- brm(
  followAI ~ attributedUserExplanationType + gender + education + age + (1 | userID),
  data = df,
  family = bernoulli(link = "logit"),
  chains = 4,
  cores = 4,
  iter = 4000,
  control = list(adapt_delta = 0.95),
  seed=21
)


print(summary(brms_model_2), digits=3)
plot(brms_model_2)


## ------------- get bootstrap samples for plot in main paper using mixed_model_2 ---------------


# plot marginal means
library(emmeans)
summary(emmeans(
  mixed_model_2,
  ~ attributedUserExplanationType,
  type = "response",
  at = list(
    gender = "female",
    education = "bachelors",
    age = mean(df$age, na.rm = TRUE)
  )
))

# Function to return emmeans per bootstrap replicate
emm_fun <- function(fit) {
  est <- summary(emmeans(
    fit,
    ~ attributedUserExplanationType,
    type = "response",
    at = list(
      gender = "female",
      education = "bachelors",
      age = mean(df$age, na.rm = TRUE)
    )
  ))
  return(est$prob)
}



# Run bootstrap
set.seed(123)  # for reproducibility
boot_res <- bootMer(mixed_model_2, FUN = emm_fun, nsim = 100, use.u = FALSE, type = "parametric", parallel = "no", verbose=1)

# Filter out failed bootstraps (rows with any NA)
valid_boots <- boot_res$t[complete.cases(boot_res$t), ]

# Now bind into matrix
# boot_mat <- do.call(rbind, lapply(valid_boots, as.numeric))

# Convert to matrix
# boot_mat <- do.call(rbind, boot_res$t)

# Get means, SEs, CIs
boot_summary <- data.frame(
  attributedUserExplanationType = levels(df$attributedUserExplanationType),
  estimate = colMeans(valid_boots),
  sd = apply(valid_boots, 2, sd),
  se = apply(valid_boots, 2, function(x) sd(x) / sqrt(length(x))),
  lower = apply(valid_boots, 2, quantile, 0.025),
  upper = apply(valid_boots, 2, quantile, 0.975)
)


library(ggplot2)

ggplot(boot_summary, aes(x = attributedUserExplanationType, y = estimate)) +
  geom_bar(stat = "identity", fill = "lightskyblue", width = 0.6) +
  # geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_errorbar(aes(ymin = estimate - se, ymax = estimate + se),
                width = 0.2, color = "black") +
  labs(
    y = "Predicted Probability of Following AI",
    x = "Explanation Type"
  ) +
  theme_minimal(base_size = 14)

# save .csv file of boot_summary in p2-hiddenFalse-232-bootstrap/female-bachelor-100-samples.csv

######## MODEL 3 ########

# mixed model
mixed_model_3 <- glmer(
  followAI ~ attributedUserExplanationType + gender + education + age + practicalExperience + dailyAIWork + houseBuying + (1 | userID),
  data = df,
  family = binomial(link = "logit")
)

print(summary(mixed_model_3), digits=2)

# bayesian model
brms_model_3 <- brm(
  followAI ~ attributedUserExplanationType + gender + education + age + practicalExperience + dailyAIWork + houseBuying + (1 | userID),
  data = df,
  family = bernoulli(link = "logit"),
  chains = 4,
  cores = 4,
  iter = 4000,
  control = list(adapt_delta = 0.95),
  seed=21
)

print(summary(brms_model_3), digits=3)

######## MODEL 4 ########
# mixed model
mixed_model_4 <- glmer(
  followAI ~ attributedUserExplanationType + gender + education + age + practicalExperience + dailyAIWork + houseBuying + attributedUserExplanationViewMode + (1 | userID),
  data = df,
  family = binomial(link = "logit")
)

print(summary(mixed_model_4), digits=2)


# bayesian model
brms_model_4 <- brm(
  followAI ~ attributedUserExplanationType + gender + education + age + practicalExperience + dailyAIWork + houseBuying + attributedUserExplanationViewMode + (1 | userID),
  data = df,
  family = bernoulli(link = "logit"),
  chains = 4,
  cores = 4,
  iter = 4000,
  control = list(adapt_delta = 0.95),
  seed=21
)

print(summary(brms_model_4), digits=3)



######## MODEL 5 ########
# mixed model
mixed_model_5 <- glmer(
  followAI ~ attributedUserExplanationType + gender + education + age + practicalExperience + dailyAIWork + houseBuying + attributedUserExplanationViewMode + PublishingTime + (1 | userID),
  data = df,
  family = binomial(link = "logit")
)

print(summary(mixed_model_5), digits=2)


# bayesian model
brms_model_5 <- brm(
  followAI ~ attributedUserExplanationType + gender + education + age + practicalExperience + dailyAIWork + houseBuying + attributedUserExplanationViewMode + PublishingTime + (1 | userID),
  data = df,
  family = bernoulli(link = "logit"),
  chains = 4,
  cores = 4,
  iter = 4000,
  control = list(adapt_delta = 0.95),
  seed=21
)

print(summary(brms_model_5), digits=3)

