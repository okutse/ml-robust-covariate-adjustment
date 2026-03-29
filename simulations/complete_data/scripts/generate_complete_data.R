## install all required packages
ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg))
    install.packages(new.pkg, dependencies = TRUE, repos='http://cran.rstudio.com/')
  sapply(pkg, require, character.only = TRUE)
}
packages =c( "tidyverse","knitr", "kableExtra","skimr", "MatchIt", "RItools","optmatch", "ggplot2", "tufte", "tufterhandout", "plotly", "snowfall", "rstan", "gridExtra", "knitr", "gtsummary", "data.table", "GGally", "MASS", "broom", "boot", "foreach", "doParallel", "glmnet", "tidymodels" , "usemodels", "magrittr")
ipak(packages)

## set the working directory
##setwd("C:\\Users\\aokutse\\OneDrive - Brown University\\ThesisResults\\data")

## data generating function
set.seed(123)
rm(list = ls())

dgp <- function(n = NULL, ss = NULL)
{
  # treatment variable
  A <- c()
  for (i in 1:n){if (i <= n/2){ A[i] = 1} else {if (i > n/2){A[i] = 0}}}
  # create the variance-covariance matrix
  sigma <- matrix(0, nrow = 4, ncol = 4)
  diag(sigma) <- 1
  # create the vector of Zi's which generate the outcome yi
  mat <- MASS::mvrnorm(n = n, mu = c(0, 0, 0, 0), Sigma = sigma)
  colnames(mat) <- c("z1", "z2", "z3", "z4")
  # generate the error term
  e <- rnorm(n = n, mean = 0, sd = ss)
  # generate the outcome variable based on the specified model
  y <- 210 + 50*A + 27.4*mat[ ,1] + 13.7*mat[, 2] + 13.7*mat[, 3] + 13.7*mat[, 4] + e
  # generate the xi's actually observed by analyst
  x1 <- exp(mat[, 1]/2); x2 <- (mat[, 2]/ (1 + exp(mat[, 1]))) + 10; x3 <- (((mat[, 1]*mat[, 3])/25) + 0.6)^3; x4 <- (mat[,2] + mat[,4] + 20)^2
  # create the missing data variable based on the treatment
  pi <- locfit::expit(-mat[, 1]+0.5*mat[, 2]-0.25*mat[, 3]-0.1*mat[, 4])
  R = rbinom(n = n, size = 1, prob = pi)
  # save variables as a data frame
  df <-data.frame(y, A, x1, x2, x3, x4, R)
}


## generate and save the individual data frames under different scenarios
df_one <- dgp(n = 500, ss = 1)    ## n = 500, sd = 1
df_oneb <- dgp(n = 500, ss = 23)  ## n = 500, sd = 23
df_two <- dgp(n = 500, ss = 45)   ## n = 500, sd = 45
df_twob <- dgp(n = 500, ss = 68)  ## n = 500, sd = 67
df_three <- dgp(n = 2000, ss = 1) ## n = 2000, sd = 1
df_threeb <- dgp(n = 2000, ss = 23) ## n = 2000, sd = 23
df_four <- dgp(n = 2000, ss = 45) ## n = 2000, sd = 45
df_fourb <- dgp(n = 2000, ss = 68) ## n = 2000, sd = 68

## these data refer to case when n = 10000 and sd = 1 and 45 respectively (added to see pattern in efficiency for complete data)
df_five <- dgp(n = 10000, ss = 1) ## n = 1000, sd = 1
df_fiveb <- dgp(n = 10000, ss = 45) ## n = 10000, sd = 45

## save the individual data sets
save(df_one, file = "df_one.RData")
save(df_two, file = "df_two.RData")
save(df_three, file = "df_three.RData")
save(df_four, file = "df_four.RData")

## save the new batch of data sets
save(df_oneb, file = "df_oneb.RData")
save(df_twob, file = "df_twob.RData")
save(df_threeb, file = "df_threeb.RData")
save(df_fourb, file = "df_fourb.RData")

## save the additional data for efficiency computation at n = 10000
save(df_five, file = "df_five.RData")
save(df_fiveb, file = "df_six.RData")



## create the 1000 replications of each data set and save them
cores <- detectCores()
registerDoParallel(cores-1)
dsets1 = foreach(1:1000) %dopar% dgp(n = 500, ss = 1)
dsets2 = foreach(1:1000) %dopar% dgp(n = 500, ss = 45)
dsets3 = foreach(1:1000) %dopar% dgp(n = 2000, ss = 1)
dsets4 = foreach(1:1000) %dopar% dgp(n = 2000, ss = 45)
dsets5 = foreach(1:1000) %dopar% dgp(n = 10000, ss = 1)
dsets6 = foreach(1:1000) %dopar% dgp(n = 10000, ss = 45)

## save the list of 1000 data frames under different scenarios
save(dsets1, file = "dsets1.RData") ## n = 500, sd = 1
save(dsets2, file = "dsets2.RData") ## n = 500, sd = 45
save(dsets3, file = "dsets3.RData") ## n = 2000, sd = 1
save(dsets4, file = "dsets4.RData") ## n = 2000, sd = 45
save(dsets5, file = "dsets5.RData") ## n = 10000, sd = 1
save(dsets6, file = "dsets6.RData") ## n = 10000, sd = 45

#########################################
## create the new batch of data files for the case when sd = 23 and sd = 68, respectively.
dsets11 = foreach(1:1000) %dopar% dgp(n = 500, ss = 23)
dsets22 = foreach(1:1000) %dopar% dgp(n = 500, ss = 68)
dsets33 = foreach(1:1000) %dopar% dgp(n = 2000, ss = 23)
dsets44 = foreach(1:1000) %dopar% dgp(n = 2000, ss = 68)
## save the 1000 simulated data sets for each additional scenario
save(dsets11, file = "dsets11.RData") ## n = 500, sd = 23
save(dsets22, file = "dsets22.RData") ## n = 500, sd = 68
save(dsets33, file = "dsets33.RData") ## n = 2000, sd = 23
save(dsets44, file = "dsets44.RData") ## n = 2000, sd = 68


########################################
## create additional data files for additional simulations under full data analysis with n = 10, 000
## for each case under complete data, add a single case with n = 10, 000. dsets 5.



## end of data file
