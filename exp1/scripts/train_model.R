

train_model <- function(df, target = "BAD"){
  
  set.seed(1991)
  
  df <- df %>% createDummyFeatures(target = target, method = 'reference')
  
  
  ndf <- normalizeFeatures(df, target = target)
  
  # Define machine learning task
  ml_task <- makeClassifTask(data = ndf, target = target, positive=1)
  
  # Create repeated cross validation folds
  cv_folds <- makeResampleDesc("CV", iters = 5)
  
  # Define model
  model <- makeLearner( "classif.xgboost", predict.type = "prob")
  
  #
  random_tune <- makeTuneControlRandom(maxit = 10000L)
  
  # Define parameters of model and search grid ~ !!!! MODEL SPECIFIC !!!!
  model_Params <- makeParamSet(
    makeIntegerParam("max_depth",lower=1,upper=8),
    makeNumericParam("lambda",lower=0.001,upper=0.30),
    makeNumericParam("eta", lower = 0.001, upper = 0.3),
    makeNumericParam("alpha", lower = 0.001, upper = 0.3),
    makeNumericParam("subsample", lower = 0.70, upper = 1.0),
    makeNumericParam("min_child_weight",lower=0,upper=5),
    makeNumericParam("colsample_bytree",lower = 0.7,upper = 1.0)
  )
  
  parallelStartSocket(2)
  
  # Tune model to find best performing parameter settings using random search algorithm
  tuned_model <- tuneParams(learner = model,
                            task = ml_task,
                            resampling = cv_folds,
                            measures = auc,       # R-Squared performance measure, this can be changed to one or many
                            par.set = model_Params,
                            control = random_tune,
                            show.info = FALSE)
  parallelStop()
  
  
  tuned_model
  
  params <- tuned_model$x
  
  return(params)
}