library(nsga3)


for(f in datasets){
  
  df <- readRDS(file.path(data.folder, f))
  
  
  # Convert values in target column into binary
  
  levels(df$BAD)[levels(df$BAD) == "GOOD"] <- "0"
  levels(df$BAD)[levels(df$BAD) == "BAD"] <- "1"
  
  print(paste(f, "rows: ", nrow(df), "features: ", ncol(df)))
  
  
  # Tune model parameters
  params <- train_model(df)
  
  # Create Classifier
  xgb_learner <- makeLearner(
    "classif.xgboost",
    predict.type = "prob",
    par.vals = list(
      objective = "binary:logistic",
      eval_metric = "error",
      early_stopping_rounds = 100,
      nrounds = 10000,
      max_depth = params$max_depth,
      lambda = params$lambda,
      alpha = params$alpha,
      eta = params$eta,
      subsample = params$subsample,
      min_child_weight = params$min_child_weight,
      colsample_bytree = params$colsample_bytree
    )
  )
  
  #Cross Validation
  resampling <- makeResampleDesc("CV", iters = 5)
  
  #Objective functions
  obj_list <- c(mshare, emp) #get_spec) #list of objective functions
  obj_names <- c("mshare", "emp", "nf")#names of objective fns will be used as column names
  
  #specify pareto criteria
  pareto <- low(mshare)*low(emp)*low(nf)#*low(fcost) # high = maximize
  
  #Activate  parallelisation
  parallelStartSocket(24, show.info = FALSE)
  
  #start NSGA III
  ans <- nsga3fs(df = df, target = "BAD", obj_list, obj_names, pareto, 
             n = 100, max_gen = 100, 
             model = xgb_learner,
             resampling = resampling,
             num_features = TRUE,
             mutation_rate = 0.01)
  
  parallelStop()
  

  save(ans, file = paste0(f, ".nsga3.RData"))

}