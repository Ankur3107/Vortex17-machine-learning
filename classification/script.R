train_data <- read.csv("Dataset-1/competition_first_train.csv")
test_data <- read.csv("Dataset-1/competition_first_test.csv")
sample_submission <- read.csv("Dataset-1/competition_first_sample.csv")

str(train_data)
summary(train_data)

test_data$predict <- NA
all_data <- rbind(train_data,test_data)
summary(all_data)

all_data$f[is.na(all_data$f)] <- 28

m_train_data <- all_data[1:65000,]
m_test_data <- all_data[65001:88758,]

feature.names <- names(m_train_data)
feature.names <- feature.names[feature.names!= "ID" & feature.names!="predict"]
                                 

set.seed(1960)
h<-sample(nrow(m_train_data),floor(0.3*nrow(m_train_data)))
train_sample <- m_train_data[-h,]
train_val <- m_train_data[h,]



dval<-xgb.DMatrix(data=data.matrix(train_val[,feature.names]),label=as.numeric(train_val$predict))
dtrain<-xgb.DMatrix(data=data.matrix(train_sample[,feature.names]),label=as.numeric(train_val$predict))
watchlist<-list(val=dval,train=dtrain)

xg.test <- m_test_data[,feature.names]

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eta                 = 0.48,
                max_depth           = 4, #7
                subsample           = 0.9,
                colsample_bytree    = 0.9
)
set.seed(1429)
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 2000, 
                    verbose             = 1,
                    early.stop.round    = 100,
                    watchlist           = watchlist,
                    maximize            = TRUE
)

pred_noexp=predict(clf,data.matrix(m_test_data[,feature.names]))


solutionXgBoost<- data.frame(Id = m_test_data$V1, Prediction = pred_noexp)

write.csv(solutionXgBoost, file = 'solutionXgBoost.csv', row.names = F)

#######
x_train<-xgb.DMatrix(data=data.matrix(m_train_data[,feature.names]))
x_text <- xgb.DMatrix(data=data.matrix(m_test_data[,feature.names]))


bstDense <- xgboost(data = (x_train), label = as.numeric(target), max.depth = 4,early.stop.round= 100,subsample= 0.9,colsample_bytree= 0.9,eta = 0.48, nthread = 2, nround = 2000, objective = "binary:logistic")



#########
