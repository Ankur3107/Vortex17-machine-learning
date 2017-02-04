
train_data <- read.csv("Dataset-2/competition_second_train.csv",header = FALSE)
test_data <- read.csv("Dataset-2/competition_second_test.csv",header = FALSE)
sample_submission <- read.csv("Dataset-2/competition_second_sample.csv")

str(train_data)
str(test_data)

names(train_data)
names(test_data)
names(sample_submission)

summary(train_data)
head(test_data)


test_data$V76 <- NA
all_data <- rbind(train_data,test_data)

###### NA imputation #########

# First function
naCol <- function(train){
  return(colnames(train)[colSums(is.na(train)) > 0])
}

all_na_col <- naCol(all_data)

# Second function

missingTypeVariable <- function(df,nadf,n=18){
  
  intType <- c()
  factorType <- c()
  for(i in 1:18)
  {
    if(class(df[,nadf[i]])=="integer")
      intType <- c(intType,nadf[i])
    else
      factorType <- c(factorType,nadf[i])
  }
  
  
  
  return (list(intType=intType,factorType=factorType))
  
}

all_NA_Missing_Type <- missingTypeVariable(all_data,all_na_col)

all_NA_int_type <- unlist(all_NA_Missing_Type[1])
all_NA_factor_type <- unlist(all_NA_Missing_Type[2])

#integer type correlation with target

cor(train_data[,all_NA_int_type[1:3]],train_data$V76,use="pairwise.complete.obs")

#factor type correlation with target

library(ggplot2)
ggplot(train_data,aes(train_data$V76,train_data[,all_NA_factor_type[2]])) + geom_boxplot()


#imputing int type variable

all_data$V23[is.na(all_data[23])] <- 0
all_data$V55[is.na(all_data$V55)] <- 1980

qplot(all_data[4])
all_data$V4[is.na(all_data$V4)] <- 70
all_data$V4 <- ifelse(all_data$V4>150,70,all_data$V4)

#imputing factor type variable

summary(all_data[all_NA_factor_type])


#mice work
library(mice)
Dat1 <- subset(all_data, select=c(V7,V27,V28,V29,V30,
                                  V32,V39,V54,V56,V59,
                                  V60,V68,V69,V70))
imp <- mice(Dat1, m=3, maxit=10)

all_data$V7[is.na(all_data$V7)] <- imp$imp$V7$`3`
all_data$V27[is.na(all_data$V27)] <- imp$imp$V27$`3`
all_data$V28[is.na(all_data$V28)] <- imp$imp$V28$`3`
all_data$V29[is.na(all_data$V29)] <- imp$imp$V29$`3`
all_data$V30[is.na(all_data$V30)] <- imp$imp$V30$`3`
all_data$V32[is.na(all_data$V32)] <- imp$imp$V32$`3`
all_data$V39[is.na(all_data$V39)] <- imp$imp$V39$`3`
all_data$V54[is.na(all_data$V54)] <- imp$imp$V54$`3`
all_data$V56[is.na(all_data$V56)] <- imp$imp$V56$`3`
all_data$V59[is.na(all_data$V59)] <- imp$imp$V59$`3`
all_data$V60[is.na(all_data$V60)] <- imp$imp$V60$`3`
all_data$V68[is.na(all_data$V68)] <- imp$imp$V68$`3`
all_data$V69[is.na(all_data$V69)] <- imp$imp$V69$`3`
all_data$V70[is.na(all_data$V70)] <- imp$imp$V70$`3`

####### modelling ######

m_train_data <- all_data[1:1050,]
m_test_data <- all_data[1051:1460,]

m_test_data$V76 <- NULL

feature.names <- names(m_train_data)
feature.names <- feature.names[feature.names!= "V1" & feature.names!="V76"]

library(xgboost)





set.seed(1960)
h<-sample(nrow(m_train_data),floor(0.3*nrow(m_train_data)))
train_sample <- m_train_data[-h,]
train_val <- m_train_data[h,]



dval<-xgb.DMatrix(data=data.matrix(train_val[,feature.names]),label=train_val[,76])
dtrain<-xgb.DMatrix(data=data.matrix(train_sample[,feature.names]),label=train_sample[,76])
watchlist<-list(val=dval,train=dtrain)

xg.test <- m_test_data[,feature.names]

param <- list(  objective           = "reg:linear", 
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
