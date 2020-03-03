#MovieLens Project
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(randomForest)
library(rpart)
library(ggplot2)
library(caret)
library(tidyverse)
library(data.table)
library(lubridate)
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#create test and train sets, test set is about equal in size to validation set
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.111, list = FALSE)
edx_train <- edx[-test_index,]
edx_test <- edx[test_index,]
edx_train <- edx_train %>% 
  semi_join(edx_test, by = "movieId") %>%
  semi_join(edx_test, by = "userId")

lambdas<-function(i,y,z){
  #overall mean rating
  omtemp<-mean(y$rating)
  
  #finding the regularized mean rating
  mtemp<-y%>%group_by_(z)%>%summarize(mean=omtemp+(sum(rating-omtemp)/(n()+i)))
  
  #adding value and sampling ten percent at random for bootstrapping
  temp_comp<-y%>%left_join(mtemp,by=z)
  length<-nrow(temp_comp)
  temp_comp_small<-temp_comp[sample(c(1:length), length/10, replace=FALSE),]

  #bootstrapping samples
  bstps<-createResample(1:nrow(temp_comp_small),times=10,list=TRUE)
  rsmes<-sapply(bstps,function(x){RMSE(temp_comp_small$mean[x],temp_comp_small$rating[x])})
  final<-c(i,mean(rsmes))
  final
}

#train set mean rating
om<-mean(edx_train$rating)

#train set user mean rating
cvls<-sapply(seq(0,18,0.5),function(x){lambdas(i=x,y=edx_train,z="userId")})
la<-cvls[1,which.min(cvls[2,])]#col 1 val with min of second column in cvls
um<-edx_train%>%group_by(userId)%>%summarize(usermean=om+(sum(rating-om)/(n()+la)))

#train set movie mean rating
cvlm<-sapply(seq(2,7,0.5),function(x){lambdas(i=x,y=edx_train,z="movieId")})
lc<-cvlm[1,which.min(cvlm[2,])]#col 1 val  with min of second column in cvlm
mm<-edx_train%>%group_by(movieId)%>%summarize(moviemean=om+(sum(rating-om)/(n()+lc)))

#train set genres average rating, based on previously regularized movie mean rating
cvlg<-sapply(seq(1,20,1),function(x){lambdas(i=x,y=edx_train,p=10,z="genres")})
lg<-cvlg[1,which.min(cvlg[2,])]#col 1 val  with min of second column in cvlsg
gm<-edx_train%>%group_by(genres)%>%summarize(genremean=om+(sum(rating-om)/(n()+lg)))

#list of movieIds and years
yearslist<-as.numeric(gsub("[^0-9]","",str_extract(edx_train$title,"\\([0-9]{4}\\)")))
trainwithyears<-mutate(edx_train, year=yearslist)

#train set regularized year mean rating
cvlsy<-sapply(seq(2,20,2),function(x){lambdas(i=x,y=trainwithyears,p=10,z="year")})
ly<-cvlsy[1,which.min(cvlsy[2,])]#col 1 val  with min of second column in cvlsy
ym<-trainwithyears%>%group_by(year)%>%summarize(yearmean=om+(sum(rating-om)/(n()+ly)))

#slight correlations exist (about 0.5 and -.6) for the month and year of ratings
#though only over a narrow range
dates<-edx_train%>%select(timestamp,rating)%>%mutate(date=as_datetime(timestamp),ratyear=year(date),ratmonth=month(date))

ydate<-dates%>%group_by(ratyear)%>%summarize(rateyearmean=mean(rating))
ydate[1,2]=om #replace earliest year, only 4 ratings, with overall mean

mdate<-dates%>%group_by(ratmonth)%>%summarize(ratemonthmean=mean(rating))

#prepare table
rtprep<-function(x){
  #separate year from name
  yl<-as.numeric(gsub("[^0-9]","",str_extract(x$title,"\\([0-9]{4}\\)")))
  x<-mutate(x, year=yl,date=as_datetime(timestamp),ratyear=year(date),ratmonth=month(date))
  
  #add regularized movie, user, and genre mean values, calculated on the train set only
  x<-x%>%left_join(um,by="userId")%>%
    left_join(mm,by="movieId")%>%
    left_join(gm,by="genres")%>%
    left_join(ym,by="year")%>%
    left_join(ydate,by="ratyear")%>%
    left_join(mdate,by="ratmonth")
  
  #fill overall mean rating in case where user, movie, or genre is unavailable
  replace_na(x$usermean,om)
  replace_na(x$moviemean,om)
  replace_na(x$genremean,om)
  replace_na(x$yearmean,om)
  replace_na(x$ratemonthmean,om)
  replace_na(x$rateyearmean,om)
  
  #simple prediction based on component columns
  x<-mutate(x,prediction=usermean+moviemean+genremean+yearmean+ratemonthmean+rateyearmean-(5*om))
  
  #separate genres and create a column for each with a boolean value
  genrelist<-sort(gsub("[^A-z]","",unique(unlist(strsplit(x$genres,"|",fixed=TRUE)))))
  for(i in 1:length(genrelist)) {
    x[[genrelist[i]]] <- with(x,grepl(genrelist[i],x$genres))
  }
  
  #clear any NAs and remove text columns,maybe text columns don't need to be excluded
  x<-na.omit(select(x,-title,-genres,-timestamp,-num.x,-num.y))
  
}

prep_train<-rtprep(edx_train)
prep_test<-rtprep(edx_test)
prep_val<-rtprep(validation)

#check simple prediction
RMSE(prep_train$prediction,prep_train$rating)
RMSE(prep_test$prediction,prep_test$rating)
RMSE(prep_val$prediction,prep_val$rating)

#random forest, final model
prep_train_finalmodel<-prep_train[sample(c(1:nrow(prep_train)), 100000, replace=FALSE),]
controlrffinal <- trainControl(method = "none")
mtryfinal<-data.frame(mtry=12)
train_rffinal <- train(rating ~ ., method="rf",
                  data = prep_train_finalmodel,
                  tuneGrid=mtryfinal,
                  trControl = controlrffinal)

#check on test set
predictions_rffinal<-predict(train_rffinal,prep_test)
testsetRMSE<-RMSE(predictions_rffinal,prep_test$rating)
testsetRMSE

#check on validation set
predictions_validation<-predict(train_rffinal,prep_val)
validationRMSE<-RMSE(predictions_validation,prep_val$rating)
validationRMSE

#this simple lm was better than my random forest
train_lm <- train(rating ~ usermean+moviemean+genremean+yearmean+rateyearmean+ratemonthmean, method="lm",
                  data = prep_train)
predictions_lm<-predict(train_lm,prep_test)
lmRMSE<-RMSE(predictions_lm,prep_test$rating)
lmRMSE

predictions_lmval<-predict(train_lm,prep_val)
lmRMSEval<-RMSE(predictions_lmval,prep_val$rating)
lmRMSEval

train_gam <- train(rating ~ usermean+moviemean+genremean+yearmean+rateyearmean+ratemonthmean, method="gamboost",
                  data = prep_train)
