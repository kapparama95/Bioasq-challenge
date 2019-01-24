rm(list=ls())
# Function that turns sparse matrix returned by the tm package into a more common to R Matrix 
# sparse matrix format
asSparseMatrix = function (stm) {
  sparseMatrix(i=stm$i,j=stm$j,x=stm$v,dims=c(stm$nrow, stm$ncol),dimnames=dimnames(stm))
}

# A function that turns a text vector into a document-term matrix
makeDTM <- function(tx, dict=NULL) {
  c = Corpus(VectorSource(tx))
  c <- tm_map(c, content_transformer(tolower))
  c <- tm_map(c, removePunctuation)
  c <- tm_map(c, stripWhitespace)
  c <- tm_map(c, removeNumbers)
  c <- tm_map(c, stemDocument, language = "english")
  
  
  if( is.null(dict) ) {
    ctl = list(weighting=weightBin)
  } else {
    ctl = list(dictionary=dict,weighting=weightBin)
  }
  res = DocumentTermMatrix(c, control=ctl)
  if( !is.null(dict) ) {
    res = res[,dict]
  }
  asSparseMatrix(res)
}

library(Matrix)
library(tm, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(SnowballC, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(Matrix, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(e1071, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(pROC, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(ROCR, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(rpart)


data = read.csv("Questions.CSV", sep=";")
ID = seq.int(nrow(data))
QUESTIONS = as.character(data$Question)
TYPE=data$Type
d=data.frame(ID,QUESTIONS,TYPE)
d$QUESTIONS=as.character(QUESTIONS)

summary(d)


dtm = makeDTM(d$QUESTIONS)
dim(dtm)

summ <- summary(dtm)
df <- data.frame(as.matrix(dtm), stringsAsFactors=FALSE)

require(caret)
flds <- createFolds(as.factor(dtm@Dimnames[["Docs"]]), k = 10, list = TRUE, returnTrain = FALSE)
rm(tab)
tab <- 0
for(i in 1:10){
  train.set=1:dim(data)[1]
  test.set=flds[[i]]
  train.set=train.set[-test.set]
  fit <- rpart(TYPE[train.set]~.,data=df[train.set,], control = rpart.control(minsplit=1))
  fit<- prune(fit, cp=   fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
  pred<-predict(fit, newdata=df[-train.set,], type="class")
  t<-prop.table(table(pred==d$TYPE[-train.set]))
  tab <-tab+t
}
tab/10

