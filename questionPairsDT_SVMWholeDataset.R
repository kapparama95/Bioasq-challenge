rm(list = ls())

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

createDict <- function(tx){
  c = Corpus(VectorSource(tx))
  c <- tm_map(c, content_transformer(tolower))
  c <- tm_map(c, removePunctuation)
  c <- tm_map(c, stripWhitespace)
  c <- tm_map(c, removeNumbers)
  c <- tm_map(c, stemDocument, language = "english")
  
  ctl = list(weighting=weightBin)
  res = DocumentTermMatrix(c, control=ctl)
  findFreqTerms(res, 1)
}

library(Matrix)
library(tm, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(SnowballC, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(Matrix, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(e1071, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(pROC, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(ROCR, warn.conflicts = FALSE, quietly=TRUE, verbose=FALSE)
library(rpart)
library(rpart.plot)



data = read.csv("Questions.CSV", sep=";", encoding = "UTF8")
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

fit1 <- svm(x=dtm, y=d$TYPE, kernel="linear", scale=FALSE, cost=1, probability=FALSE)

fit <- rpart(TYPE~.,data=df, control = rpart.control(minsplit=1))
fit <- prune(fit, cp=   fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

#rpart.plot(fit, type = 3, clip.right.lab = FALSE, branch = .3, under = TRUE)

#table(classified=pred,actual=d$TYPE[-train.set])

#prop.table(table(pred==d$TYPE[-train.set]))

#act=d$TYPE[-train.set]

data2 = read.csv("quora_duplicate_questions.tsv", sep="\t", encoding = "UTF8")
data2$question1 = as.character(data2$question1)
data2$question2 = as.character(data2$question2)

dict <- createDict(d$QUESTIONS)
batch_size = 10000

for (ii in 0:40){

  ini = ii*batch_size+1
  end = ini + batch_size-1
  interval = ini:end
  
  dtm1 = makeDTM(data2$question1[interval], dict = dict)
  dtm2 = makeDTM(data2$question2[interval], dict = dict)
  df1 <- data.frame(as.matrix(dtm1), stringsAsFactors=FALSE)
  df2 <- data.frame(as.matrix(dtm2), stringsAsFactors=FALSE)

  pred11<-predict(fit, newdata=df1, type="class")
  pred21<-predict(fit, newdata=df2, type="class")

  pred12<-predict(fit1, newdata=df1, type="class")
  pred22<-predict(fit1, newdata=df2, type="class")

  docs <- c()
  labels <- c()

  for (i in 1:length(pred11)){
    if(pred11[i] == pred21[i] && pred11[i] == pred22[i] && pred11[i] == pred12[i]){
      docs <- c(docs,ini+i-1)
      labels <- c(labels,as.character(pred11[i]))
    }
  }

  questions = c(data2[docs,]$question1,data2[docs,]$question2)

  newDf = data.frame(Question = questions, Type = labels )
  write.table(newDf, file = paste0("newData/DT_SVMnewdata",ii,".tsv"), sep = "\t", row.names = FALSE)
}
