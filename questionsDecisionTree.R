

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
library(rpart.plot)



data = read.csv("Questions.CSV", sep=";")
ID = seq.int(nrow(data))
QUESTIONS = as.character(data$Question)
TYPE=data$Type
d=data.frame(ID,QUESTIONS,TYPE)
d$QUESTIONS=as.character(QUESTIONS)

summary(d)


dtm = makeDTM(d$QUESTIONS)
dim(dtm)

train.set=sample(nrow(d),nrow(d)*0.8)
test.set = 1:dim(d)[1]
test.set=test.set[-train.set]

summ <- summary(dtm)
df <- data.frame(as.matrix(dtm), stringsAsFactors=FALSE)

fit <- rpart(TYPE[train.set]~.,data=df[train.set,], control = rpart.control(minsplit=1))
pred<-predict(fit, newdata=df[-train.set,], type="class")

fit<- prune(fit, cp=   fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

rpart.plot(fit, type = 3, clip.right.lab = FALSE, branch = .3, under = TRUE)

table(classified=pred,actual=d$TYPE[-train.set])

prop.table(table(pred==d$TYPE[-train.set]))

act=d$TYPE[-train.set]

# Print wrong results
#for(i in 1:length(pred)){
#  if(pred[i]!=act[i]){
#      print(x=paste("predicted:",pred[i],"actual:",act[i],"doc:",d$ID[test.set[i]]))
#  }
#}



