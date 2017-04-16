library(tm)
library(dplyr)
library(stringr)
library(caret)
library(caTools)

set.seed(19)

texts <- read.csv("C://Users/bdaet/Desktop/myProjects/Text_Spam/spam.csv")

#taking a look at the data set
glimpse(texts)
summary(texts)

#looks like v1 and v2 and the only meaningful variables so let's remove the other ones
texts <- select(texts, v1, v2)

#converting the Spam or Ham column to a binary format where 0 = ham and 1 = spam
texts$v1 <- as.character(texts$v1)
texts$v1 <- str_replace(texts$v1, "ham", "0")
texts$v1 <- str_replace(texts$v1, "spam", "1")
texts$v1 <- as.factor(texts$v1)

#checking to see how imbalanced the data set is
table(texts$v1)

#it looks like the data set is imbalanced: roughly 85% negative (ham) and 15% positive (spam)

#converting the texts data frame into a corpus
texts_corp <- Corpus(VectorSource(texts$v2))

#applying multiple tm text cleaning functions using the pipe operator from the dplyr package
texts_corp <- texts_corp %>%
  tm_map(PlainTextDocument) %>% #converting to a plain text document (necessary to create a DTM later)
  tm_map(removePunctuation) %>% #removing punctuation
  tm_map(removeNumbers) %>% #removing numbers 
  tm_map(content_transformer(tolower)) %>% #making all words lowercase
  tm_map(removeWords, stopwords(kind = "en")) %>% #removing stopwords
  tm_map(content_transformer(stripWhitespace)) #getting rid of extra whitespace between words

#creating a Document Term Matrix
dtm <- DocumentTermMatrix(x = texts_corp,
                          control = list(tokenize = "words",
                                         stemming = "english",
                                         weighting = weightTf))

#converting the DTM to a dataframe and checking the dimensions
tf <- as.data.frame.matrix(dtm)
dim(tf)

#looking at the most frequent terms overall
term_freq <- colSums(tf)
freq <- data.frame(term = names(term_freq), count = term_freq)
arrange(freq, desc(count))[1:20,]

#finding the terms that appear more than 50 times total
most_freq <- filter(freq, count >= 50)

#selecting only the terms in the tf data frame that appear over 50 times overall
# because the select function can't select columns contained in a character vector, I had to use a text editor to 
# quickly create a list of the terms that I wanted to select using the output of the dataframe "most_freq"
tf <- select(tf, already,    
             also,    
             always,    
             amp,    
             anything,    
             around,    
             ask,    
             babe,    
             back,   
             box,    
             buy,    
             call,   
             can,   
             cant,   
             care,    
             cash,    
             chat,    
             claim,   
             come,   
             coming,    
             contact,    
             cos,    
             customer,    
             day,   
             dear,   
             didnt,    
             dont,   
             dun,    
             even,    
             every,    
             feel,    
             find,    
             first,    
             free,   
             friends,    
             get,   
             getting,    
             give,   
             going,   
             gonna,    
             good,   
             got,   
             great,   
             guaranteed,    
             gud,    
             happy,   
             help,    
             hey,   
             home,   
             hope,   
             ill,   
             ive,    
             just,   
             keep,    
             know,   
             last,    
             late,    
             later,   
             leave,    
             let,    
             life,    
             like,   
             lol,    
             lor,   
             love,   
             ltgt,   
             make,   
             many,    
             meet,    
             message,    
             mins,    
             miss,    
             mobile,   
             money,    
             morning,    
             msg,    
             much,   
             name,    
             need,   
             new,   
             nice,    
             night,   
             nokia,    
             now,   
             number,    
             one,   
             people,    
             per,    
             phone,   
             pick,    
             place,    
             please,   
             pls,   
             prize,    
             really,    
             reply,   
             right,    
             said,    
             say,    
             see,   
             send,   
             sent,    
             service,    
             sleep,    
             someone,    
             something,    
             soon,    
             sorry,   
             still,   
             stop,   
             sure,    
             take,   
             tell,   
             text,   
             thanks,    
             thats,    
             thing,    
             things,    
             think,   
             thk,    
             time,   
             today,   
             told,    
             tomorrow,    
             tonight,    
             txt,   
             urgent,    
             wait,    
             waiting,    
             wan,    
             want,   
             wat,    
             way,   
             week,   
             well,   
             went,    
             will,   
             win,    
             wish,    
             won,    
             wont,    
             work,    
             yeah,    
             year,    
             yes,    
             yet,    
             youre)   


#adding the Spam or ham column from the original texts dataframe to the new term frequency data frame

texts_tf <- mutate(tf, Class = texts$v1)

#splitting the data into a test and training set
trainIndex <- createDataPartition(texts_tf$Class, 
                                  times = 1,
                                  p = 0.8, 
                                  list = FALSE)

train <- texts_tf[trainIndex, ]
test <- texts_tf[-trainIndex, ]

#checking to see how imbalanced the training set is
table(train$Class)


#fitting a logistic regression model
model <- train(Class ~ .,
               data = train,
               method = "glm",
               trControl = trainControl(method = "cv",
                                        number = 10)
               )
model

#looking at variable importances
varImp(model)

# Calculate class probabilities:
p <- predict(model, test, type = "prob")


# through trial and error I found that setting the classification threshold to 0.4 slightly improved the accuracy
# of predicting spam
predictions <- factor(ifelse(p["0"] > 0.4,
                              "0",
                              "1"))

# Create confusion matrix
#remember ham = 0 and spam = 1
confusionMatrix(predictions, test$Class)

#creating ROC curve
colAUC(p, test$Class, plotROC = TRUE)








