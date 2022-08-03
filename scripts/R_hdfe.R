
df = read.csv("C:\\Users\\Samuel\\fta\\analyze_exportdf.csv")

library(fixest)
library(dplyr)
library(tidyr)
library(pracma)


df=select(df, -c(X))

ftadummies = read.csv("C:\\Users\\Samuel\\fta\\analyze_ftaFE.csv")

#df = left_join(df, ftadummies, by=c('year', 'parties_perm'))

#df = df %>% replace(is.na(.), 0)

df = select(df, -parties_perm)

my_vector = colnames(df)

my_vector = my_vector[!my_vector %in% c("year", "export_value", "exporter", "importer", "parties_perm")]

dep_formula = paste(my_vector,collapse=" + ")

formula = paste0("export_value ~ ", dep_formula)

formula = paste0(formula, " | exporter^importer + exporter^year + importer^year")

evaluator = paste0("feglm(", formula, ", data= df, family ='poisson')")

gravity_glm = eval(parse(text=evaluator))


gravity_glm = feglm(export_value ~ x_0 + x_1 + x_2 + x_3 | exporter^importer + exporter^year + importer^year, data=df, family='gaussian')





############## chapter level counts first

clausekey = read.csv("C:\\Users\\Samuel\\fta\\core_excels\\clauseid_key.csv")






## check for singularity

mt=data.matrix(df[,5:ncol(df)])
library(Matrix)

rankMatrix(mt)

mt=rref


# check if there are any two 

ftadummies2=pivot_longer(ftadummies, cols = colnames(ftadummies[4:ncol(ftadummies)]))
ftadummies2 = ftadummies2 %>% group_by(year, parties_perm) %>% summarize(activeFTAs = sum(value))

unique_parties = unique(ftadummies2$parties_perm)

for(party in unique_parties){
  subdummy = ftadummies2 %>% filter(parties_perm == party)
  difference_vector = diff(subdummy$activeFTAs)
  
  if(any(difference_vector > 1)){
    print(party)
  }
}

checkvnmtha = ftadummies %>%  filter(parties_perm == "{'VNM', 'THA'}")






######################## chapter level regression

chapterlevel= read.csv("C:\\Users\\Samuel\\fta\\analyze_chapterlevel.csv")

chapterpoisson_1 = feglm(export_value ~ x_1 + x_2 + x_9 + x_10 + x_11 + x_15| exporter^importer + exporter^year + importer^year, data=chapterlevel, family='poisson', cluster='exporter^importer')
chapterpoisson_2 = feglm(export_value ~ x_1 + x_2 + x_9 + x_10 + x_11 + x_15| exporter^importer + exporter^year + importer^year, data=chapterlevel, family='poisson', cluster='exporter')
chapterpoisson_3 = feglm(export_value ~ x_1 + x_2 + x_9 + x_10 + x_11 + x_15| exporter^importer + exporter^year + importer^year, data=chapterlevel, family='poisson', cluster='importer')
chapterpoisson_4 = feglm(export_value ~ x_1 + x_2 + x_9 + x_10 + x_11 + x_15| exporter^importer + exporter^year + importer^year, data=chapterlevel, family='poisson', cluster='year')

chapterpoisson = feglm(export_value ~ x_0 +  x_1 + x_2 + x_3 + x_4 + x_5 +x_6 + x_7 + x_8 + x_9 + x_10 + x_11 + x_12 + x_13 + x_14 + x_15 + x_16| exporter^importer + exporter^year + importer^year, data=chapterlevel, family='poisson', cluster='exporter^importer')

newdf = cbind(tidy(chapterpoisson_1),
tidy(chapterpoisson_2),
tidy(chapterpoisson_3),
tidy(chapterpoisson_4))

write.csv(newdf,'chapterdroppedpoison.csv')

etable(gravity_glm)

######################## chapter level regression with lags

library(plm)

chapterlevelplm = pdata.frame(chapterlevel, index=c('parties_perm','year'),drop.index=TRUE)

chapterlevelplm$x_0 = lag(chapterlevelplm$x_0)
chapterlevelplm$x_1 = lag(chapterlevelplm$x_1)
chapterlevelplm$x_2 = lag(chapterlevelplm$x_2)
chapterlevelplm$x_3 = lag(chapterlevelplm$x_3)
chapterlevelplm$x_4 = lag(chapterlevelplm$x_4)
chapterlevelplm$x_5 = lag(chapterlevelplm$x_5)
chapterlevelplm$x_6 = lag(chapterlevelplm$x_6)
chapterlevelplm$x_7 = lag(chapterlevelplm$x_7)
chapterlevelplm$x_8 = lag(chapterlevelplm$x_8)
chapterlevelplm$x_9 = lag(chapterlevelplm$x_9)
chapterlevelplm$x_10 = lag(chapterlevelplm$x_10)
chapterlevelplm$x_11 = lag(chapterlevelplm$x_11)
chapterlevelplm$x_12 = lag(chapterlevelplm$x_12)
chapterlevelplm$x_13 = lag(chapterlevelplm$x_13)
chapterlevelplm$x_14 = lag(chapterlevelplm$x_14)
chapterlevelplm$x_15 = lag(chapterlevelplm$x_15)
chapterlevelplm$x_16 = lag(chapterlevelplm$x_16)

new.b <- sapply(chapterlevelplm, function(x){attr(x, "index") <- NULL; x})

library(stringr)

backtodf = data.frame(names(chapterlevelplm$X))
colnames(backtodf) = 'compound'

                               
backtodf = data.frame(str_split_fixed(backtodf$compound, '-', 2))
colnames(backtodf) = c('parties_perm', 'year')

library(readr)

new.b = type_convert(data.frame(new.b))

cllo = new.b

cllo$parties_perm = backtodf$parties_perm
cllo$year = backtodf$year

cllo$export_value = as.numeric(cllo$export_value)

gravity_glm = feglm(export_value ~ x_0 + x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_10 + x_11 + x_12 + x_13 + x_14 + x_15 + x_16| exporter^importer + exporter^year + importer^year, data=cllo, family='poisson', cluster='exporter^importer')

etable(gravity_glm)


####################################
# separate subregressions

subregresults = data.frame()

df = read.csv("C:\\Users\\Samuel\\fta\\analyze_exportdf.csv")

constantdf = df %>% select('year', 'export_value', 'exporter', 'importer', 'parties_perm')

list_of_interest = c('competition.policy', 'trade.facilitation', 'intellectual.property.rights', 'rules.of.origin', 'sanitary.and.phytosanitary.measures', 'technical.barriers.to.trade', 'trade.in.goods')


allrest = df %>% select(-year, -export_value, -exporter, -importer, -parties_perm, -Unnamed..0, -X, -year.1)

allrest = unique(str_split_fixed(colnames(allrest), "_", 2)[,1])

allrest = setdiff(allrest, list_of_interest)

for(chapter in allrest){

  print(ncol(df[ , grepl( chapter , names( df ) ) ]))
  
  
  subregdf = cbind(constantdf, df[ , grepl( chapter , names( df ) ) ])
  
  
  my_vector = colnames(subregdf)
  
  my_vector = my_vector[!my_vector %in% c("year", "export_value", "exporter", "importer", "parties_perm")]
  
  dep_formula = paste(my_vector,collapse=" + ")
  
  formula = paste0("export_value ~ ", dep_formula)
  
  formula = paste0(formula, " | exporter^importer + exporter^year + importer^year")
  
  evaluator = paste0("feglm(", formula, ", data= subregdf, family ='poisson')")
  
  print('evaluating')
  print(chapter)
  
  gravity_glm = eval(parse(text=evaluator))
  
  library(broom)
  
  subregresults = rbind(subregresults, tidy(gravity_glm))
}

write.csv(subregresults, 'subregresults2.csv')

interestfulldf = df[ , grepl( 'competition.policy' , names( df ) ) ]

for(chapter in setdiff(list_of_interest, 'competition.policy')){
  subdf = cbind(constantdf, df[ , grepl( chapter , names( df ) ) ])
  interestfulldf = cbind(interestfulldf, subdf)
  
}


my_vector = colnames(interestfulldf)

my_vector = my_vector[!my_vector %in% c("year", "export_value", "exporter", "importer", "parties_perm")]

dep_formula = paste(my_vector,collapse=" + ")

formula = paste0("export_value ~ ", dep_formula)

formula = paste0(formula, " | exporter^importer + exporter^year + importer^year")

evaluator = paste0("feglm(", formula, ", data= interestfulldf, family ='gaussian')")

print('evaluating')
print(chapter)

gravity_glm = eval(parse(text=evaluator))


evaluator = paste0("fenegbin(", formula, ",  data= interestfulldf)")

gravity_negbin = eval(parse(text=evaluator))










############################
# attempt an all out poisson regression, not enough iterations last time

library(fixest)
library(dplyr)

df = read.csv("C:\\Users\\Samuel\\fta\\analyze_exportdf.csv")

constantdf = df %>% select('year', 'export_value', 'exporter', 'importer', 'parties_perm')

list_of_interest = c('competition.policy', 'trade.facilitation', 'intellectual.property.rights', 'rules.of.origin', 'sanitary.and.phytosanitary.measures', 'technical.barriers.to.trade', 'trade.in.goods')

interestfulldf = df[ , grepl( 'competition.policy' , names( df ) ) ]

for(chapter in setdiff(list_of_interest, 'competition.policy')){
  subdf = cbind(constantdf, df[ , grepl( chapter , names( df ) ) ])
  interestfulldf = cbind(interestfulldf, subdf)
  
}


my_vector = colnames(interestfulldf)

my_vector = my_vector[!my_vector %in% c("year", "export_value", "exporter", "importer", "parties_perm")]

#my_vector = paste("log(", my_vector, sep="")

#my_vector = paste(my_vector, ")", sep="")

dep_formula = paste(my_vector,collapse=" + ")

formula = paste0("export_value ~ ", dep_formula)

formula = paste0(formula, " | exporter^importer + exporter^year + importer^year")

evaluator = paste0("feNmlm(", formula, ", data= interestfulldf, family='poisson', verbose=1, warn=FALSE)")

gravity_glm = eval(parse(text=evaluator))




































####################################
# separate subregressions


df = read.csv("C:\\Users\\Samuel\\fta\\analyze_exportdf.csv")

constantdf = df %>% select('year', 'export_value', 'exporter', 'importer', 'parties_perm')
  
allrest = df %>% select(-c(year, export_value, exporter, importer, parties_perm))

allrest = unique(str_split_fixed(colnames(allrest), "_", 2)[,1])


subregresults = data.frame()

for(chapter in allrest){
  
  print(ncol(df[ , grepl( chapter , names( df ) ) ]))
  
  
  subregdf = cbind(constantdf, df[ , grepl( chapter , names( df ) ) ])
  
  
  my_vector = colnames(subregdf)
  
  my_vector = my_vector[!my_vector %in% c("year", "export_value", "exporter", "importer", "parties_perm")]
  
  
  dep_formula = paste(my_vector,collapse=" + ")
  
  formula = paste0("export_value ~ ", dep_formula)
  
  formula = paste0(formula, " | exporter^importer + exporter^year + importer^year")
  
  evaluator = paste0("feglm(", formula, ", data= subregdf, family ='poisson', cluster = 'exporter^year + importer^year')")
  
  print('evaluating')
  print(chapter)
  
  gravity_glm = eval(parse(text=evaluator))
  
  library(broom)
  
  subregresults = rbind(subregresults, tidy(gravity_glm))
}

write.csv(subregresults, 'subregresultsSpecific.csv')



#########################################
# lag subregressions 



df = read.csv("C:\\Users\\Samuel\\fta\\analyze_exportdf.csv")

constantdf = df %>% select('year', 'export_value', 'exporter', 'importer', 'parties_perm')

allrest = df %>% select(-c(year, export_value, exporter, importer, parties_perm))

allrest = unique(str_split_fixed(colnames(allrest), "_", 2)[,1])


subregresults = data.frame()

for(chapter in allrest){
  
  print(ncol(df[ , grepl( chapter , names( df ) ) ]))
  
  
  subregdf = cbind(constantdf, df[ , grepl( chapter , names( df ) ) ])
  
  
  subregplm = pdata.frame(subregdf, index=c('parties_perm','year'),drop.index=FALSE)
  
  my_vector = colnames(subregdf)

  
  my_vector = my_vector[!my_vector %in% c("year", "export_value", "exporter", "importer", "parties_perm")]
  
  for(col in my_vector){
    tochange_index = which(colnames(subregplm) == col)
    subregplm[,tochange_index] = lag(subregplm[,tochange_index], 2)
  }
  
  
  dep_formula = paste(my_vector,collapse=" + ")
  
  formula = paste0("export_value ~ ", dep_formula)
  
  formula = paste0(formula, " | exporter^importer + exporter^year + importer^year")
  
  evaluator = paste0("feglm(", formula, ", data= subregplm, family ='poisson', cluster = 'exporter^year + importer^year')")
  
  print('evaluating')
  print(chapter)
  
  gravity_glm = eval(parse(text=evaluator))
  
  library(broom)
  
  subregresults = rbind(subregresults, tidy(gravity_glm))
}



write.csv(subregresults, 'lagsubreg.csv')
