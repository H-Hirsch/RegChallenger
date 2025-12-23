#install.packages("haven")
library(haven)

df <- read_sas("ap21to25.sas7bdat")
head(df)
