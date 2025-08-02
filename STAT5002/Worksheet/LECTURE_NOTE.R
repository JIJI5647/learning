# Read in data
data = read.csv("data/2023fatalities.csv",header = TRUE)

# Name of the variable
names(data)

# Select the State of Data
State = data$State
print(table(State))

# Barplot the State
barplot(table(State),
        main = "State in Fatilities",
        ylab = "Fatalities",
        xlab = "State Name")

chisq.test(table(State))

# Size of data
dim(data)

# Get Dayweek
Dayweek = data$Dayweek
table(Dayweek)
# Sort Dayweek
mycols = factor(Dayweek, levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))
table(mycols)
# Barplot the Dayweek
barplot(table(Dayweek), col = mycols, main = "Fatalities by Day of the week", ylab = "Fatalities", xlab = "Day of the week")

# Get Gender
Gender = data$Gender
# Produce a double frequency table
data1 = table(Gender, Dayweek)
data1

# Historgram
hist(data$Age)

Age = data$Age
breaks = c(-10, 18, 25, 70, 101)
table(cut(Age,breaks, right = F))
hist(Age, br = breaks, right = F, freq = F)

data$Age[data$Age == -9] = NA

# Example of removing NA values
clean_data <- na.omit(data)

# Check for missing values
sum(is.na(data$Age))

# Identifying outliers using boxplot
boxplot(data$Age, main="Age Distribution")

# Removing duplicates
clean_data <- unique(data)
