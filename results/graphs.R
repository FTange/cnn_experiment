setwd("~/Dropbox/uni/nc/project/code/results/")
library(tidyverse)

layer1 <- read.csv("nn_1_layer.csv", header = T)
head(layer1)
layer1$reg1 <- as.factor(layer1$reg1)
layer1$layer1 <- as.factor(layer1$layer1)
layer1$epochs <- as.factor(layer1$epochs)

layer1 %>%
  filter(reg1 == 0,
         dropout1 == 0.2) %>%
ggplot(aes(layer1, accuracy)) +
  geom_point(aes(color = epochs)) +
  theme_minimal()

layer2 <- read.csv("nn_2_layer_aws.csv", header = T)
layer2$layer1 <- as.ordered(layer2$layer1)
layer2$layer2 <- as.ordered(layer2$layer2)

layer2 %>%
  filter(layer1 >= 256) %>%
ggplot(aes(layer1, layer2)) +
  geom_point(aes(color = f1)) +
  theme_minimal()










