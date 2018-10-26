setwd("~/Dropbox/uni/nc/project/code/results/")
library(tidyverse)

layer1 <- read.csv("nn_1_layer.csv", header = T)
head(layer1)
layer1$reg1 <- as.factor(layer1$reg1)
layer1$layer1 <- as.ordered(layer1$layer1)
layer1$epochs <- as.ordered(layer1$epochs)

# best 1 layer: 2048 nodes, 100 epochs, batch normalization
layer1 %>%
  filter(reg1 == '0.0' | reg1 == 'batch',
         dropout1 == 0.2) %>%
ggplot(aes(layer1, accuracy)) +
  geom_point(aes(color = epochs)) +
  theme_minimal()


layer1 %>%
  filter(epochs == 60,
         dropout1 == 0.2) %>%
ggplot(aes(layer1, accuracy)) +
  geom_point(aes(color = reg1)) +
  theme_minimal()















layer2 <- read.csv("nn_2_layer_aws.csv", header = T)
layer2$layer1 <- as.ordered(layer2$layer1)
layer2$layer2 <- as.ordered(layer2$layer2)


# (1024, 512) seems to be the best layer sizes
library(viridis)
layer2 %>%
  filter(layer1 >= 256,
         layer2 >= 256,
         as.numeric(layer2) <= as.numeric(layer1)) %>%
ggplot(aes(layer1, layer2)) +
  geom_tile(aes(fill = f1), color = 'black') +
  scale_fill_viridis() +
  theme_minimal()



layer2 %>%
  filter(layer1 >= 256) %>%
ggplot(aes(layer1, layer2)) +
  geom_point(aes(color = f1)) +
  theme_minimal()


# early stopping
library(reshape2)
layer2_stopping <- read.csv("2layer_early_stopping.csv", header = T)
layer2_stopping <- rename(layer2_stopping, out_sample = val_acc,
                          in_sample = acc)


layer2_stopping_melted <- melt(layer2_stopping, measure.vars = c("in_sample", "out_sample"),
                        value.name = "Accuracy", variable.name = "Sample")

ggplot(layer2_stopping_melted) +
  geom_line(aes(epoch, Accuracy, color = Sample)) +
  geom_vline(xintercept = 50, size = 1.2) +
  theme_minimal() +
  scale_color_hue(labels = c("In Sample", "Out Sample")) +
  labs(xlab("Epochs"))







