##############################

# LOAD REQUIRED PACKAGES

##############################

library(raster)
library(gbm)
library(dismo)
library(caret)
library(ROCR)
library(dplyr)
library(pROC)
library(plotmo)
library(ggplot2)
library(tidyverse)
library(gt)
library(DescToolsAddIns)
library(prettymapr)

#================================

# SET PERSONAL WORKING DIRECTORY

#================================

setwd(dir.choose(caption = "Select Directory")) # change to local working directory on your personal computer

#================================

# LOAD AND PREPARE DATA

#================================

#Load in all raster layers
ref <- raster("RdNBR_classified.tif")
c.area <- raster("Catchment_Area.tif") #uses a 20m resampled DEM
ci <- raster("Convergence_Index.tif") #uses a 20m resampled DEM
swi <- raster("SAGA_Wetness_Index.tif") #uses a 20m resampled DEM
tpi.200 <- raster("TPI_200m.tif") #uses a 20m resampled DEM
slope <- raster("Slope.tif") #uses a 20m resampled DEM
dist.water <- raster("Distance_to_Water.tif") #at a 20m resolution
ndmi <- raster("NDMI_June29_2018.tif")

#Create a list of all predictor variables that need to be resampled
ras.list <- as.list(c.area, ci, swi, tpi.200, slope, dist.water, ndmi)

#Create a list to send resampled layers to
rs <- list(ref)

#Run a loop to resample all rasters in ras.list with the binary refugia layer
for(i in 1:length(ras.list)){
  rs[[i]] <- resample(ras.list[[i]], ref, method = 'bilinear')
}

#Add the refugia layer to the final list
rs[[8]] <- ref

#Stack all rasters in the list
s <- stack(rs)

#Create a data frame from the raster stack, extracting values as points
df <- as.data.frame(rasterToPoints(s))

#Omit any rows with missing data
df <- na.omit(df)

#Change the names in df
colnames(df) = c('x','y','c.area','ci','swi', 'tpi.200', 'slope', 'dist.water', 'ndmi', 'ref')

#Reduce the amount of data using stratified random sampling (15000 refugia cells, 15000 burned cells)
set.seed(45)
df.red <- df %>%
  group_by(ref) %>%
  sample_n(size = 15000)

#Split the data into a testing and a training dataset (80% training, 20% testing)
set.seed(500)
splitIndex <- createDataPartition(df.red$ref, p = .8, list = FALSE, times = 1)
train.ref <- df.red[splitIndex,]
test.ref <- df.red[-splitIndex,]

#Convert the training set to a dataframe (avoids errors with gbm.step)
train.ref <- as.data.frame(train.ref)

#==========================

# TRAIN THE GBM MODEL

#==========================


#Run gbm.step *THIS CREATES THE MAIN MODEL OBJECT
gbm_run <- gbm.step(data=train.ref, gbm.x = 3:9, gbm.y = 10, 
                family = "bernoulli", tree.complexity = 5, 
                learning.rate = 0.02, bag.fraction = 0.5, tolerance.method = 'fixed',
                cv.folds = 5, n.trees = 500, step.size = 50)

#Create a dataframe with the model results
ntrees = gbm_run$n.trees
trainAUC = gbm_run$self.statistics$discrimination
cvAUC = gbm_run$cv.statistics$discrimination.mean
train.cor = gbm_run$self.statistics$correlation
cv.cor = gbm_run$cv.statistics$correlation.mean
train.perdevex = 1-(gbm_run$self.statistics$mean.resid/gbm_run$self.statistics$mean.null)
cv.dev = gbm_run$cv.statistics$deviance.mean

mod.output <- data.frame(stat = c("ntrees", "trainAUC", "cvAUC", "train.cor", "cv.cor", "train.perdevex", "cv.dev"), 
                         result = c(ntrees, trainAUC, cvAUC, train.cor, cv.cor, train.perdevex, cv.dev))

#round the results to 3 decimal places
mod.output$result <- round(mod.output$result, 3)


#===========================

# EXAMINE THE MODEL RESULTS

#===========================

#Examine the level of correlation between variables
library(corrplot)
corrplot(cor(train.ref, method = "pearson"), method = "number")

#Print the relative influence plot
rel.inf <- summary(gbm_run, plotit = T)
rel.inf$var <- c("NDMI", "TPI (200m)", "Slope", 
                 "SWI", "Distance to Water", "Convergence Index", "Catchment Area")
rel.inf %>% 
  ggplot(aes(x = reorder(var, rel.inf), y = rel.inf))+
  geom_col(aes(fill = as.factor(reorder(var, -rel.inf))), show.legend = F) +
  scale_fill_grey() +
  coord_flip()+
  scale_y_continuous(expand = c(0,0), limits = c(0, 50)) +
  labs(y = "Relative Influence (%)", x = "")+
  geom_text(aes(label = paste(round(rel.inf, 1), "%")), nudge_y = 2, color="black", size = 4) +
  theme_minimal() +
  theme(panel.grid.major = element_line(color = "white"), panel.grid.minor = element_line(color = "white"),
        axis.line = element_line(color = "black"), axis.ticks.length.x = unit(0.2, "cm"), 
        axis.ticks.x = element_line(color = "black"), axis.text = element_text(size = 12),
        axis.title = element_text(size = 13, vjust = -2, face = "bold"),
        axis.text.y = element_text(face = "bold")) 

#Print the partial dependence plots 
par(mfrow = c(2,4))
title <- c(paste0('Catchment Area ', '(m\u00b2)'),'Convergence Index (°)','SAGA Wetness Index', 
           'TPI (200m)', 'Slope (°)', 'Distance to Water (m)', 'NDMI')

#note: readjust order of variables to match relative influence
rel.inf$varnum <- c(7, 4, 5, 6, 3, 2, 1)

for (i in rel.inf$varnum){
  plotmo(gbm_run, pmethod = "partdep", degree1 = i, degree2 = FALSE, type = 'response', 
         ylab = "Refugia Probability", cex.lab = 1.1, nrug = 0, 
         do.par = F, lwd = 0.1, lty = 2, xlab = title[[i]], main = '', smooth.col = 1, smooth.f = 0.2,
         smooth.lwd = 2, col = alpha("grey", f = 0), ylim = c(0,1), #, xlim = xlim[[i]]
  )
  rect(xleft = quantile(train.ref[,i+2], probs = 0.05), ybottom = 0, xright = quantile(train.ref[,i+2], probs = 0.95), ytop = 1, col = alpha('black', f = 0.2))
}

#Make predictions on the testing set
preds <- predict.gbm(gbm_run, test.ref,
                     n.trees=gbm_run$gbm.call$best.trees, type="response")

#Construct on a confusion matrix to assess predictions, convert continuous output to binary with threshold = 0.5
preds.thresh <- ifelse(preds <= 0.5, 0, 1)
preds.thresh <- as.factor(preds.thresh)
confusionMatrix(preds.thresh, reference = as.factor(test.ref$ref), positive = '1')

#Calculate AUC (* on the test set)
AUC <- Metrics::auc(test.ref$ref,preds)
AUC

#ROC plot
prediction <- prediction(predictions = preds,labels = test.ref$ref)
perf <- performance(prediction,"tpr","fpr")
plot(perf,lwd=2,col='blue',main="ROC Curve")
abline(a=0, b= 1)

#Output the predictions in raster format (NOTE: model end users would input their data here)
outdir <- getwd() #Set to preferred output directory
names(s) <- c('c.area','ci','swi', 'tpi.200', 'slope', 'dist.water', 'ndmi', 'ref')
pred.test <- predict(s, gbm_run, type = "response", n.trees=gbm_run$gbm.call$best.trees, na.rm=TRUE, progress="text",
                     filename=paste(outdir, "gbm_run", ".tif", sep = ""), format = "GTiff", overwrite = T)

#reclassify the continuous refugia probability output into categories
class <- c(0, 0.1, 1, 
           0.1, 0.5, 2, 
           0.5, 0.9, 3, 
           0.7, 1, 4)
rcl <- matrix(class, 
              ncol=3, 
              byrow=TRUE)
pred.rcl <- reclassify(pred.test, 
                       rcl)

#Export the reclassified raster
pred.rcl <- ratify(pred.rcl, count = TRUE)
writeRaster(pred.rcl, filename = paste0(outdir, "predrcl.tif"), options = c('dbf = YES'), overwrite = T)

#Make a map of results
shp = shapefile("E:/_ags_DMTI_2019_CMCS_WaterbodiesRegion/DMTI_2019_CMCS_WaterbodiesRegion_proj.shp") #### change to local directory on your personal computer, can use personal basemap ####
my_col = c("chocolate2", "goldenrod1", "mediumpurple1", "mediumpurple4")
colfunc <- colorRampPalette(my_col)
legend_image <- as.raster(matrix(colfunc(20), ncol=1))
par(mar = c(0,0,0,0))
plot(shp, col="grey80", bg="white", lwd=0.25, border=0)
plot(pred.test,
     axes = T, legend = F, col = my_col, add = TRUE)
text(x=531500, y = seq(5088400,5090680,l=3), labels = seq(0,1,l=3), cex = 0.95)
text(x=530900, y = 5091880, label = "Refugia", cex = 1)
text(x=530900, y = 5091280, label = "Probability", cex = 1)
rasterImage(legend_image, 530500,5090680,531000,5088400)
#legend.gradient(515600, 515610, 5097680, 5097690, legend = c("Very low", "Very high"), cols = my_col, gradient = 'y')
#legend.args = list(text='Refugia Probability', side = 4, line = 2.5, 
#font = 2, cex = 0.8))
rect(xleft =514600, xright = 520500, ytop = 5077680, ybottom = 5075400, col = 'white')
scalebar(5000, xy = c(515100, 5076300), divs = 3, type = 'bar', below = 'Kilometers', 
         label = c(0, 2.5, 5), adj = c(0.5, -1.8), col = 'black', bg = 'white')
addnortharrow(pos = "topright", padin = c(2.5, 1), scale = 0.8)

#========================================

#Create a table with the model variables

#========================================

#First create a dataframe with all variables and their stats
Variable <- c('Catchment Area','Convergence Index (CI)','SAGA Wetness Index (SWI)', 
              'Topographic Position Index (200m)', 'Slope', 'Distance to Water', 'NDMI')
df.table <- as.data.frame(Variable)
Mean <- c()
Median <- c()
SD <- c()
Min. <- c()
Max. <- c()
df$c.area = (df$c.area)/10000
for (i in c(3:9)){
  Mean[i] <- mean(df[,i], na.rm = T)
  Median[i] <- median(df[,i], na.rm = T)
  SD[i] <- sd(df[,i], na.rm = T)
  Min.[i] <- min(df[,i], na.rm = T)
  Max.[i] <- max(df[,i], na.rm = T)
}

Mean <- Mean[-c(1:2)]
Median <- Median[-c(1:2)]
SD <- SD[-c(1:2)]
Min. <- Min.[-c(1:2)]
Max. <- Max.[-c(1:2)]

Units <- c("ha", "\u00B0", "NA", "m", "\u00B0", "m", "NA")
Description <- c("Total contributing upslope area derived from flow accumulation", "Degree to which the aspects of the surrounding cells
                 converge towards (+), or diverge from (-) the center cell in a 3x3 window",
                 "Index approximating soil wetness based on topographic controls, SWI uses a modified catchment
                 area which does not treat flow as a very thin film",
                 "The elevation of a cell relative to a 200m-radius circular neighbourhood surrounding that cell (negative values
                 are lower than surroundings, positive values are higher than surroundings",
                 "Slope (in degrees) of a cell", "Euclidean distance to the nearest mapped water body",
                 "Normalized Difference Moisture Index (taken in June 2018, pre-fire), increasing values (approaching 1)
                 indicate dense, healthy vegetation not experiencing water stress")
Source <- c("DEM","DEM","DEM","DEM","DEM","DMTI Spatial Inc.", "Sentinel-2 L2A Imagery")

df.table <- cbind(df.table, Units, Description, Mean, Median, SD, Min., Max., Source)


gt(df.table, rowname_col = 'Variable') %>%
  fmt_number(columns = c(4:8), rows = everything(), decimals = 2) %>%
  cols_align(align = 'center') %>%
  tab_style(cell_text(weight = 'bold'), locations = cells_column_labels()) %>%
  tab_style(style = cell_borders(sides = "all", color = "lightgray", 
                                 style = "solid", weight = px(0.5)), locations = cells_body())
