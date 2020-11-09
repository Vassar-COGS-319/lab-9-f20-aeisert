Sys.setenv('CUDA_VISIBLE_DEVICES' = "5")

# STAGE 2: Feature Embeddings

# This script loads the image data from STAGE 1 and runs it through
# the pre-trained CNN to extract feature embeddings. 

library(keras)

# Read in the image data
data <- readRDS('scene-data.rds')

# Preprocess the images through the tf model
images.preprocessed <- imagenet_preprocess_input(data$images, mode="tf")

# We download the VGG16 model with weights from the Keras server.
# If we set include_top = F then the model has only the convolutional
# feature layers. include_top = T will include the two dense layers at
# the top of the network, plus the categorization layer.

conv_base <- application_vgg16(input_shape = c(128,128,3), include_top = FALSE)
?application_vgg16
summary(conv_base)

# We can setup a model that incorporated the VGG16 model (pre-trained), but with a 
# flatten layer at the top to turn the feature maps into a single
# vector of features, which will be easier to use in our MINERVA model.

##Run it thru the feature dectors of the VGG model, and then flatten it,
##So now it will be an apt starting point for another model

model <- keras::keras_model_sequential()
model %>%
  conv_base %>%
  layer_flatten()

## another option for our starting layer:
keras_model(inputs=conv_base$input, outputs = get_layer(conv_base, 'block4_pool')$outputs)
## can also change "include_top" to TRUE


# Get the features by running predict()

features <- model %>% predict(images.preprocessed)

# Save the resulting output
saveRDS(features, file="scene-features-vgg16-128.rds")
