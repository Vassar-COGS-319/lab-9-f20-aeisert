# MINERVA2 model

# This will be a partial implementation of MINERVA2. We're 
# only interested in the ENCODING and INTENSITY functions of the
# model. The ECHO feature isn't necessary for our purposes.

######
# encode.items should take a matrix of items to encode
# into memory and return the updated memory of the model
# after applying a lossy (i.e., lose some features) encoding.

# parameters:
# items - matrix. each row in the matrix is the features
#         for one item.
# store - matrix. any existing items already encoded.
# L - number between 0 and 1. Lossy encoding parameter.
#     1 means that all features will be perfectly encoded.
#     0 means that nothing will be encoded.
minerva.encode.items <- function(items, L=0.5){
  features.to.keep <- matrix(data=sample(0:1,length(items), replace=T, prob=c(1-L,L)), nrow=nrow(items))
  lossy.encoding <- items * features.to.keep
  return(lossy.encoding)
}

# This is a somewhat optimized cosine similarity function.
# We will need to be able to computer cosine similarity of
# a single item to all items in the store. This operation
# is the most computationally demanding part of the model.
# This function uses some linear algebra tricks to compute
# this more efficiently.

cosine.similarity.vector.to.matrix <- function(x,m){
  x <- x / as.vector(sqrt(crossprod(x)))
  return(  as.vector((m %*% x) / sqrt(rowSums(m^2))) )
}

# For reference, here's what a simple cosine similarity
# function might look like, virtually the same as computing a single similarity score in the original Minerva paper

cosine.similarity <- function(x,y){
  return( sum(x*y) / sqrt(sum(x^2) * sum(y^2)))
}

# intensity takes a single item and existing memory
# store and returns the intensity with which the item
# resonates in memory.

# parameters:
# item - vector. the features of the item to test.
# store - matrix. memory store of the model.
# tau - number >= 1. controls the steepness of the 
#       similarity function. larger numbers mean
#       that less similar items contribute less to
#       the intensity of activation.
minerva.intensity <- function(item, store, tau=3){
 similarities <- cosine.similarity.vector.to.matrix(item, store)
 ##Gives back a list, a vector, where the first item is similarity to first item in memory,
 ##Second item is similarity to second item in memory, and so on.
 activations <- similarities^tau
 intensity <- sum(activations)
 return(intensity)
}

