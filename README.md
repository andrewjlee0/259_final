# Final Project for PSYCH 259 at UCLA

## Abstract
The development of ever more capable artificial intelligence far outpaces our understanding of them. The need for tools to understand them—-beyond the use of performance metrics—-is crucial to developing systems that are safer, more trustworthy, and more intelligent. Here, I probe the representational structure of convolutional neural networks trained on the Synthetic Visual Reasoning Test (SVRT) as a first step toward understanding such unintelligibility. I do so by visualizing embedding spaces with multidimensional scaling and by testing generalization performance to untrained, but relationally similar, datasets (such as the Parametric SVRT) with Support Vector Machines. Generalization tests suggest that training on one SVRT problem yields model states that form robust linearly separable representations of similar problems. In one type of SVRT problem, inside-outside problems, training on any one of them typically leads to generalization on the others, suggesting that the representations of inside-outside problems lie along similar dimensions.
MDS visualizations reinforce this view. In particular, two- and three-dimensional solutions of inside-outside image embeddings reveal a balanced flower structure, in which any line that crosses the axis of the flower separates all inside- outside problems nearly perfectly, and all petal lengths are roughly equal. MDS visualizations provide other interesting insights, including that ResNet-18’s trained on SVRT same- different problems cannot generalize to PSVRT because PSVRT embeddings form a cluster so small in embedding space that they are practically identical to each other. I take this to be in-accord with the idea that ResNet-18’s fail to generalize to the PSVRT from the SVRT because they lack the perceptual, not reasoning, capacity to segment PSVRT objects.
