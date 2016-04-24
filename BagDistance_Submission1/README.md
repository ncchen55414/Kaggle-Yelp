This submission implements the ChamferDistance+SVM method mentioned in Amores' paper,
[Multiple Instance Classification: review, taxonomy and comparative study](http://158.109.8.37/files/Amo2013.pdf#Page=14), p.14.

Step 1: Extract features using [ResNet-101 model](https://github.com/facebook/fb.resnet.torch).
I'm [*not* sure](https://www.reddit.com/r/cs231n/comments/4csdsm/convnet_as_feature_extractor/) if it's the best feature extractor available. 

Step 2: Compute Chamfer distances between bags.

Step 3: Convert the distances to a kernel. Train SVM classifier for each attribute (9 in total) separately.

The submission scores are 0.81427 (public LB) and 0.82218 (private LB).
