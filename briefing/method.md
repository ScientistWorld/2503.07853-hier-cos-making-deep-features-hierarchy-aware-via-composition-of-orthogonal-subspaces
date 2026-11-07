# Paper's Method

## Key Contribution

Hier-COS represents a hierarchy as a vector space with one fixed orthonormal basis vector per taxonomy node. Each class is assigned the subspace spanned by its ancestors, itself, and its descendants, so related classes share dimensions and distant classes share fewer or none.

## Approach

A visual backbone maps an image to features, and a transformation module maps those features into the hierarchy-node coordinate space. Training uses a weighted path distribution target: the true nodes along the sample's taxonomy path receive increasing weights toward the leaf, and all other nodes receive zero mass. The loss combines KL divergence between this path distribution and a regularizer that encourages one active coordinate per true hierarchy level.

Training follows the paper's CIFAR-100 recipe: WRN-28-8 with the transformation head, 100 epochs, batch 64, SGD with cosine decay, alpha 0.05, on the 45K training split. As in the official pipeline, the checkpoint with the best validation top-1 accuracy is kept.

At inference, each leaf class is scored by the norm of the feature vector projected onto that leaf's subspace. The same projection-norm rule can be applied at coarser levels, producing a hierarchy-aware ranked prediction and a multi-level prediction path.

## Main Claims

The primary claim is that hierarchy-aware subspace composition improves hierarchical ranking quality and consistency without sacrificing fine-grained accuracy. Secondary claims are that the same classifier supports both fine-grained and multi-level classification, and that the HOPS metric better reflects ranked hierarchical preference than MS or AHD alone.
