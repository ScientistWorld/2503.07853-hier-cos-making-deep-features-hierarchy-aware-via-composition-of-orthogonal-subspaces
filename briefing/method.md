# Paper's Method

## Key Contribution

Hier-COS represents a hierarchy as a vector space with one fixed orthonormal basis vector per taxonomy node. Each class is assigned the subspace spanned by its ancestors, itself, and its descendants, so related classes share dimensions and distant classes share fewer or none.

## Approach

A visual backbone maps an image to features, and a transformation module maps those features into the hierarchy-node coordinate space. Training uses a weighted path distribution target: the true nodes along the sample's taxonomy path receive increasing weights toward the leaf, and all other nodes receive zero mass. The loss combines KL divergence between this path distribution and a regularizer that encourages one active coordinate per true hierarchy level.

At inference, each leaf class is scored by the norm of the feature vector projected onto that leaf's subspace. The same projection-norm rule can be applied at coarser levels, producing a hierarchy-aware ranked prediction and a multi-level prediction path.

## Main Claims

The primary claim is that hierarchy-aware subspace composition improves hierarchical ranking quality and consistency without sacrificing fine-grained accuracy. Secondary claims are that the same classifier supports both fine-grained and multi-level classification, and that the HOPS metric better reflects ranked hierarchical preference than MS or AHD alone.
