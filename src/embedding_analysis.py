import numpy as np
import matplotlib.pyplot as plt

embedding=np.load('embedding.npy')
print(embedding.shape)
print(embedding)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=5, alpha=0.7)
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
ax.set_title("3D UMAP Embedding")
plt.show()
