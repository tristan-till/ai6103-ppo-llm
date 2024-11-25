from PIL import Image
from custom_states import get_custom_state_str
import utils.render_env as render_env
import utils.enums as enums
import numpy as np
import matplotlib.pyplot as plt

def cosine_distances():
    original_grid = "SFFFFHFHFFFHHFFG"
    embeddings = []
    
    for i in range(16):
        agent = i
        state_str = get_custom_state_str(original_grid, agent)
        embedding = render_env.encode_str(state_str)
        embeddings.append(embedding)
    
    similarity_matrix = np.zeros((16, 16))
    img = render_env.render_state(original_grid, "0_1", enums.EnvMode.TRAIN)
    img = img.convert("RGB")
    img = img.resize((256, 256), Image.NEAREST)
    bg_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        
    bg_image = np.array(bg_image)
    fig, ax = plt.subplots()

    ax.imshow(bg_image, extent=[3.5, -0.5, 3.5, -0.5], aspect='auto', zorder=0)
    
    # Calculate cosine similarity between embeddings
    for i, embedding in enumerate(embeddings):
        for j, other_embedding in enumerate(embeddings):
            cos_sim = np.dot(embedding, other_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(other_embedding)
            )
            similarity_matrix[i, j] = cos_sim
        
        pos_dist = similarity_matrix[i]
        pos_dist = np.reshape(pos_dist, (4, 4))
        
        heatmap = ax.imshow(pos_dist, cmap="coolwarm", interpolation="nearest", vmin=0, vmax=1, alpha=0.5)
        
        plt.colorbar(heatmap)
        
        for (j, k), val in np.ndenumerate(pos_dist):
            plt.text(k, j, f"{val:.2f}", ha='center', va='center', color='black')
        
        plt.title("Cosine Similarity Heatmap Between States")
        plt.xlabel("State Index")
        plt.ylabel("State Index")
        plt.xticks(ticks=np.arange(4), labels=np.arange(4))
        plt.yticks(ticks=np.arange(4), labels=np.arange(4))
        
        plt.show()
        return
   
if __name__ == '__main__':
    cosine_distances()