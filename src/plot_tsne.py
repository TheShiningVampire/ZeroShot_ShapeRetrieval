# %% 
import os
import yaml
import hydra
import pyrootutils
from omegaconf import DictConfig
import torch
import warnings
from torchvision.transforms import transforms
from PIL import Image
import faiss
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=UserWarning)


# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


# %%
@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> float:
    domain_disentangled_model = hydra.utils.instantiate(cfg.model)
    domain_disentangled_model.load_state_dict(torch.load(cfg.ckpt_path)["state_dict"])
    domain_disentangled_model.eval()
    domain_disentangled_model.cuda()

    data_dir = cfg.paths.data_dir

    test_dataset = os.path.join(data_dir, "test")

    test_images = os.path.join(test_dataset, "img")
    test_models = os.path.join(test_dataset, "model")

    # Pass all the images through the model and get the feature vectors
    image_features = []
    model_features = []
    synset_set = {
                synset
                for synset in os.listdir(test_images)
                if os.path.isdir(os.path.join(test_images, synset))
            }
    classes = sorted(list(synset_set))
    label_by_number = {k: v for v, k in enumerate(classes)}

    # Transform for the images
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                    mean=[0.9799, 0.9799, 0.9799],
                                                    std=[0.1075, 0.1075, 0.1075]
                                                )
                                             ])
    
    # Dictionary to store number of models per class
    classwise_model_count = {}

    for class_name in classes:
        class_dir = os.path.join(test_images, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0)
            image = image.cuda()
            with torch.no_grad():
                image_feature = domain_disentangled_model(image=image)

            # Tuple of (image_feature, class_number)
            image_features.append((image_feature, class_name))

        for model_name in os.listdir(os.path.join(test_models, class_name)):
            model_path = os.path.join(test_models, class_name, model_name)
            model = torch.load(model_path)
            model = model.unsqueeze(0)
            model = model.cuda()
            with torch.no_grad():
                model_feature = domain_disentangled_model(rendered_images_p = model)

            # Tuple of (model_feature, class_number)
            model_features.append((model_feature, class_name))

    # We will plot the t-SNE for the images and models together

    # Get the image features and labels
    image_feat = np.array([image_feature[0].cpu().numpy() for image_feature in image_features])
    image_labels = np.array([image_feature[1] for image_feature in image_features])

    image_feat = image_feat.squeeze()

    # Get the model features and labels
    shape_feat = np.array([model_feature[0].cpu().numpy() for model_feature in model_features])
    shape_labels = np.array([model_feature[1] for model_feature in model_features])

    shape_feat = shape_feat.squeeze()

    tsne = TSNE(n_components=2, random_state=0)

    # Concatenate the model and image features
    num_images = image_feat.shape[0]
    features = np.concatenate((image_feat, shape_feat), axis=0)

    # Get the t-SNE embeddings
    tsne_embeddings = tsne.fit_transform(features)
    image_tsne_embeddings = tsne_embeddings[:num_images]
    shape_tsne_embeddings = tsne_embeddings[num_images:]
    
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    custom_palette = sns.color_palette("tab20", n_colors=len(set(classes)))
    
    sns.scatterplot(x=image_tsne_embeddings[:, 0], y=image_tsne_embeddings[:, 1], hue=image_labels, ax=ax, s=50, palette=custom_palette)
    sns.scatterplot(x=shape_tsne_embeddings[:, 0], y=shape_tsne_embeddings[:, 1], hue=shape_labels, ax=ax, marker="^", s=300, palette=custom_palette, legend=False)

    plt.savefig("tsne_4.png")


if __name__ == "__main__":
    main()
