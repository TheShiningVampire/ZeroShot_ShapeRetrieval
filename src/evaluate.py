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

warnings.filterwarnings("ignore", category=UserWarning)


# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def calculate_nn_faiss(image_features, shape_features):
    # Convert the lists of tuples to separate lists of features and class_numbers
    image_feature_list, image_class_list = zip(*image_features)
    shape_feature_list, shape_class_list = zip(*shape_features)

    # Convert the lists to numpy arrays
    image_features_np = np.vstack(image_feature_list)
    image_classes_np = np.array(image_class_list)
    shape_features_np = np.vstack(shape_feature_list)
    shape_classes_np = np.array(shape_class_list)
    
    # Build the FAISS index for the shape features
    # index = faiss.IndexFlatL2(shape_features_np.shape[1])
    # index.add(shape_features_np)

    index = faiss.IndexFlatIP(shape_features_np.shape[1])
    index.add(shape_features_np)

    D, I = index.search(image_features_np, 1)  # D is the array of distances, I is the array of indices

    nn_classes = shape_classes_np[I]

    # Perform the nearest neighbor search
    D, I = index.search(image_features_np, 1)  # D is the array of distances, I is the array of indices

    # Retrieve the class of the most similar shape feature for each image feature
    nn_classes = np.array(shape_class_list)[I]

    # Calculate the NN precision as the proportion of images where the class of the NN shape matches the image class
    # nn_precision = np.mean(image_classes_np == nn_classes)
    nn_precision = sum([1 for image_class, nn_class in zip(image_classes_np, nn_classes) if image_class == nn_class])/len(image_classes_np)

    return nn_precision




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
            image_features.append((image_feature, label_by_number[class_name]))

        for model_name in os.listdir(os.path.join(test_models, class_name)):
            model_path = os.path.join(test_models, class_name, model_name)
            model = torch.load(model_path)
            model = model.unsqueeze(0)
            model = model.cuda()
            with torch.no_grad():
                model_feature = domain_disentangled_model(rendered_images_p = model)

            # Tuple of (model_feature, class_number)
            model_features.append((model_feature, label_by_number[class_name]))

    # Retrieval
    # For each image, find k closest models
    k = 1
    acc = 0
    class_acc = 0
    classwise_acc = {}
    classwise_count = {}
    for image_feature, image_class in image_features:
        distances = []
        for model_feature, model_class in model_features:
            similarity = torch.cosine_similarity(image_feature, model_feature, dim=1)
            distance = 1 - similarity
            distances.append((distance, model_class))

        distances.sort(key=lambda x: x[0])
        distances = distances[:k]

        class_acc = sum([1 for _, model_class in distances if model_class == image_class])/k
        acc += class_acc

        # Save classwise accuracy
        if image_class not in classwise_acc:
            classwise_acc[image_class] = class_acc
        else:
            classwise_acc[image_class] += class_acc

        # Save classwise count
        if image_class not in classwise_count:
            classwise_count[image_class] = 1
        else:
            classwise_count[image_class] += 1
        

    acc = acc/len(image_features)
    print(f"Accuracy: {acc}")

    print("Classwise accuracy:")
    for class_name in classes:
        print(f"{class_name}: {classwise_acc[label_by_number[class_name]]/classwise_count[label_by_number[class_name]]}")


    # Use the function
    image_features = [(image_feature.cpu().numpy(), image_class) for image_feature, image_class in image_features]
    model_features = [(model_feature.cpu().numpy(), model_class) for model_feature, model_class in model_features]
    nn_precision = calculate_nn_faiss(image_features, model_features)
    print(f"NN precision: {nn_precision}")


if __name__ == "__main__":
    main()
