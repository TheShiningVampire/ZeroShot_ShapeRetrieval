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

warnings.filterwarnings("ignore", category=UserWarning)


# project root setup
# searches for root indicators in parent dirs, like ".git", "pyproject.toml", etc.
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def calculate_metrics(image_features, shape_features):
    # Convert the lists of tuples to separate lists of features and class_numbers
    image_feature_list, image_class_list = zip(*image_features)
    shape_feature_list, shape_class_list = zip(*shape_features)

    # Convert the lists to numpy arrays
    image_features_np = np.vstack(image_feature_list)
    image_classes_np = np.array(image_class_list)
    shape_features_np = np.vstack(shape_feature_list)
    shape_classes_np = np.array(shape_class_list)
    
    # Normalize the features for cosine similarity
    faiss.normalize_L2(image_features_np)
    faiss.normalize_L2(shape_features_np)

    # Build the FAISS index for the shape features
    index = faiss.IndexFlatIP(shape_features_np.shape[1])
    index.add(shape_features_np)

    D, I = index.search(image_features_np, 1)  # D is the array of distances, I is the array of indices

    # Calculate metrics
    binary_ground_truth = (np.expand_dims(image_classes_np, 1) == shape_classes_np[I])

    # For NN precision calculation
    nn_classes = shape_classes_np[I[:, 0]]
    nn_precision = np.mean(image_classes_np == nn_classes)

    metrics = {}

    metrics["NN"] = nn_precision
    # Calculate mAP
    aps = [average_precision_score(gt_row, score_row) for gt_row, score_row in zip(binary_ground_truth, D)]
    metrics["mAP"] = np.mean(aps)
    
    return metrics


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
            image_features.append((image_feature, label_by_number[class_name], image_name))

        for model_name in os.listdir(os.path.join(test_models, class_name)):
            model_path = os.path.join(test_models, class_name, model_name)
            model = torch.load(model_path)
            model = model.unsqueeze(0)
            model = model.cuda()
            with torch.no_grad():
                model_feature = domain_disentangled_model(rendered_images_p = model)

            # Tuple of (model_feature, class_number)
            model_features.append((model_feature, label_by_number[class_name], model_name))

            # Save the number of models per class
            if label_by_number[class_name] not in classwise_model_count:
                classwise_model_count[label_by_number[class_name]] = 1
            else:
                classwise_model_count[label_by_number[class_name]] += 1


    # Retrieval
    # For each image, find k closest models
    k = 3
    acc = 0
    class_acc = 0
    classwise_acc = {}
    classwise_count = {}

    ft = 0
    st = 0
    # Start a retrieval results image
    retrieval_results = torch.zeros((3, 224, 224*4)).cuda()
    

    for (image_feature, image_class, image_name) in image_features:
        pl = 0
        distances = []
        for (model_feature, model_class, model_name) in model_features:
            similarity = torch.cosine_similarity(image_feature, model_feature, dim=1)
            distance = 1 - similarity
            distances.append((distance, model_class, model_name))

        distances.sort(key=lambda x: x[0])
        d = distances.copy()
        distances = distances[:k]

        class_acc = sum([1 for (_, model_class, model_name) in distances if model_class == image_class])/k
        acc += class_acc

        # First Tier metric
        ## In the first classwise_model_count[image_class] models, see how many are correct
        ft += sum([1 for (_, model_class, model_name) in d[:classwise_model_count[image_class]] if model_class == image_class])/classwise_model_count[image_class]

        # # Second Tier metric
        # ## In the first 2* classwise_model_count[image_class] models, see how many are correct
        st += sum([1 for (_, model_class, model_name) in d[:2*classwise_model_count[image_class]] if model_class == image_class])/(classwise_model_count[image_class])


        # # Plot the image and the closest model only if the first model is correct
        # if (distances[0][1] == image_class) and (pl == 0):
        #     pl = 1
        #     image_path = os.path.join(test_images, classes[image_class], image_name)
        #     image = Image.open(image_path).convert("RGB")
        #     # Convert the image to tensor
        #     image = transforms.ToTensor()(image).cuda()

        #     for i in range(3):
        #         model_path = os.path.join(test_models, classes[distances[i][1]], distances[i][2])
        #         model = torch.load(model_path)
        #         rendered_image = model[3, :, :, :]

        #         # Concatenate the image and the rendered image in a row
        #         image = torch.cat((image, rendered_image), dim=2)
            
        #     # Concate the images in a of retrival results
        #     retrieval_results = torch.cat((retrieval_results, image), dim=1)



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
    
    # Save the retrieval results
    # save_image(retrieval_results, "retrieval_results.png", nrow=4)

    acc = acc/len(image_features)

    print("Classwise accuracy:")
    for class_name in classes:
        print(f"{class_name}: {classwise_acc[label_by_number[class_name]]/classwise_count[label_by_number[class_name]]}")

    print(f"NN: {acc}")
    print(f"First Tier Accuracy: {ft/len(image_features)}")
    print(f"Second Tier Accuracy: {st/len(image_features)}")

    # # Use the function
    # image_features = [(image_feature.cpu().numpy(), image_class) for image_feature, image_class in image_features]
    # model_features = [(model_feature.cpu().numpy(), model_class) for model_feature, model_class in model_features]
    # metrics = calculate_metrics(image_features, model_features)
    # print(metrics)

if __name__ == "__main__":
    main()
