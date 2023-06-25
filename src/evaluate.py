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


def compute_ap(query_feature, query_label, model_features):
    similarities = []
    labels = []

    for model_feature in model_features:
        similarity = torch.cosine_similarity(query_feature, model_feature[0]).cpu().numpy()
        is_same_class = int(query_label == model_feature[1])
        
        similarities.append(similarity)
        labels.append(is_same_class)

    # Sorting both lists (similarities and labels) by similarity in descending order
    similarities, labels = zip(*sorted(zip(similarities, labels), reverse=True))
    
    return average_precision_score(labels, similarities)


def compute_map(image_features, model_features):
    ap_values = []

    for image_feature in image_features:
        ap = compute_ap(image_feature[0], image_feature[1], model_features)
        ap_values.append(ap)

    return np.mean(ap_values)

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2)))
    return 0.

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
    k = 1
    acc = 0
    class_acc = 0
    classwise_acc = {}
    classwise_count = {}

    ft = 0
    st = 0

    reciprocal_rank = []
    relevance_scores = []
    r_k = 10
    # Start a retrieval results image
    retrieval_results = torch.zeros((3, 224, 224*(k+1))).cuda()
    

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


        # Find rank of the first correct model
        for i in range(len(d)):
            if d[i][1] == image_class:
                reciprocal_rank.append(1/(i+1))
                break
            
        # Computing relevance scores
        relevance_scores.append([1 if d[i][1] == image_class else 0 for i in range(r_k)])

        # # Plot the image and the closest model only if the first model is correct
        # if (distances[0][1] == image_class) and (pl == 0):
        #     pl = 1
        #     image_path = os.path.join(test_images, classes[image_class], image_name)
        #     image = Image.open(image_path).convert("RGB")
        #     # Convert the image to tensor
        #     image = transforms.ToTensor()(image).cuda()

        #     for i in range(min(k, len(distances))):
        #         model_path = os.path.join(test_models, classes[distances[i][1]], distances[i][2])
        #         model = torch.load(model_path)
        #         rendered_image = model[3, :, :, :]

        #         # Concatenate the image and the rendered image in a row
        #         image = torch.cat((image, rendered_image), dim=2)
            
        #     # Concate the images in a of retrival results
        #     retrieval_results = torch.cat((retrieval_results, image), dim=1)

        #     if (len(distances) < k):
        #         for i in range(k - len(distances)):
        #             retrieval_results = torch.cat((retrieval_results, torch.ones((3, 224, 224)).cuda()), dim=1)



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
    # save_image(retrieval_results, "retrieval_results_seen_classes.png", nrow=6)

    acc = acc/len(image_features)

    mAP = compute_map(image_features, model_features)

    mrr = np.mean(reciprocal_rank)

    # Computing DCG
    dcg_k = [ dcg_at_k(r, k) for r in relevance_scores ]
    avg_dcg = np.mean(dcg_k)

    print("Classwise accuracy:")
    for class_name in classes:
        print(f"{class_name}: {classwise_acc[label_by_number[class_name]]/classwise_count[label_by_number[class_name]]}")

    # Save the classwise accuracy in a csv file

    # # create the csv file
    # with open("classwise_accuracy_seen.csv", "w") as f:
    #     for class_name in classes:
    #         f.write(f"{class_name},{classwise_acc[label_by_number[class_name]]/classwise_count[label_by_number[class_name]]}\n")

    print(f"NN: {acc}")
    print(f"First Tier Accuracy: {ft/len(image_features)}")
    print(f"Second Tier Accuracy: {st/len(image_features)}")
    print(f"E: {mrr}")
    print(f"DCG: {avg_dcg}")
    print(f"mAP: {mAP}")

if __name__ == "__main__":
    main()
