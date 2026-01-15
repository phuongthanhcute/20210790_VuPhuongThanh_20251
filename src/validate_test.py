import multiprocessing
import json
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from combiner_moe import Combiner
from utils import extract_index_features, collate_fn, element_wise_sum, device

# Đánh giá trên test set, có thể không có ground truth, hỗ trợ tạo submission file.

# Tính metrics validation cho FashionIQ.
def compute_fiq_metrics(relative_dataset: FashionIQDataset, clip_model: CLIP, index_features: torch.tensor,
                        index_names: List[str], combining_function: callable, 
                        dress_type: str, save_submission: bool = False) -> Tuple[float, float, dict]:
    """
    Compute metrics or generate predictions for FashionIQ
    """
    
    # Generate predictions
    predicted_features, target_names = generate_fiq_predictions(clip_model, relative_dataset,
                                                               combining_function, index_names, index_features)

    print(f"Processing FashionIQ {relative_dataset.dress_types}...")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the (1-cosine similarity) between predicted features and index features
    distances = 1 - predicted_features @ index_features.T
    
    # Sort indices (getting top matches)
    # Lấy top 50 để tính metrics hoặc lưu submission (sắp xếp theo khoảng cách)
    sorted_indices = torch.argsort(distances, dim=-1)[:, :50].cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Nếu muốn lưu submission (đặc biệt cho tập Test không có nhãn)
    submission_part = {}
    if save_submission:
        # Lấy tên file query (tương đối phức tạp do logic data loader, 
        # ở đây ta dùng target_names làm placeholder hoặc cần sửa loader để trả về query_id)
        # Lưu ý: Code gốc không trả về query_id, nên ta sẽ map theo thứ tự.
        pass # Phần này phụ thuộc vào format submission cụ thể của bạn

    # Nếu là tập Test chuẩn (không có target_names thực sự hoặc target là dummy), 
    # việc tính metrics sẽ sai hoặc lỗi. Ta dùng try/except hoặc cờ kiểm tra.
    
    try:
        # Compute the ground-truth labels wrt the predictions
        labels = torch.tensor(
            sorted_index_names == np.repeat(np.array(target_names), 50).reshape(len(target_names), -1))
        
        # Compute the metrics
        recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
        recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
        print(f"[{dress_type}] R@10: {recall_at10:.2f}, R@50: {recall_at50:.2f}")
    except Exception as e:
        print(f"Could not compute metrics for {dress_type} (likely missing ground truth in Test set). Error: {e}")
        return 0.0, 0.0, {dress_type: sorted_index_names.tolist()}

    # Trả về kết quả dự đoán (để lưu file nếu cần)
    predictions_dict = {dress_type: sorted_index_names.tolist()}
    return recall_at10, recall_at50, predictions_dict

# Tạo predicted features cho FashionIQ validation.
def generate_fiq_predictions(clip_model: CLIP, relative_dataset: FashionIQDataset,
                             combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions
    """
    print(f"Compute FashionIQ {relative_dataset.dress_types} predictions")

    # Data loader
    relative_loader = DataLoader(dataset=relative_dataset, batch_size=32,
                                     num_workers=multiprocessing.cpu_count(), pin_memory=False, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []

    # Duyệt từng batch
    for reference_names, batch_target_names, captions in tqdm(relative_loader):  # Load data
        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True) # tokenize captions với CLIP

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs) # encode captions với CLIP
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0) # lấy features của reference image từ index features
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(name_to_feat))
            
            batch_predicted_features = combining_function(reference_image_features, text_features) # combine features của reference image và captions

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1))) # normalize và lưu predicted features của từng query
        target_names.extend(batch_target_names)

    return predicted_features, target_names # return predicted features và target names


# Wrapper để chạy retrieval trên FashionIQ validation cho một category.
def fashioniq_retrieval(split: str, dress_type: str, combining_function: callable, clip_model: CLIP, preprocess: callable):
    """
    Perform retrieval on FashionIQ (Valid or Test)
    """
    clip_model = clip_model.float().eval() # chuyển model sang float và set thành eval mode

    # Define the datasets and extract the index features
    # Note: Tập index (kho ảnh) thường giống nhau cho cả val và test, 
    # nhưng ta vẫn truyền tham số split vào để đúng logic dataset
    classic_dataset = FashionIQDataset(split, [dress_type], 'classic', preprocess) # load dataset classic
    index_features, index_names = extract_index_features(classic_dataset, clip_model) # trích xuất features index
    
    relative_dataset = FashionIQDataset(split, [dress_type], 'relative', preprocess) # load dataset relative

    return compute_fiq_metrics(relative_dataset, clip_model, index_features, index_names,
                               combining_function, dress_type) # tính metrics

# --- Giữ nguyên các hàm cho CIRR nếu không cần dùng ---

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='FashionIQ', help="should be 'FashionIQ'")
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'], help="Dataset split to use")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=Path, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x16", type=str, help="CLIP model to use")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    
    args = parser.parse_args()

    # Load CLIP Model
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    # Preprocessing
    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    # Load Combiner
    if args.combining_function.lower() == 'sum':
        combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    if args.dataset.lower() == 'fashioniq':
        print(f"Running on FashionIQ {args.split} set...")
        average_recall10_list = []
        average_recall50_list = []
        
        # Dictionary to store predictions if needed
        all_predictions = {}

        categories = ['shirt', 'dress', 'toptee']
        
        for cat in categories:
            r10, r50, preds = fashioniq_retrieval(args.split, cat, combining_function, clip_model, preprocess)
            
            # Chỉ thêm vào list tính trung bình nếu kết quả > 0 (tránh trường hợp test set không tính được)
            if r10 > 0:
                average_recall10_list.append(r10)
                average_recall50_list.append(r50)
            
            all_predictions.update(preds)

        if average_recall10_list:
            print(f"\n--- Overall Results ({args.split}) ---")
            print(f"Average R@10 = {mean(average_recall10_list):.2f}")
            print(f"Average R@50 = {mean(average_recall50_list):.2f}")
        else:
            print("\nCould not calculate metrics (possibly due to missing ground truth in Test set).")
            print("Generated predictions for submission/inspection.")
            
        # Optional: Save predictions to json
        if args.split == 'test':
             with open('fashioniq_test_predictions.json', 'w') as f:
                 # Chuyển đổi numpy/tensor sang list để save json
                 json.dump(all_predictions, f)
             print("Saved predictions to 'fashioniq_test_predictions.json'")

    else:
        # Giữ lại phần CIRR nếu cần
        pass

if __name__ == '__main__':
    main()