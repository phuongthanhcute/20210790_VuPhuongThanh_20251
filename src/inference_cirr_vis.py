import json
import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict
import os

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import math

# File dùng để visual kết quả, nhưng không dùng nữa
from combiner_train import extract_index_features
from data_utils import CIRRDataset, targetpad_transform, squarepad_transform, base_path
from combiner_moe import Combiner
from utils import element_wise_sum, device


def visualize_and_save_results(reference_names, captions, sorted_index_names, sorted_group_names, dataset_split, num_vis=5, save_dir="vis_results"):
    """
    Hàm trực quan hóa kết quả: 5 samples, mỗi sample gồm Ref + Top 5 Global + Top 3 Subset.
    Lưu thành 1 file ảnh lớn.
    """
    save_path = Path(base_path) / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Tìm thư mục chứa ảnh
    possible_img_dirs = [
        Path(base_path) / dataset_split,
        Path(base_path) / 'images' / dataset_split,
        Path(base_path) / 'CIRR' / 'images' / dataset_split,
        Path(base_path) / 'CIRR' / dataset_split,
    ]
    
    img_dir = None
    for p in possible_img_dirs:
        if p.exists():
            img_dir = p
            break
            
    if img_dir is None:
        print(f"WARNING: Could not find image directory for split '{dataset_split}'. Visualization will show placeholders.")
        img_dir = Path(base_path)

    print(f"Visualizing {num_vis} samples (Global + Subset) to {save_path}...")

    # Chọn ngẫu nhiên num_vis mẫu để vẽ
    indices = np.random.choice(len(reference_names), min(num_vis, len(reference_names)), replace=False)

    # Tạo figure: số dòng = num_vis, số cột = 9 (1 Ref + 5 Global + 3 Subset)
    # Kích thước: Rộng 28 inch, Cao 4 * số dòng inch
    fig, axes = plt.subplots(len(indices), 9, figsize=(28, 4.5 * len(indices)))
    
    # Xử lý trường hợp chỉ vẽ 1 sample (axes là mảng 1 chiều)
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    # Helper function để load ảnh
    def get_image(name):
        found_paths = list(img_dir.rglob(f"{name}.png")) + list(img_dir.rglob(f"{name}.jpg"))
        if found_paths:
            return Image.open(found_paths[0]).convert("RGB")
        return None

    # Helper để vẽ 1 ô
    def plot_cell(ax, img_name, title, title_color='black', is_ref=False, caption=""):
        img = get_image(img_name)
        if img:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Img Not Found', ha='center', va='center', fontsize=10)
            ax.set_facecolor('#f0f0f0')
        
        # Nếu là ảnh Ref, thêm caption vào title
        if is_ref:
            wrapped_cap = "\n".join([caption[j:j+20] for j in range(0, len(caption), 20)])
            full_title = f"{title}\n{wrapped_cap}"
            ax.set_title(full_title, fontsize=11, fontweight='bold', color='blue', pad=10)
        else:
            ax.set_title(f"{title}\n{img_name}", fontsize=10, color=title_color)
        
        ax.axis("off")

    for row_idx, idx in enumerate(indices):
        ref_name = reference_names[idx]
        caption = captions[idx]
        
        # Lấy Top 5 Global và Top 3 Subset
        global_top = sorted_index_names[idx][:5]
        subset_top = sorted_group_names[idx][:3]

        # 1. Cột đầu tiên: Reference Image
        plot_cell(axes[row_idx, 0], ref_name, "REF + Query", is_ref=True, caption=caption)

        # 2. Cột 2-6: Top 5 Global (Màu đen)
        for k in range(5):
            title = f"Global Top {k+1}" if row_idx == 0 else f"G-{k+1}"
            plot_cell(axes[row_idx, 1 + k], global_top[k], title)

        # 3. Cột 7-9: Top 3 Subset (Màu xanh lá để phân biệt)
        for k in range(3):
            title = f"Subset Top {k+1}" if row_idx == 0 else f"S-{k+1}"
            plot_cell(axes[row_idx, 6 + k], subset_top[k], title, title_color='green')
            
            # Vẽ thêm viền xanh cho nhóm Subset để dễ nhìn
            for spine in axes[row_idx, 6 + k].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2)

    plt.tight_layout()
    
    # Lưu ra 1 file duy nhất
    out_file = save_path / f"summary_global_subset_{dataset_split}.png"
    plt.savefig(out_file, bbox_inches='tight', dpi=100)
    plt.close()
    
    print(f"Visualization saved to: {out_file}")


def generate_cirr_test_submissions(combining_function: callable, file_name: str, clip_model: CLIP,
                                   preprocess: callable):
    """
    Generate and save CIRR test submission files to be submitted to evaluation server
    """
    clip_model = clip_model.float().eval()

    # Define the dataset and extract index features
    SPLIT_NAME = 'test1' 
    classic_test_dataset = CIRRDataset(SPLIT_NAME, 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_test_dataset, clip_model)
    relative_test_dataset = CIRRDataset(SPLIT_NAME, 'relative', preprocess)

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(
        relative_test_dataset, clip_model, index_features, index_names, combining_function, split_name=SPLIT_NAME
    )

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    submissions_folder_path = base_path / "submission" / 'CIRR'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions")
    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(relative_test_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable, split_name: str) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for CIRR dataset
    """

    # Generate predictions
    predicted_features, reference_names, group_members, pairs_id, all_captions = \
        generate_cirr_test_predictions(clip_model, relative_test_dataset, combining_function, index_names,
                                     index_features)

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    
    # Compute the subset predictions (MOVE UP FOR VISUALIZATION)
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # --- VISUALIZATION SECTION ---
    try:
        # Pass cả sorted_index_names (Global) và sorted_group_names (Subset)
        visualize_and_save_results(reference_names, all_captions, sorted_index_names, sorted_group_names, 
                                   dataset_split=split_name, num_vis=5)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Skipping visualization due to error: {e}")
    # -----------------------------

    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(clip_model: CLIP, relative_test_dataset: CIRRDataset, combining_function: callable,
                                   index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[List[str]], List[str], List[str]]:
    """
    Compute CIRR predictions on the test set
    """
    print(f"Compute CIRR test predictions")

    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=multiprocessing.cpu_count(), pin_memory=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    group_members = []
    reference_names = []
    all_captions = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(
            relative_test_loader):  # Load data
        text_inputs = clip.tokenize(captions, context_length=77).to(device)
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)
        all_captions.extend(captions)

    return predicted_features, reference_names, group_members, pairs_id, all_captions


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, required=True, help="submission file name")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=str, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    args = parser.parse_args()
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if args.clip_model_path:
        print('Trying to load the CLIP model')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device)
        saved_state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(saved_state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path should be in ['sum', 'combiner']")

    generate_cirr_test_submissions(combining_function, args.submission_name, clip_model, preprocess)


if __name__ == '__main__':
    main()