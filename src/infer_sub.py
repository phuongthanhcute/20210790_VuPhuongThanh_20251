import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple
import shutil
import json

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset, base_path
from combiner_moe import Combiner
from utils import extract_index_features, collate_fn, element_wise_sum, device

# Lưu kết quả validation thành các folder riêng cho từng case
def save_results_to_folders(reference_names, target_names, captions, sorted_index_names, 
                            sorted_group_names, dataset_name, dataset_split, num_cases=20, 
                            save_dir="validation_results"):
    """
    Lưu kết quả validation thành các folder riêng cho từng case
    Mỗi folder chứa: Reference + Expected + Global Top 5 + Subset Top 3
    """
    save_path = Path(base_path) / save_dir / f"{dataset_name}_{dataset_split}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"DEBUG: base_path = {base_path}")
    print(f"DEBUG: dataset_name = {dataset_name}, dataset_split = {dataset_split}")
    
    # Tìm thư mục chứa ảnh dựa trên dataset
    if dataset_name == "CIRR":
        possible_img_dirs = [
            Path(base_path) / dataset_split,
            Path(base_path) / 'images' / dataset_split,
            Path(base_path) / 'CIRR' / 'images' / dataset_split,
            Path(base_path) / 'CIRR' / dataset_split,
            Path(base_path).parent / 'cirr_dataset' / dataset_split,
        ]
    else:  # FashionIQ
        possible_img_dirs = [
            Path(base_path) / 'fashionIQ_dataset' / 'images',
            Path(base_path) / 'images',
            Path(base_path).parent / 'fashionIQ_dataset' / 'images',
            Path(base_path) / 'FashionIQ' / 'images',
        ]
    
    img_dir = None
    for p in possible_img_dirs:
        print(f"DEBUG: Checking {p} ... exists={p.exists()}")
        if p.exists():
            img_dir = p
            break
            
    if img_dir is None:
        print(f"WARNING: Could not find image directory for {dataset_name} {dataset_split}. Will save structure without images.")
        img_dir = Path(base_path)

    print(f"Saving {num_cases} cases to {save_path}...")

    # Chọn ngẫu nhiên num_cases mẫu
    indices = np.random.choice(len(reference_names), min(num_cases, len(reference_names)), replace=False)

    # Helper function để tìm và copy ảnh
    def copy_image(img_name, dest_path):
        # Tìm file với nhiều extension
        found_paths = (list(img_dir.rglob(f"{img_name}.png")) + 
                      list(img_dir.rglob(f"{img_name}.jpg")) +
                      list(img_dir.rglob(f"{img_name}.jpeg")))
        if found_paths:
            shutil.copy2(found_paths[0], dest_path)
            return True
        return False

    # Lưu thông tin tổng hợp
    global_info = []

    for case_idx, idx in enumerate(indices, start=1):
        ref_name = reference_names[idx]
        target_name = target_names[idx]
        caption = captions[idx] if isinstance(captions[idx], str) else " and ".join(captions[idx])
        
        # Lấy Top 5 Global và Top 3 Subset (nếu có)
        global_top = sorted_index_names[idx][:5]
        subset_top = sorted_group_names[idx][:3] if sorted_group_names is not None else []

        # Tạo folder cho case này
        case_folder = save_path / f"case_{case_idx}"
        case_folder.mkdir(exist_ok=True)
        
        # Tạo subfolder
        global_folder = case_folder / "Global_Retrieved"
        global_folder.mkdir(exist_ok=True)
        
        if len(subset_top) > 0:
            subset_folder = case_folder / "Subset_Retrieved"
            subset_folder.mkdir(exist_ok=True)

        # 1. Copy Reference Image
        ref_basename = ref_name.split('/')[-1]
        ref_dest = case_folder / f"0_Reference_{ref_basename}.png"
        if not copy_image(ref_name, ref_dest):
            print(f"WARNING: Could not find reference image: {ref_name}")
        
        # 2. Copy Expected/Target Image
        target_basename = target_name.split('/')[-1]
        target_dest = case_folder / f"1_Expected_{target_basename}.png"
        if not copy_image(target_name, target_dest):
            print(f"WARNING: Could not find target image: {target_name}")
        
        # 3. Copy Global Retrieved Top 5
        for k, img_name in enumerate(global_top, start=1):
            img_basename = img_name.split('/')[-1]
            dest = global_folder / f"Rank{k}_{img_basename}.png"
            if not copy_image(img_name, dest):
                print(f"WARNING: Could not find global image: {img_name}")
        
        # 4. Copy Subset Retrieved Top 3 (chỉ cho CIRR)
        for k, img_name in enumerate(subset_top, start=1):
            img_basename = img_name.split('/')[-1]
            dest = subset_folder / f"Rank{k}_{img_basename}.png"
            if not copy_image(img_name, dest):
                print(f"WARNING: Could not find subset image: {img_name}")

        # 5. Lưu thông tin case
        case_info = {
            "case": case_idx,
            "reference": ref_name,
            "target": target_name,
            "caption": caption,
            "global_top5": global_top.tolist(),
        }
        if len(subset_top) > 0:
            case_info["subset_top3"] = subset_top.tolist()
        
        global_info.append(case_info)

    # Lưu file global_info.txt
    with open(save_path / "global_info.txt", "w", encoding="utf-8") as f:
        json.dump(global_info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to: {save_path}")
    print(f"✓ Created {len(indices)} case folders\n")


# Tính toán các chỉ số đánh giá trên tập validation FashionIQ
def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, clip_model: CLIP, index_features: torch.tensor,
                            index_names: List[str], combining_function: callable, save_visualization: bool = False,
                            split_name: str = 'val') -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    """
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} {split_name} metrics")

    # Chuẩn hoá các features index
    index_features = F.normalize(index_features, dim=-1).float()
    
    # Generate predictions incrementally
    predicted_features, reference_names, target_names, captions, sorted_index_names = \
        generate_fiq_val_predictions_incremental(
            clip_model, relative_val_dataset, combining_function, index_names, index_features,
            save_visualization, split_name
        )

    # Tính toán các nhãn ground truth dựa trên các dự đoán
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Tính toán các chỉ số đánh giá
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50


# Tính toán các dự đoán trên tập validation FashionIQ
def generate_fiq_val_predictions_incremental(clip_model: CLIP, relative_val_dataset: FashionIQDataset,
                                             combining_function: callable, index_names: List[str], 
                                             index_features: torch.tensor, save_visualization: bool,
                                             split_name: str) -> Tuple[torch.tensor, List[str], List[str], List, np.ndarray]:
    """
    Compute FashionIQ predictions incrementally to save memory
    """
    print(f"Tính toán dự đoán FashionIQ {relative_val_dataset.dress_types} {split_name}")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=0, pin_memory=False, collate_fn=collate_fn,
                                     shuffle=False)

    # Lấy mapping từ các tên index đến các features index
    name_to_feat = dict(zip(index_names, index_features))

    # Khởi tạo các danh sách để lưu trữ
    all_reference_names = []
    all_target_names = []
    all_captions = []
    all_sorted_indices = []
    
    # Để tính toán các chỉ số đánh giá
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)

    # Thiết lập thư mục để lưu trữ kết quả nếu cần
    if save_visualization:
        save_path = Path(base_path) / "validation_results" / f"FashionIQ_{'-'.join(relative_val_dataset.dress_types)}_{split_name}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Tìm thư mục chứa ảnh
        possible_img_dirs = [
            Path(base_path) / 'fashionIQ_dataset' / 'images',
            Path(base_path) / 'images',
            Path(base_path).parent / 'fashionIQ_dataset' / 'images',
            Path(base_path) / 'FashionIQ' / 'images',
        ]
        img_dir = None
        for p in possible_img_dirs:
            if p.exists():
                img_dir = p
                break
        
        case_counter = 1
        max_cases = 20

    for batch_reference_names, batch_target_names, captions in tqdm(relative_val_loader):
        # Ghép các captions
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True)

        # Tính toán các dự đoán features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))
            batch_predicted_features = combining_function(reference_image_features, text_features)
            batch_predicted_features = F.normalize(batch_predicted_features, dim=-1)

        # Tính toán các khoảng cách cho batch này
        distances = 1 - batch_predicted_features @ index_features.T
        sorted_indices = torch.argsort(distances, dim=-1).cpu()
        batch_sorted_names = np.array(index_names)[sorted_indices]
        
        # Lưu trữ kết quả nếu cần
        if save_visualization and case_counter <= max_cases and img_dir:
            for i in range(len(batch_reference_names)):
                if case_counter > max_cases:
                    break
                    
                case_folder = save_path / f"case_{case_counter}"
                case_folder.mkdir(exist_ok=True)
                global_folder = case_folder / "Global_Retrieved"
                global_folder.mkdir(exist_ok=True)
                
                # Sao chép ảnh
                def copy_image(img_name, dest_path):
                    found_paths = (list(img_dir.rglob(f"{img_name}.png")) + 
                                  list(img_dir.rglob(f"{img_name}.jpg")) +
                                  list(img_dir.rglob(f"{img_name}.jpeg")))
                    if found_paths:
                        shutil.copy2(found_paths[0], dest_path)
                        return True
                    return False
                
                ref_name = batch_reference_names[i]
                target_name = batch_target_names[i]
                caption = input_captions[i]
                top5 = batch_sorted_names[i][:5]
                
                # Sao chép ảnh reference
                copy_image(ref_name, case_folder / f"0_Reference_{ref_name.split('/')[-1]}.png")
                # Sao chép ảnh target
                copy_image(target_name, case_folder / f"1_Expected_{target_name.split('/')[-1]}.png")
                # Sao chép ảnh top 5
                for k, img_name in enumerate(top5, start=1):
                    copy_image(img_name, global_folder / f"Rank{k}_{img_name.split('/')[-1]}.png")
                
                # Lưu trữ thông tin case
                case_info = {
                    "case": case_counter,
                    "reference": ref_name,
                    "target": target_name,
                    "caption": caption,
                    "global_top5": top5.tolist()
                }
                with open(case_folder / "info.json", "w", encoding="utf-8") as f:
                    json.dump(case_info, f, indent=2, ensure_ascii=False)
                
                case_counter += 1

        # Tích lũy cho các chỉ số đánh giá
        predicted_features = torch.vstack((predicted_features, batch_predicted_features))
        all_reference_names.extend(batch_reference_names)
        all_target_names.extend(batch_target_names)
        all_captions.extend(input_captions)
        all_sorted_indices.append(sorted_indices)
        
        # Giải phóng bộ nhớ
        del batch_predicted_features, distances, sorted_indices
        torch.cuda.empty_cache()

    # Ghép tất cả các chỉ số đã sắp xếp
    all_sorted_indices = torch.cat(all_sorted_indices, dim=0)
    final_sorted_names = np.array(index_names)[all_sorted_indices]
    
    if save_visualization:
        # Lưu trữ thông tin global
        global_info = []
        for i in range(min(max_cases, len(all_reference_names))):
            global_info.append({
                "case": i + 1,
                "reference": all_reference_names[i],
                "target": all_target_names[i],
                "caption": all_captions[i],
                "global_top5": final_sorted_names[i][:5].tolist()
            })
        with open(save_path / "global_info.txt", "w", encoding="utf-8") as f:
            json.dump(global_info, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {min(max_cases, len(all_reference_names))} cases to {save_path}\n")

    return predicted_features, all_reference_names, all_target_names, all_captions, final_sorted_names


# Truy xuất trên tập validation FashionIQ
def fashioniq_val_retrieval(dress_types: List[str], combining_function: callable, clip_model: CLIP, 
                           preprocess: callable, split_name: str = 'val', save_visualization: bool = False):
    """
    Thực hiện truy xuất trên tập validation/test FashionIQ và tính toán các chỉ số đánh giá
    """
    clip_model = clip_model.float().eval()

    # Xác định các datasets và trích xuất các features index
    classic_dataset = FashionIQDataset(split_name, dress_types, 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_dataset, clip_model)
    relative_dataset = FashionIQDataset(split_name, dress_types, 'relative', preprocess)

    return compute_fiq_val_metrics(relative_dataset, clip_model, index_features, index_names,
                                   combining_function, save_visualization, split_name)


# Tính toán các chỉ số đánh giá trên tập validation CIRR
def compute_cirr_val_metrics(relative_val_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable, save_visualization: bool = False) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Tính toán các chỉ số đánh giá trên tập validation CIRR
    """
    # Tính toán các dự đoán
    predicted_features, reference_names, target_names, group_members, captions = \
        generate_cirr_val_predictions(clip_model, relative_val_dataset, combining_function, index_names, index_features)

    print("Tính toán các chỉ số đánh giá trên tập validation CIRR")

    # Chuẩn hoá các features index
    index_features = F.normalize(index_features, dim=-1).float()

    # Tính toán các khoảng cách và sắp xếp kết quả
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Xóa ảnh reference khỏi kết quả
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Tính toán các nhãn ground truth dựa trên các dự đoán
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Tính toán các dự đoán subset và nhãn ground truth
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    # --- Lưu trữ kết quả ---
    if save_visualization:
        try:
            save_results_to_folders(
                reference_names, target_names, captions, sorted_index_names, 
                sorted_group_names,
                dataset_name="CIRR", 
                dataset_split="val", 
                num_cases=20
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Skipping visualization due to error: {e}")

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Tính toán các chỉ số đánh giá
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


# Tính toán các dự đoán trên tập validation CIRR
def generate_cirr_val_predictions(clip_model: CLIP, relative_val_dataset: CIRRDataset,
                                  combining_function: callable, index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]], List[str]]:
    """
    Tính toán các dự đoán trên tập validation CIRR
    """
    print("Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=0,
                                     pin_memory=False, collate_fn=collate_fn)

    # Lấy mapping từ các tên index đến các features index
    name_to_feat = dict(zip(index_names, index_features))

    # Khởi tạo các danh sách để lưu trữ
    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []
    all_captions = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(relative_val_loader):
        text_inputs = clip.tokenize(captions).to(device, non_blocking=True)
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Tính toán các dự đoán features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(name_to_feat))
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        all_captions.extend(captions)

    return predicted_features, reference_names, target_names, group_members, all_captions


# Truy xuất trên tập validation CIRR
def cirr_val_retrieval(combining_function: callable, clip_model: CLIP, preprocess: callable, 
                      save_visualization: bool = False):
    """
    Thực hiện truy xuất trên tập validation CIRR và tính toán các chỉ số đánh giá
    """
    clip_model = clip_model.float().eval()

    # Xác định các datasets và trích xuất các features index
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)

    return compute_cirr_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                    combining_function, save_visualization)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="nên là 'CIRR' hoặc 'fashionIQ' hoặc 'both'")
    parser.add_argument("--combining-function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner-path", type=Path, help="path to trained Combiner")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-visualization", action='store_true', help="Save visualization results to folders")

    args = parser.parse_args()

    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    if args.clip_model_path:
        print('Đang cố gắng tải mô hình CLIP')
        saved_state_dict = torch.load(args.clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('Mô hình CLIP đã được tải thành công')

    if args.transform == 'targetpad':
        print('Pipeline tiền xử lý TargetPad được sử dụng')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Pipeline tiền xử lý SquarePad được sử dụng')
        preprocess = squarepad_transform(input_dim)
    else:
        print('Pipeline tiền xử lý mặc định của CLIP được sử dụng')
        preprocess = clip_preprocess

    if args.combining_function.lower() == 'sum':
        if args.combiner_path:
            print("Be careful, you are using the element-wise sum as combining_function but you have also passed a path"
                  " to a trained Combiner. Such Combiner will not be used")
        combining_function = element_wise_sum
    elif args.combining_function.lower() == 'combiner':
        combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
        state_dict = torch.load(args.combiner_path, map_location=device)
        combiner.load_state_dict(state_dict["Combiner"])
        combiner.eval()
        combining_function = combiner.combine_features
    else:
        raise ValueError("combiner_path phải là: ['sum', 'combiner']")

    # Chạy validation với hiển thị kết quả
    if args.dataset.lower() == 'cirr':
        print("\n" + "="*60)
        print("Chạy validation CIRR")
        print("="*60)
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(combining_function, clip_model, preprocess, args.save_visualization)

        print(f"\n{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")

    elif args.dataset.lower() == 'fashioniq':
        print("\n" + "="*60)
        print("Chạy validation và test FashionIQ")
        print("="*60)
        
        # FashionIQ Val
        print("\n--- tập validation FashionIQ ---")
        val_recall10_list = []
        val_recall50_list = []
        
        for dress_type in ['shirt', 'dress', 'toptee']:
            r10, r50 = fashioniq_val_retrieval([dress_type], combining_function, clip_model,
                                              preprocess, 'val', args.save_visualization)
            val_recall10_list.append(r10)
            val_recall50_list.append(r50)
            print(f"{dress_type}: recall@10={r10:.2f}, recall@50={r50:.2f}")
        
        print(f"\nValidation Average: recall@10={mean(val_recall10_list):.2f}, recall@50={mean(val_recall50_list):.2f}")
        
        # FashionIQ Test
        print("\n--- Tập test FashionIQ ---")
        test_recall10_list = []
        test_recall50_list = []
        
        for dress_type in ['shirt', 'dress', 'toptee']:
            r10, r50 = fashioniq_val_retrieval([dress_type], combining_function, clip_model,
                                              preprocess, 'test', args.save_visualization)
            test_recall10_list.append(r10)
            test_recall50_list.append(r50)
            print(f"{dress_type}: recall@10={r10:.2f}, recall@50={r50:.2f}")
        
        print(f"\nTest Average: recall@10={mean(test_recall10_list):.2f}, recall@50={mean(test_recall50_list):.2f}")

    elif args.dataset.lower() == 'both':
        # Chạy cả CIRR và FashionIQ
        print("\n" + "="*60)
        print("Chạy validation CIRR")
        print("="*60)
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(combining_function, clip_model, preprocess, args.save_visualization)

        print(f"\n{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")
        
        print("\n" + "="*60)
        print("Chạy validation và test FashionIQ")
        print("="*60)
        
        # FashionIQ Val
        print("\n--- tập validation FashionIQ ---")
        val_recall10_list = []
        val_recall50_list = []
        
        for dress_type in ['shirt', 'dress', 'toptee']:
            r10, r50 = fashioniq_val_retrieval([dress_type], combining_function, clip_model,
                                              preprocess, 'val', args.save_visualization)
            val_recall10_list.append(r10)
            val_recall50_list.append(r50)
            print(f"{dress_type}: recall@10={r10:.2f}, recall@50={r50:.2f}")
        
        print(f"\nValidation Average: recall@10={mean(val_recall10_list):.2f}, recall@50={mean(val_recall50_list):.2f}")
        
        # FashionIQ Test
        print("\n--- Tập test FashionIQ ---")
        test_recall10_list = []
        test_recall50_list = []
        
        for dress_type in ['shirt', 'dress', 'toptee']:
            r10, r50 = fashioniq_val_retrieval([dress_type], combining_function, clip_model,
                                              preprocess, 'test', args.save_visualization)
            test_recall10_list.append(r10)
            test_recall50_list.append(r50)
            print(f"{dress_type}: recall@10={r10:.2f}, recall@50={r50:.2f}")
        
        print(f"\nTest Average: recall@10={mean(test_recall10_list):.2f}, recall@50={mean(test_recall50_list):.2f}")
    
    else:
        raise ValueError("Dataset phải là: 'CIRR', 'fashionIQ', hoặc 'both'")


if __name__ == '__main__':
    main()