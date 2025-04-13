# utils.py
import os
import pandas as pd

def load_partition_bbox(data_dir):
    """
    Tải dữ liệu phân vùng và bounding box từ các tệp CSV.
    
    Args:
        data_dir (str): Đường dẫn đến thư mục chứa dữ liệu (Data/)
    
    Returns:
        train_df, val_df, test_df: Các DataFrame cho tập huấn luyện, tập xác thực và tập kiểm tra
    """
    # Đường dẫn đến các tệp CSV
    partition_path = os.path.join(data_dir, "list_eval_partition.csv")
    bbox_path = os.path.join(data_dir, "list_bbox_celeba.csv")

    # Đọc tệp phân vùng
    partition_df = pd.read_csv(partition_path, sep=r",", skiprows=1, header=None, names=["image_id", "partition"])
    print("Tổng số mẫu trong partition_df:", len(partition_df))
    print("Mẫu image_id trong partition_df:", partition_df["image_id"].iloc[0])

    # Đọc tệp bounding box
    bbox_df = pd.read_csv(bbox_path, sep=r",", skiprows=1, header=None, names=["image_id", "x_1", "y_1", "width", "height"])
    print("Tổng số mẫu trong bbox_df:", len(bbox_df))
    print("Mẫu image_id trong bbox_df:", bbox_df["image_id"].iloc[0])

    # Gộp partition_df và bbox_df dựa trên image_id
    merged_df = partition_df.merge(bbox_df, on="image_id")
    print("Tổng số mẫu sau khi gộp:", len(merged_df))

    # Tách thành các tập train, val, test
    train_df = merged_df[merged_df["partition"] == 0]
    val_df = merged_df[merged_df["partition"] == 1]
    test_df = merged_df[merged_df["partition"] == 2]

    return train_df, val_df, test_df