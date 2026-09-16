import os
import zipfile
import shutil
import kaggle

def download_and_extract_plantvillage(target_dir="data/plantvillage"):
    print("==================================================")
    print(" Downloading Full 38-Class PlantVillage Dataset (1.27 GB)...")
    print("==================================================")
    
    download_dir = "data/download_temp"
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    try:
        # Download dataset using Kaggle API
        dataset_slug = "mohamedsaber0/plantvillage-dataset"
        print(f"Downloading from Kaggle slug: {dataset_slug}")
        kaggle.api.dataset_download_files(dataset_slug, path=download_dir, unzip=True)
        print("\nDataset download & unzip complete. Organizing class folders...")

        ingested_count = 0
        for root, dirs, files in os.walk(download_dir):
            for d in dirs:
                # Target PlantVillage class directories
                if "___" in d or "healthy" in d.lower():
                    src_folder = os.path.join(root, d)
                    dst_folder = os.path.join(target_dir, d)
                    if not os.path.exists(dst_folder):
                        shutil.move(src_folder, dst_folder)
                        ingested_count += 1
                        print(f"[{ingested_count}] Ingested class folder: {d}")

        print(f"\nPlantVillage ingestion complete! {ingested_count} classes available in: {target_dir}")
        shutil.rmtree(download_dir, ignore_errors=True)

    except Exception as e:
        print(f"[ERROR] Failed downloading PlantVillage dataset: {e}")

if __name__ == "__main__":
    download_and_extract_plantvillage()
