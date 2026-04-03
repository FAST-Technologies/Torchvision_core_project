import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path

parquet_path: str = 'coco-panoptic-val2017/data/train-00001-of-00002-94aa8570497c415e.parquet'
output_dir_path: str = 'coco-panoptic-val2017/images/test'
mask_dir_path: str = 'coco-panoptic-val2017/annotations/test'

pq_file = Path(parquet_path)
df: pd.DataFrame = pd.read_parquet(pq_file)

output_dir: Path = Path(output_dir_path)
mask_dir: Path = Path(mask_dir_path)
output_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)

for idx, row in df.iterrows():
    # Изображение
    img_data = row['image']
    if isinstance(img_data, dict) and 'bytes' in img_data:
        img = Image.open(BytesIO(img_data['bytes'])).convert('RGB')
        img.save(output_dir / f'coco-panoptic-val2017_{idx:06d}1.jpg')
    
    # Маска
    if 'label' in row and row['label'] is not None:
        mask_data = row['label']
        if isinstance(mask_data, dict) and 'bytes' in mask_data:
            mask = Image.open(BytesIO(mask_data['bytes'])).convert('L')
            mask.save(mask_dir / f'coco-panoptic-val2017_{idx:06d}1.png')
    
    if idx % 100 == 0:
        print(f'Processed {idx}/{len(df)}')

print('✅ Done!')