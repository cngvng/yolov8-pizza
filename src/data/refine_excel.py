import pandas 
import pandas as pd
import os

images_folder = 'dataset_yolov8x/test'
# images_folder = 'dataset_yolov8x/test'
image_files = [file for file in os.listdir(images_folder) if file.endswith('.jpg')]
df = pd.read_excel('convert_data/data_refine_all_errors.xlsx')

image_files_df = pd.DataFrame({'img_filename': image_files})
filtered_df = df[df['img_url'].str.split('/').str[-1].isin(image_files_df['img_filename'])]
filtered_df.to_excel('dataset_yolov8x/data_refine_all_errors_test.xlsx', index=False)
# filtered_df.to_excel('dataset_yolov8x/data_refine_all_errors_test.xlsx', index=False)