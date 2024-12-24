import xml.etree.ElementTree as ET
import glob
import os
from tqdm import tqdm

dataset_path = 'labelImg Output'
output_base = 'YOLO_labels'


def convert_xml_to_yolo(xml_path, class_names):
    """Convert single XML file to YOLO format"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image dimensions
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)

    result = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        # Get class index from class_names list
        class_id = class_names.index(class_name)

        # Get bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # Convert to YOLO format (normalized coordinates)
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        # Format: class_id x_center y_center width height
        result.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return result


def create_yolo_dataset():
    class_names = [folder for folder in os.listdir(dataset_path)
                   if os.path.isdir(os.path.join(dataset_path, folder))]

    print(f"Found classes: {class_names}")

    os.makedirs(output_base, exist_ok=True)

    # Create classes.txt
    with open(os.path.join(output_base, 'classes.txt'), 'w') as f:
        f.write('\n'.join(class_names))

    # Process each class
    for class_name in class_names:
        print(f"\nProcessing class: {class_name}")

        # Create output directory for this class
        os.makedirs(os.path.join(output_base, class_name), exist_ok=True)

        # Get all XML files for this class
        xml_files = glob.glob(os.path.join(dataset_path, class_name, '*.xml'))

        # Process each XML file
        for xml_file in tqdm(xml_files, desc=f'Converting {class_name}'):
            try:
                # Convert XML to YOLO format
                yolo_annotations = convert_xml_to_yolo(xml_file, class_names)

                # Get base filename
                base_name = os.path.splitext(os.path.basename(xml_file))[0]

                # Save YOLO format annotation
                output_path = os.path.join(output_base, class_name, f'{base_name}.txt')
                with open(output_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))

            except Exception as e:
                print(f"Error processing {xml_file}: {str(e)}")


if __name__ == "__main__":
    # Create YOLO format dataset
    create_yolo_dataset()
    print("\nConversion completed!")

    total_files = sum([len(files) for r, d, files in os.walk(output_base)])
    print(f"\nTotal converted files: {total_files}")
    print(f"Output directory: {os.path.abspath(output_base)}")
