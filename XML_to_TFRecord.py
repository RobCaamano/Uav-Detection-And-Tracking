import os
import xml.etree.ElementTree as ET
import tensorflow as tf
from PIL import Image
import io
from tqdm import tqdm
import random

# Get data from XML files
def parse_xml(xml_files):
    data = []
    for file in tqdm(xml_files, desc="Parsing XML"):
        tree = ET.parse(file)
        root = tree.getroot()
        img_data = {'filename': root.find('filename').text,
                      'size': {'width': int(root.find('size/width').text),
                               'height': int(root.find('size/height').text),
                               'depth': int(root.find('size/depth').text)},
                      'objects': []}
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            img_data['objects'].append({
                'name': obj.find('name').text,
                'bbox': [int(bbox.find('xmin').text),
                         int(bbox.find('ymin').text),
                         int(bbox.find('xmax').text),
                         int(bbox.find('ymax').text)]
            })
        data.append(img_data)
    return data

# Find the correct image file
def find_img_file(base_path, filename):
    for ext in ['.jpg', '.jpeg', '.png']:
        path = os.path.join(base_path, filename.split('.')[0] + ext)
        if os.path.exists(path):
            return path
    return None

# Create TFRecord example
def create_tf_example(data, label_map, image_dir):
    image_filename = data['filename']
    image_path = find_img_file(image_dir, image_filename)

    # Error Handling
    if image_path is None:
        print(f"File not found for: {image_filename}")
        return None
    try:
        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_jpg = fid.read()
    except Exception as e:
        print(f"Error reading file {image_path}: {e}")
        return None
    
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for obj in data['objects']:
        xmins.append(float(obj['bbox'][0]) / width)
        xmaxs.append(float(obj['bbox'][2]) / width)
        ymins.append(float(obj['bbox'][1]) / height)
        ymaxs.append(float(obj['bbox'][3]) / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map[obj['name']])

    # Determine the image format
    image_format = b'jpg' if image_path.lower().endswith('.jpg') else b'png'

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data['filename'].encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example

# Write TFRecord file
def write_tfrecord(data, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for item in tqdm(data, desc="Creating TFRecord"):
            tf_example = create_tf_example(item, label_map, training_dir)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())

# Training Path
training_dir = 'C:/Users/caama/Documents/School/NJIT/CS370/UAV Detection/training'

# Get XML files and parse data
xml_files = [os.path.join(training_dir, x) for x in os.listdir(training_dir) if x.endswith('.xml')]
parsed_data = parse_xml(xml_files)

# Shuffle and split data
random.shuffle(parsed_data)
split_index = int(0.8 * len(parsed_data))
train_data = parsed_data[:split_index]
val_data = parsed_data[split_index:]

label_map = {'drone': 1}

# Output Paths
train_output_path = 'C:/Users/caama/Documents/School/NJIT/CS370/UAV Detection/train_output.tfrecord'
val_output_path = 'C:/Users/caama/Documents/School/NJIT/CS370/UAV Detection/val_output.tfrecord'

write_tfrecord(train_data, train_output_path)
write_tfrecord(val_data, val_output_path)

print(f"Training and Validation TFRecord files created at {train_output_path} and {val_output_path}")