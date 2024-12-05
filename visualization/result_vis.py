import re
import numpy as np


def create_train_id_to_name_mapping(label_dictionary):
    """Create a mapping from train_id to class name, excluding unlabeled class"""
    train_id_to_name = {}
    for class_info in label_dictionary.values():
        if class_info['train_id'] != 255:  # Exclude unlabeled class
            train_id_to_name[class_info['train_id']] = class_info['name']
    return train_id_to_name


def parse_log_file(log_file_path):
    """Parse log file to extract IoU arrays for each epoch"""
    results = {}
    current_epoch = None
    iou_array = None

    with open(log_file_path, 'r') as f:
        epoch_results = f.read().split("Epoch: [")
        for res in epoch_results[1:-1]:
            current_epoch = res.split('/')[0]
            upto_ind = 10
            for r_i, res_line in enumerate(res.split("\n")):
                if "-----------------------------------" in res_line:
                    upto_ind = r_i
                    break

            res_arr = " ".join(res.split("\n")[1:upto_ind]).replace(
                '0 [', '|').replace(
                ']', '|'
            )
            res_arr = re.sub(r'(\d)\.', r'|\1.', res_arr)
            res_arr = res_arr.split('|')[2: -2]
            res_arr = [float(r_e.strip()) for r_e in res_arr]
            assert len(res_arr) == 51
            results[current_epoch] = np.array(res_arr)

        # for line in f:
        #     # Extract epoch number
        #     epoch_match = re.search(r'Epoch: \[(\d+)/500\]', line)
        #     if epoch_match:
        #         current_epoch = int(epoch_match.group(1))
        #
        #     # Extract IoU array
        #     if line.strip().count('0.') > 5:  # Heuristic to identify IoU array lines
        #         try:
        #             # Convert the string of numbers to a numpy array
        #             numbers = re.findall(r'[\d.]+', line)
        #             iou_array = np.array([float(x) for x in numbers[:-1]])  # Exclude the last number which is mean
        #             if current_epoch is not None:
        #                 results[current_epoch] = iou_array
        #         except:
        #             continue

    return results


def analyze_iou_groups(results):
    """Find top 5 epochs with maximum mean IoU for specified groups"""
    epoch_means = []

    for epoch, ious in results.items():
        # Calculate mean for specified groups
        selected_indices = np.concatenate([
            [9, 10],  # sea_anemone and sea_urchin
            np.arange(21, 29),  # coral to dynamic
            np.arange(30, len(ious))  # beaver onwards
        ])

        mean_iou = np.mean(ious[selected_indices])
        epoch_means.append((epoch, mean_iou, ious))

    # Sort by mean IoU and get top 5
    top_5 = sorted(epoch_means, key=lambda x: x[1], reverse=True)[:5]
    return top_5


def print_iou_table(ious, train_id_to_name):
    """Print IoU table for selected classes"""
    print("-----------------------------------")
    print("Class Name   | Train ID |    IoU    ")
    print("-----------------------------------")

    # Get selected indices
    selected_indices = np.concatenate([
        [9, 10],  # sea_anemone and sea_urchin
        np.arange(21, 29),  # coral to dynamic
        np.arange(30, len(ious))  # beaver onwards
    ])

    # Print IoUs for selected indices
    for train_id in selected_indices:
        class_name = train_id_to_name.get(train_id, 'unknown')
        print(f"{class_name:<12} |    {train_id:<5} |  {ious[train_id]:.6f}")

    print("-----------------------------------")
    print(f"Mean IoU     |          |  {np.mean(ious[selected_indices]):.6f}")
    print("-----------------------------------")

label_dictionary = {
    0:  {'name': 'unlabeled',   'train_id': 255, 'color': (0,   0,   0)},
    1:  {'name': 'crab',        'train_id': 0,   'color': (255, 178, 204)},
    2:  {'name': 'crocodile',   'train_id': 1,   'color': (0, 0, 128)},
    3:  {'name': 'dolphin',     'train_id': 2,   'color': (0, 0, 178)},
    4:  {'name': 'frog',        'train_id': 3,   'color': (51, 51, 51)},
    5:  {'name': 'nettles',     'train_id': 4,   'color': (0, 0, 0)},
    6:  {'name': 'octopus',     'train_id': 5,   'color': (51, 306, 51)},
    7:  {'name': 'otter',       'train_id': 6,   'color': (102, 102, 102)},
    8:  {'name': 'penguin',     'train_id': 7,   'color': (10, 0, 255)},
    9:  {'name': 'polar_bear',  'train_id': 8,   'color': (255, 178, 102)},
    10: {'name': 'sea_anemone', 'train_id': 9,  'color':  (153, 255, 255)},
    11: {'name': 'sea_urchin',  'train_id': 10,  'color': (0, 255, 255)},
    12: {'name': 'seahorse',    'train_id': 11,  'color': (255, 153, 153)},
    13: {'name': 'seal',        'train_id': 12,  'color': (255, 0, 0)},
    14: {'name': 'shark',       'train_id': 13,  'color': (178, 178, 0)},
    15: {'name': 'shrimp',      'train_id': 14,  'color': (255, 102, 178)},
    16: {'name': 'star_fish',   'train_id': 15,  'color': (153, 204, 255)},
    17: {'name': 'stingray',    'train_id': 16,  'color': (255, 153, 178)},
    18: {'name': 'squid',       'train_id': 17,  'color': (229, 0, 0)},
    19: {'name': 'turtle',      'train_id': 18,  'color': (0, 153, 0)},
    20: {'name': 'whale',       'train_id': 19,  'color': (0, 229, 77)},
    21: {'name': 'nudibranch',  'train_id': 20,  'color': (242, 243, 245)},
    22: {'name': 'coral',       'train_id': 21,  'color': (0, 0, 77)},
    23: {'name': 'rock',        'train_id': 22,  'color': (0, 178, 0)},
    24: {'name': 'water',       'train_id': 23,  'color': (255, 77, 77)},
    25: {'name': 'sand',        'train_id': 24,  'color': (178, 0, 0)},
    26: {'name': 'plant',       'train_id': 25,  'color': (255, 178, 255)},
    27: {'name': 'human',       'train_id': 26,  'color': (128, 128, 0)},
    28: {'name': 'reef',        'train_id': 27,  'color': (0, 0, 255)},
    29: {'name': 'others',      'train_id': 28,  'color': (178, 178, 178)},
    30: {'name': 'dynamic',     'train_id': 29,  'color': (0, 77, 0)}, ## begining of UWSv2 new semantic categories
    31: {'name': 'beaver',      'train_id': 30,  'color': (151, 191, 201)},
    32: {'name': 'duck',        'train_id': 31,  'color': (153, 102, 51)},
    33: {'name': 'dugong',      'train_id': 32,  'color': (229, 0, 229)},
    34: {'name': 'hippo',       'train_id': 33,  'color': (255, 255, 178)},
    35: {'name': 'lobster',     'train_id': 34,  'color': (222, 128, 4)},
    36: {'name': 'platypus',    'train_id': 35,  'color': (102, 87, 110)},
    37: {'name': 'nautilus',    'train_id': 36,  'color': (229, 229, 0)},
    38: {'name': 'sea_cucumber','train_id': 37,  'color': (229, 255, 255)},
    39: {'name': 'sea_lion',    'train_id': 38,  'color': (173, 173, 0)},
    40: {'name': 'sea_snake',   'train_id': 39,  'color': (0, 0, 102)},
    41: {'name': 'barracouta',  'train_id': 40,  'color': (77, 0, 0)},
    42: {'name': 'billfish',    'train_id': 41,  'color': (170, 184, 90)},
    43: {'name': 'coho',        'train_id': 42,  'color': (174, 230, 187)},
    44: {'name': 'eel',         'train_id': 43,  'color': (0, 178, 178)},
    45: {'name': 'goldfish',    'train_id': 44,  'color': (173, 121, 0)},
    46: {'name': 'jellyfish',   'train_id': 45,  'color': (97, 194, 157)},
    47: {'name': 'lionfish',    'train_id': 46,  'color': (0, 128, 255)},
    48: {'name': 'puffer',      'train_id': 47,  'color': (87, 106, 110)},
    49: {'name': 'rock_beauty', 'train_id': 48,  'color': (142, 173, 0)},
    50: {'name': 'sturgeon',    'train_id': 49,  'color': (27, 71, 74)},
    51: {'name': 'tench',       'train_id': 50,  'color': (209, 88, 88)}
}

def main():
    # Create train_id to name mapping
    train_id_to_name = create_train_id_to_name_mapping(label_dictionary)

    # Parse log file and analyze
    log_file_path = '/Users/ibk5106/Desktop/research/under_water/hrnetv2_sem_seg/log/uws_v2_train_hrnet_v2_WACV25_CAMERA_READY_2024-12-02-15-41_train.log'
    results = parse_log_file(log_file_path)
    top_5_results = analyze_iou_groups(results)

    # Print results for each of the top 5
    for i, (epoch, mean_iou, ious) in enumerate(top_5_results, 1):
        print(f"\n=== Rank {i} ===")
        print(f"Epoch: {epoch}")
        print(f"Mean IoU: {mean_iou:.6f}")
        print("\nDetailed IoUs for this epoch:")
        print_iou_table(ious, train_id_to_name)


if __name__ == "__main__":
    main()


"""
-----------------------------------
Class Name   | Train ID |    IoU    
-----------------------------------
sea_anemone  |    9     |  0.676779 
sea_urchin   |    10    |  0.169259 
coral        |    21    |  0.000334 
rock         |    22    |  0.272836 
water        |    23    |  0.816937 
sand         |    24    |  0.456539 
plant        |    25    |  0.335670 
human        |    26    |  0.393926 
reef         |    27    |  0.365553 
others       |    28    |  0.081310 
beaver       |    30    |  0.844293 
duck         |    31    |  0.563097 
dugong       |    32    |  0.808759 
hippo        |    33    |  0.786891 
lobster      |    34    |  0.884476 
platypus     |    35    |  0.774039 
nautilus     |    36    |  0.915772 
sea_cucumber |    37    |  0.559548 
sea_lion     |    38    |  0.718431 
sea_snake    |    39    |  0.702486 
barracouta   |    40    |  0.704157 
billfish     |    41    |  0.596513 
coho         |    42    |  0.791634 
eel          |    43    |  0.576207 
goldfish     |    44    |  0.726167 
jellyfish    |    45    |  0.805757 
lionfish     |    46    |  0.919011 
puffer       |    47    |  0.707435 
rock_beauty  |    48    |  0.533017 
sturgeon     |    49    |  0.724514 
tench        |    50    |  0.885692 
-----------------------------------
Mean IoU     |          |  0.616033 
-----------------------------------
Loss: 1.316, MeanIU:  0.6160
[0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 6.76779120e-01 1.69258920e-01 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 3.33533453e-04 2.72836426e-01 8.16936718e-01
 4.56539367e-01 3.35670227e-01 3.93925860e-01 3.65553359e-01
 8.13098853e-02 0.00000000e+00 8.44292929e-01 5.63096644e-01
 8.08759404e-01 7.86891377e-01 8.84475665e-01 7.74039253e-01
 9.15771613e-01 5.59548188e-01 7.18430639e-01 7.02486456e-01
 7.04156663e-01 5.96512700e-01 7.91634367e-01 5.76206886e-01
 7.26166530e-01 8.05757123e-01 9.19010677e-01 7.07434974e-01
 5.33017358e-01 7.24513600e-01 8.85691771e-01]
Hours: 0
"""