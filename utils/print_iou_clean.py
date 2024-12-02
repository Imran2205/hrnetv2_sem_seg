

def print_selected_iou(iou_list, label_dict, selected_ids, logger):
    """
    Print IoU values for selected classes in a table format.

    Args:
        iou_list (list): List of IoU values where index corresponds to train_id
        label_dict (dict): Dictionary containing class information
        selected_ids (list): List of train_ids to display
        logger: Logger instance for output
    """
    # Calculate max length of class names for formatting
    max_name_length = max(len(label_dict[k]['name'])
                          for k in label_dict
                          if label_dict[k]['train_id'] in selected_ids)

    # Print header
    header = f"{'Class Name':<{max_name_length}} | {'Train ID':^8} | {'IoU':^10}"
    separator = "-" * (max_name_length + 23)

    logger.info(separator)
    logger.info(header)
    logger.info(separator)

    # Track valid IoU values for mean calculation
    valid_ious = []

    # Print values for selected classes
    for k, v in label_dict.items():
        if v['train_id'] in selected_ids:
            train_id = v['train_id']
            iou_value = iou_list[train_id]
            valid_ious.append(iou_value)

            logger.info(f"{v['name']:<{max_name_length}} | {train_id:^8d} | {iou_value:^10.6f}")

    # Print footer with mean
    logger.info(separator)
    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0
    logger.info(f"{'Mean IoU':<{max_name_length}} | {'':^8} | {mean_iou:^10.6f}")
    logger.info(separator)

    return mean_iou
