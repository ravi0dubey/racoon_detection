import fiftyone as fo
from fiftyone import ViewField as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def get_filepath(sample):
    return (
        sample.local_path if hasattr(sample, "local_path") else sample.filepath
    )


def gen_approx_duplicate_groups_view(dataset, index):
    """
    This function is used to generate the approximate duplicate groups view.
    """

    dup_ids = index.duplicates_view().values("id")
    view = dataset.select(dup_ids)

    for rep_id, dups in index.neighbors_map.items():
        ids = [rep_id] + [d[0] for d in dups]
        subview = view.select(ids)
        for sample in subview:
            sample["approx_dup_group_id"] = rep_id
            sample.save()

    approx_dup_groups_view = view.group_by("approx_dup_group_id")
    dataset.save_view(
        "approx_dup_groups_view", approx_dup_groups_view, overwrite=True
    )


def find_approximate_duplicates(
    sample_collection, brain_key, threshold=None, fraction=None
):
    dataset = sample_collection._dataset

    index = dataset.load_brain_results(brain_key)
    if threshold is not None:
        index.find_duplicates(thresh=threshold)
    else:
        index.find_duplicates(fraction=fraction)

    ### save the full duplicates view
    approx_dup_view = index.duplicates_view()
    dataset.save_view("approx_dup_view", approx_dup_view, overwrite=True)
    approx_dup_view = dataset.load_saved_view("approx_dup_view")

    ### save the approximate duplicate groups view
    gen_approx_duplicate_groups_view(dataset, index)

    ### compute the number of images with duplicates
    num_images_with_approx_dups = len(approx_dup_view)
    num_approx_dup_groups = len(index.neighbors_map)
    num_dups = num_images_with_approx_dups - num_approx_dup_groups

    response = {
        "num_images_with_approx_dups": num_images_with_approx_dups,
        "num_dups": num_dups,
    }

    return response


def get_approximate_duplicate_groups(sample_collection):
    dataset = sample_collection._dataset
    approx_dup_view = dataset.load_saved_view("approx_dup_groups_view")
    return approx_dup_view


def remove_all_approximate_duplicates(sample_collection):
    dataset = sample_collection._dataset

    if "approx_dup_view" not in dataset.list_saved_views():
        raise ValueError("Approximate duplicates have not been computed yet.")

    approx_dup_view = dataset.load_saved_view("approx_dup_view")
    dataset.delete_samples(approx_dup_view.values("id"))

    ## remove the saved views
    dataset.delete_saved_view("approx_dup_view")
    dataset.delete_saved_view("approx_dup_groups_view")


def deduplicate_approximate_duplicates(sample_collection):
    dataset = sample_collection._dataset

    # Check if approximate duplicates view exists
    if "approx_dup_view" not in dataset.list_saved_views():
        raise ValueError("Approximate duplicates have not been computed yet.")

    # Load the approximate duplicates view
    approx_dup_view = dataset.load_saved_view("approx_dup_view")

    # Initialize list to collect IDs of samples to be removed
    remove_sample_ids = []
    
    # Get distinct group IDs
    group_ids = approx_dup_view.distinct("approx_dup_group_id")
    
    # Log the number of groups (unique images)
    num_groups = len(group_ids)
    logging.info(f"Number of unique images (groups): {num_groups}")

    # Iterate over each distinct group of approximate duplicates
    for group_id in group_ids:
        # Get the view of samples in the current group
        group_view = approx_dup_view.match(
            F("approx_dup_group_id") == group_id
        )
        
        # Sort the group by file path
        group_view = group_view.sort_by("filepath")
        
        # Collect IDs of all samples except the first one in the sorted group
        remove_sample_ids.extend(group_view.values("id")[1:])

    # Delete the collected samples from the dataset
    dataset.delete_samples(remove_sample_ids)

    # Remove the saved views
    dataset.delete_saved_view("approx_dup_view")
    dataset.delete_saved_view("approx_dup_groups_view")