import random
import bittensor as bt


def filter_duplicate_predictions(product_predictions: list, miner_uids: list):
    """
    Filter out miners with same predictions (matching up to 2 decimal places).

    Args:
        product_predictions: List of prediction objects
        miner_uids: List of valid miner IDs to consider

    Returns:
        filtered_predictions, filtered_miners: Lists with duplicates removed
    """
    # First pass: Group predictions by rounded value
    prediction_groups = (
        {}
    )  # Maps rounded prediction -> list of (original_pred, miner_id)

    for pred in product_predictions:
        if pred.miner_id in miner_uids and pred.prediction is not None:
            precision_two_pred = int(pred.prediction * 100) / 100
            if precision_two_pred not in prediction_groups:
                prediction_groups[precision_two_pred] = []
            prediction_groups[precision_two_pred].append(
                (pred, pred.miner_id)
            )

    filtered_predictions = []
    filtered_miners = []

    for precision_two_pred, entries in prediction_groups.items():
        if len(entries) == 1:
            pred, miner_id = entries[0]
        else:
            miner_options = [miner_id for _, miner_id in entries]
            selected_miner = random.choice(miner_options)
            bt.logging.info(
                f"Selected miner {selected_miner} from {miner_options} for prediction {precision_two_pred}"
            )
            for pred, miner_id in entries:
                if miner_id == selected_miner:
                    break

        filtered_predictions.append(pred)
        filtered_miners.append(miner_id)

    return filtered_predictions, filtered_miners
