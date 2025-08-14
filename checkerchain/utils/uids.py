import random
import bittensor as bt
import numpy as np
from typing import List
import copy
from checkerchain.database.actions import get_blacklisted_miners_hotkeys


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_uids(self, k: int, exclude: List[int] = None) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids))
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = np.array(random.sample(available_uids, k))
    return uids


def get_filtered_uids(self, max_per_key: int = 15) -> np.ndarray:
    blacklisted_hotkeys = get_blacklisted_miners_hotkeys()
    blacklisted_hotkeys_set = {t[0] for t in blacklisted_hotkeys}
    bt.logging.info(
        f"Blacklisted hotkeys: {blacklisted_hotkeys_set}, count: {len(blacklisted_hotkeys_set)}"
    )
    hotkeys = copy.deepcopy(self.metagraph.hotkeys)
    coldkeys = copy.deepcopy(self.metagraph.coldkeys)
    counts = {}
    available_uids = []

    for i, (ck, hk) in enumerate(zip(coldkeys, hotkeys)):
        if hk in blacklisted_hotkeys_set:
            continue
        cnt = counts.get(ck, 0)
        if cnt < max_per_key:
            if check_uid_availability(
                self.metagraph, i, self.config.neuron.vpermit_tao_limit
            ):
                available_uids.append(i)
            counts[ck] = cnt + 1

    uids = np.array(available_uids)
    return uids
