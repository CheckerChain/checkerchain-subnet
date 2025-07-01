# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import bittensor as bt
import numpy as np
import traceback
import json
import jwt

from checkerchain.protocol import CheckerChainSynapse

from checkerchain.database.actions import (
    add_prediction,
    get_predictions_for_product,
    delete_a_product,
    db_get_unreviewd_products,
)
from checkerchain.validator.reward import get_rewards
from neurons.validator import Validator
from checkerchain.utils.checker_chain import fetch_products
from checkerchain.utils.config import IS_OWNER, STATS_SERVER_URL, JWT_SECRET
import requests
from checkerchain.utils.uids import get_filtered_uids
from checkerchain.utils.filter_miners import filter_duplicate_predictions
from checkerchain.miner.llm import create_llm
from langchain.schema import SystemMessage, HumanMessage


async def forward(self: Validator):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    """
    miner_uids = get_filtered_uids(self)
    data = fetch_products()
    products_to_score = []
    if len(data.reward_items):
        products_to_score = [r._id for r in data.reward_items]

    if len(data.unmined_products):
        queries = data.unmined_products
    else:
        unmined_db_products = db_get_unreviewd_products()
        queries = [p._id for p in unmined_db_products]

    responses = []
    if len(queries):
        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=CheckerChainSynapse(query=queries),
            timeout=25,
            deserialize=True,
        )
        bt.logging.info(f"Received responses: {responses}")

        for miner_uid, miner_predictions in zip(miner_uids, responses):
            for product_idx, prediction in enumerate(miner_predictions):
                score = (
                    prediction.get("score") if isinstance(prediction, dict) else None
                )
                review = (
                    prediction.get("review") if isinstance(prediction, dict) else None
                )
                keywords = (
                    prediction.get("keywords", [])
                    if isinstance(prediction, dict)
                    else []
                )

                if keywords:
                    if len(keywords) < 3:
                        bt.logging.warning(
                            f"Miner {miner_uid}: Too few keywords ({len(keywords)}): {keywords}"
                        )
                    elif len(keywords) > 7:
                        bt.logging.warning(
                            f"Miner {miner_uid}: Too many keywords ({len(keywords)}): {keywords}"
                        )

        for miner_uid, miner_predictions in zip(miner_uids, responses):
            for product_id, prediction in zip(queries, miner_predictions):
                if product_id not in products_to_score:
                    add_prediction(
                        product_id=product_id,
                        miner_id=miner_uid,
                        prediction_data=(
                            prediction
                            if isinstance(prediction, dict)
                            else {"score": prediction}
                        ),
                    )
    else:
        pass

    reward_product = None
    predictions = []
    miner_ids = np.array(miner_uids)
    rewards = np.zeros_like(miner_ids, dtype=float)
    if data.reward_items:
        prediction_logs = []
        for reward_product in data.reward_items:
            product_predictions = get_predictions_for_product(reward_product._id) or []
            if not product_predictions:
                continue

            predictions, prediction_miners = filter_duplicate_predictions(
                product_predictions, miner_ids.tolist()
            )

            # Skip if no valid predictions after filtering
            if not prediction_miners:
                bt.logging.warning(
                    f"No valid predictions found for product {reward_product._id}"
                )
                continue

            _rewards = await get_rewards(
                self,
                reward_product,
                responses=predictions,
                miner_uids=prediction_miners,
            )
            bt.logging.info(f"Product ID: {reward_product._id}")
            bt.logging.info(f"Miners: {prediction_miners}")
            bt.logging.info(f"Rewards: {_rewards}")
            if _rewards is None:
                continue

            for i, (miner_id, reward, prediction) in enumerate(
                zip(prediction_miners, _rewards, predictions)
            ):
                if reward is None:
                    continue
                try:
                    if not prediction:
                        bt.logging.warning(
                            f"Prediction score is None for miner {int(miner_id)} and product {reward_product._id}"
                        )
                        continue

                    analysis_data = {
                        "total_reward": reward,
                    }

                    # Update the prediction with analysis results
                    add_prediction(
                        product_id=reward_product._id,
                        miner_id=miner_id,
                        prediction_data={"score": prediction.prediction},
                        analysis_data=analysis_data,
                    )

                    # Find the index of the miner in the miner_ids array
                    miner_indices = np.where(miner_ids == miner_id)[0]
                    if len(miner_indices) == 0:
                        bt.logging.warning(
                            f"Miner {miner_id} not found in miner_ids array"
                        )
                        continue
                    idx = miner_indices[0]

                    prediction_logs.append(
                        {
                            "productId": reward_product._id,
                            "productName": reward_product.name,
                            "productSlug": reward_product.slug,
                            "predictionScore": prediction.prediction,
                            "actualScore": reward_product.trustScore,
                            "hotkey": self.metagraph.hotkeys[miner_id],
                            "coldkey": self.metagraph.coldkeys[miner_id],
                            "review": prediction.review,
                            "keywords": (
                                json.loads(prediction.keywords)
                                if prediction.keywords is not None
                                else []
                            ),
                            "sentiment": prediction.sentiment,
                            "uid": int(miner_id),
                        }
                    )
                    rewards[idx] += reward
                except Exception as e:
                    tb = traceback.format_exc()
                    bt.logging.error(
                        f"Error while processing product {reward_product._id}:\n{tb}"
                    )
                    continue

        try:
            # You don't need to worry about this part of the code, it's for data collection for owners
            if IS_OWNER and JWT_SECRET and len(prediction_logs) > 0:
                token = jwt.encode(
                    {"sub": self.metagraph.coldkeys[0]},
                    JWT_SECRET,
                    algorithm="HS256",
                )
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                }
                bt.logging.info(f"{STATS_SERVER_URL}/prediction/create", "url:")
                result = requests.post(
                    f"{STATS_SERVER_URL}/prediction/create",
                    json=prediction_logs,
                    headers=headers,
                )
                if result.status_code != 201:
                    bt.logging.error(
                        f"Error sending data to stats server: {result.status_code}"
                    )
                else:
                    bt.logging.info("Successfully sent data to stats server")
        except Exception as e:
            bt.logging.error(f"Error while sending data to stats server: {e}")
            bt.logging.error(f"Prediction logs: {prediction_logs}")

        bt.logging.info(f"Scored responses: {rewards}")
        bt.logging.info(f"Score ids: {miner_ids}")

        mask = rewards > 0
        filtered_rewards = rewards[mask]
        filtered_miner_ids = miner_ids[mask]  # Now miner_ids is a numpy array
        self.update_scores(filtered_rewards, filtered_miner_ids.tolist())
        for reward_product in data.reward_items:
            delete_a_product(reward_product._id)
    else:
        self.update_to_last_scores()

    # 25 mins until next validation ??
    time.sleep(25 * 60)
