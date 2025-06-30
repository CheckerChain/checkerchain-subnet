import asyncio

from neurons.miner import Miner
import checkerchain
from checkerchain.miner.llm import (
    generate_complete_assessment,
)
from checkerchain.utils.checker_chain import fetch_product_data
import bittensor as bt

miner_preds = {}


async def forward(self: Miner, synapse: checkerchain.protocol.CheckerChainSynapse):
    """
    Asynchronously fetch product data and generate complete assessments in parallel.
    Uses a single OpenAI request per product to generate score, review, and keywords.
    """
    bt.logging.info(f"Received mine requests for products {synapse.query}")

    tasks = []
    product_ids = []
    responses = [None] * len(synapse.query)  # Placeholder for response dicts
    results = []
    for i, product_id in enumerate(synapse.query):
        if product_id in miner_preds:
            bt.logging.info(
                f"Using cached prediction for {product_id}: {miner_preds[product_id]}"
            )
            cached_data = miner_preds[product_id]
            responses[i] = cached_data
        else:
            product = fetch_product_data(product_id)
            if product:
                product_ids.append((i, product_id))  # To map back later
                tasks.append(generate_complete_assessment(product))
            else:
                bt.logging.warning(f"Product not found for {product_id}")
                responses[i] = {"score": None, "review": None, "keywords": []}
    if len(tasks) > 0:
        bt.logging.info("Running OpenAI assessment tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        tasks = []

    if len(results) > 0:
        for task_index, result in enumerate(results):
            i, product_id = product_ids[task_index]
            try:
                if isinstance(result, Exception):
                    raise result

                miner_preds[product_id] = result
                responses[i] = result

                bt.logging.info(
                    f"Complete assessment for product {product_id}: Score={result['score']}, Keywords={result['keywords']}, Review={result['review']}"
                )
            except Exception as e:
                bt.logging.error(f"Error assessing product {product_id}: {e}")
                responses[i] = {"score": None, "review": None, "keywords": []}

    synapse.response = responses
    return synapse
