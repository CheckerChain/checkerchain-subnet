import asyncio

from neurons.miner import Miner
import checkerchain
from checkerchain.miner.llm import (
    generate_complete_assessment,
)
from checkerchain.utils.checker_chain import fetch_batch_product_data
import bittensor as bt

miner_preds = {}


async def forward(self: Miner, synapse: checkerchain.protocol.CheckerChainSynapse):
    """
    Asynchronously fetch product data and generate complete assessments in parallel.
    Uses a single OpenAI request per product to generate score, review, and keywords.
    """
    bt.logging.info(f"Received mine requests for products {synapse.query}")

    responses = [None] * len(synapse.query)  # Placeholder for response dicts
    
    # Separate cached and uncached products
    uncached_product_ids = []
    uncached_indices = []
    
    for i, product_id in enumerate(synapse.query):
        if product_id in miner_preds:
            bt.logging.info(
                f"Using cached prediction for {product_id}: {miner_preds[product_id]}"
            )
            cached_data = miner_preds[product_id]
            responses[i] = cached_data
        else:
            uncached_product_ids.append(product_id)
            uncached_indices.append(i)
    
    # Fetch uncached products in batch
    if uncached_product_ids:
        bt.logging.info(f"Fetching batch product data for {len(uncached_product_ids)} products")
        products = fetch_batch_product_data(uncached_product_ids)
        
        # Create mapping from product_id to product data
        product_map = {product._id: product for product in products}
        
        # Generate assessments for found products
        tasks = []
        task_mappings = []  # (index_in_responses, product_id)
        
        for idx, product_id in enumerate(uncached_product_ids):
            response_index = uncached_indices[idx]
            
            if product_id in product_map:
                product = product_map[product_id]
                tasks.append(generate_complete_assessment(product))
                task_mappings.append((response_index, product_id))
            else:
                bt.logging.warning(f"Product not found for {product_id}")
                responses[response_index] = {"score": None, "review": None, "keywords": []}
        
        # Execute all assessment tasks in parallel
        if tasks:
            bt.logging.info(f"Running {len(tasks)} OpenAI assessment tasks...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for task_index, result in enumerate(results):
                response_index, product_id = task_mappings[task_index]
                try:
                    if isinstance(result, Exception):
                        raise result

                    miner_preds[product_id] = result
                    responses[response_index] = result

                    bt.logging.info(
                        f"Complete assessment for product {product_id}: Score={result['score']}, Keywords={result['keywords']}, Review={result['review']}"
                    )
                except Exception as e:
                    bt.logging.error(f"Error assessing product {product_id}: {e}")
                    responses[response_index] = {"score": None, "review": None, "keywords": []}

    synapse.response = responses
    return synapse
