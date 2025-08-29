from dataclasses import dataclass
from typing import List
import requests
import bittensor as bt

from checkerchain.database.actions import (
    add_product,
    get_products,
    remove_bulk_products,
)
from checkerchain.types.checker_chain import (
    ReviewedProduct,
    ReviewedProductsApiResponse,
    UnreviewedProductApiResponse,
    UnreviewedProductsApiResponse,
    Category,
)

from .misc import dict_to_namespace


@dataclass
class FetchProductsReturnType:
    unmined_products: List[str]
    reward_items: List[ReviewedProduct]
    orphaned_products: List[str]


def fetch_products():
    # Reviewed products
    url_reviewed = "https://api.checkerchain.com/api/v1/products?page=1&limit=30"
    # Unreviewed (published) products
    url_unreviewed = (
        "https://api.checkerchain.com/api/v1/products?page=1&limit=30&status=published"
    )

    response_reviewed = requests.get(url_reviewed)
    response_unreviewed = requests.get(url_unreviewed)

    if response_reviewed.status_code != 200:
        bt.logging.error(
            f"Error fetching reviewed products: {response_reviewed.status_code}"
        )
        return [], [], []

    if response_unreviewed.status_code != 200:
        bt.logging.error(
            f"Error fetching unreviewed products: {response_unreviewed.status_code}"
        )
        return [], [], []

    reviewed_response = ReviewedProductsApiResponse.from_dict(response_reviewed.json())
    unreviewed_response = UnreviewedProductsApiResponse.from_dict(
        response_unreviewed.json()
    )
    if not (
        hasattr(reviewed_response, "data")
        and hasattr(unreviewed_response, "data")
        and hasattr(reviewed_response.data, "products")
        and hasattr(unreviewed_response.data, "products")
    ):
        bt.logging.error("Invalid API response structure.")
        return FetchProductsReturnType([], [], [])

    reviewed_products = reviewed_response.data.products
    unreviewed_products = unreviewed_response.data.products

    # Fetch existing product IDs from the database
    all_products = get_products()
    existing_product_ids = {p._id for p in all_products}
    unmined_products: List[str] = []
    reward_items = []

    api_product_ids = set()

    # Process unreviewed products (newly published ones)
    for product in unreviewed_products:
        api_product_ids.add(product._id)
        if product._id not in existing_product_ids:
            add_product(product._id, product.name)
            unmined_products.append(product._id)

    # Process reviewed products (existing ones for reward)
    for product in reviewed_products:
        api_product_ids.add(product._id)
        if product._id in existing_product_ids:
            reward_items.append(
                dict_to_namespace(
                    {
                        "_id": product._id,
                        "name": product.name,
                        "trustScore": product.trustScore,
                        "slug": product.slug,
                        "status": product.status,
                    }
                )
            )

    # Find and remove orphaned products (in DB but not in API)
    orphaned_product_ids = existing_product_ids - api_product_ids
    if orphaned_product_ids:
        orphaned_list = list(orphaned_product_ids)
        bt.logging.info(f"Removing orphaned products: {orphaned_list}")
        remove_bulk_products(orphaned_list)

    return unmined_products, reward_items, list(orphaned_product_ids)


def fetch_product_data(product_id):
    """Fetch product data from the API using the product ID."""
    url = f"https://api.checkerchain.com/api/v1/products/{product_id}"
    response = requests.get(url)
    if response.status_code == 200:
        productData = UnreviewedProductApiResponse.from_dict(response.json())
        if hasattr(productData, "data"):
            return dict_to_namespace(
                {
                    "_id": productData.data._id,
                    "name": productData.data.name,
                    "url": productData.data.url,
                    "description": productData.data.description,
                    "category": productData.data.category,
                }
            )
    else:
        bt.logging.error(
            "Error fetching product data:", response.status_code, response.text
        )
        return None


def fetch_batch_product_data(product_ids):
    """
    Fetch product data for a batch of product IDs.
    
    Args:
        product_ids (list): A list of product IDs to fetch data for.

    Returns:
        list: A list of product data for the specified product IDs.
    """
    url = "https://mvp-backend-testnet.checkerchain.com/api/v1/products/subnet/batch"
    body = {"productIds": product_ids}
    
    try:
        response = requests.post(url, json=body)
        if response.status_code == 200:
            products = []
            for product in response.json().get("data", []):
                category = None
                if product.get("category"):
                    category = Category.from_dict(product.get("category"))
                
                products.append(dict_to_namespace({
                    "_id": product.get("_id"),
                    "name": product.get("name"),
                    "url": product.get("url"),
                    "description": product.get("description"),
                    "category": category,
                }))
            return products
        else:
            bt.logging.error(
                f"Error fetching batch product data: {response.status_code} {response.text}"
            )
            return []
    except Exception as e:
        bt.logging.error(f"Exception while fetching batch product data: {e}")
        return []
