from dataclasses import dataclass
from typing import List
import requests
import bittensor as bt

from checkerchain.database.actions import add_product, get_products
from checkerchain.types.checker_chain import (
    ReviewedProduct,
    ReviewedProductsApiResponse,
    UnreviewedProductApiResponse,
    UnreviewedProductsApiResponse,
)


@dataclass
class FetchProductsReturnType:
    unmined_products: List[str]
    reward_items: List[ReviewedProduct]


def fetch_products():
    # Reviewed products
    url_reviewed = "https://api.checkerchain.com/api/v1/products?page=1&limit=30"
    url_unreviewed = "https://api.checkerchain.com/api/v1/products?page=1&limit=1"
    # Unreviewed (published) products
    # url_unreviewed = (
    #     "https://api.checkerchain.com/api/v1/products?page=1&limit=30&status=published"
    # )

    response_reviewed = requests.get(url_reviewed)
    response_unreviewed = requests.get(url_unreviewed)

    if response_reviewed.status_code != 200:
        bt.logging.error(
            f"Error fetching reviewed products: {response_reviewed.status_code}"
        )
        return FetchProductsReturnType([], [])

    if response_unreviewed.status_code != 200:
        bt.logging.errorint(
            f"Error fetching unreviewed products: {response_unreviewed.status_code}"
        )
        return FetchProductsReturnType([], [])

    reviewed_response = ReviewedProductsApiResponse.from_dict(response_reviewed.json())
    unreviewed_response = UnreviewedProductsApiResponse.from_dict(
        response_unreviewed.json()
    )

    if not isinstance(reviewed_response, ReviewedProductsApiResponse) or not isinstance(
        unreviewed_response, UnreviewedProductsApiResponse
    ):
        return FetchProductsReturnType([], [])

    reviewed_products = reviewed_response.data.products
    unreviewed_products = unreviewed_response.data.products

    # Fetch existing product IDs from the database
    all_products = get_products()
    existing_product_ids = {p._id for p in all_products}
    unmined_products: List[str] = []
    reward_items: List[ReviewedProduct] = []

    # Process unreviewed products (newly published ones)
    for product in unreviewed_products:
        if product._id not in existing_product_ids:
            add_product(product._id, product.name)
            unmined_products.append(product._id)

    # Process reviewed products (existing ones for reward)
    for product in reviewed_products:
        if product._id in existing_product_ids:
            reward_items.append(product)

    return FetchProductsReturnType(unmined_products, reward_items)


def fetch_product_data(product_id):
    """Fetch product data from the API using the product ID."""
    url = f"https://backend.checkerchain.com/api/v1/products/{product_id}"
    response = requests.get(url)
    if response.status_code == 200:
        productData = UnreviewedProductApiResponse.from_dict(response.json())
        if not (isinstance(productData, UnreviewedProductApiResponse)):
            return None
        return productData.data
    else:
        bt.logging.error(
            "Error fetching product data:", response.status_code, response.text
        )
        return None
