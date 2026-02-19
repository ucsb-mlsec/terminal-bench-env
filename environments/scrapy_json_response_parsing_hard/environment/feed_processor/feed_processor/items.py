# -*- coding: utf-8 -*-
"""
Product item definition for feed processing.
This item defines the structure for product data extraction from various feed sources.
"""

import scrapy


class ProductItem(scrapy.Item):
    """Item class for product data extraction from partner feeds."""
    id = scrapy.Field()
    name = scrapy.Field()
    price = scrapy.Field()