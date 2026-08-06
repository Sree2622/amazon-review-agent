"""Build the processed product catalog used by the review agent."""

from __future__ import annotations

import logging

import polars as pl

from configs.config import PRODUCT_METADATA_PATH, RAW_DATA_DIR


logger = logging.getLogger(__name__)


class ProductMetadataLoader:
    """Convert the raw All_Beauty metadata JSONL into a product catalog."""

    RAW_METADATA_PATH = RAW_DATA_DIR / "meta_All_Beauty.jsonl"
    CATALOG_COLUMNS = [
        "parent_asin",
        "title",
        "store",
        "price",
        "average_rating",
        "rating_number",
        "features",
        "description",
        "categories",
        "images",
    ]

    def ensure_catalog(self) -> pl.LazyFrame:
        """Return the catalog, building it only when the raw file is newer."""

        if not self.RAW_METADATA_PATH.exists():
            raise FileNotFoundError(f"Product metadata not found: {self.RAW_METADATA_PATH}")
        if (
            not PRODUCT_METADATA_PATH.exists()
            or PRODUCT_METADATA_PATH.stat().st_mtime < self.RAW_METADATA_PATH.stat().st_mtime
        ):
            self._build_catalog()
        return pl.scan_parquet(PRODUCT_METADATA_PATH)

    def _build_catalog(self) -> None:
        """Extract product fields from raw metadata and write a Parquet catalog."""

        logger.info("Building product catalog from %s", self.RAW_METADATA_PATH)
        metadata = pl.read_ndjson(self.RAW_METADATA_PATH)
        catalog = (
            metadata.select(self.CATALOG_COLUMNS)
            .filter(pl.col("parent_asin").is_not_null())
            .unique(subset=["parent_asin"], keep="first")
        )
        PRODUCT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        catalog.write_parquet(PRODUCT_METADATA_PATH)
        logger.info("Wrote %d products to %s", catalog.height, PRODUCT_METADATA_PATH)
