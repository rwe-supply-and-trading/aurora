"""Copyright (c) RWE Supply & Trading GmbH. Licensed under the MIT license.

Utilities for working with Zarr data using the obstore backend.
"""

import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError
from obstore.auth.boto3 import Boto3CredentialProvider
from obstore.store import S3Store
from zarr.storage import ObjectStore

LOGGER = logging.getLogger(__name__)


def create_bucket_if_not_exists(
    bucket: str, *, profile: str | None = None, region: str | None = None
) -> None:
    """Create an S3 bucket if it does not already exist.

    Args:
        bucket: The name of the bucket.
        profile: (Optional) An AWS profile name.
        region: (Optional) Region in which to create the bucket. Defaults to the
            session's configured region, or ``us-east-1`` if none is set.
    """
    session_extra_args: dict[str, Any] = {}
    if profile is not None:
        session_extra_args["profile_name"] = profile
    session = boto3.Session(**session_extra_args)
    s3 = session.client("s3")

    try:
        s3.head_bucket(Bucket=bucket)
        LOGGER.info("Bucket %s already exists", bucket)
        return
    except ClientError as err:
        code = err.response.get("Error", {}).get("Code")
        if code not in {"404", "NoSuchBucket"}:
            raise

    region = region or session.region_name or "us-east-1"
    create_kwargs: dict = {"Bucket": bucket}
    if region != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}

    s3.create_bucket(**create_kwargs)
    LOGGER.info("Created bucket %s in %s", bucket, region)


def open_s3_zarr_store(
    location: str, *, profile: str | None = None, read_only: bool = False
) -> ObjectStore:
    """Open a Zarr store using the obstore backend.

    Args:
        location: An S3 path.
        profile: (Optional) An AWS profile name.
        read_only: Zarr store read-only flag.

    Returns:
        A Zarr store.
    """
    if location.startswith("s3://"):
        location = location[5:]

    prefix: str | None
    if "/" in location:
        bucket, prefix = location.split("/", 1)
        if not prefix:
            prefix = None
        elif not prefix.endswith("/"):
            prefix += "/"
    else:
        bucket, prefix = location, None

    session_extra_args: dict[str, Any] = {}
    if profile is not None:
        session_extra_args["profile_name"] = profile

    # Start a session, discover the location of the bucket requested, and then init a new
    # session which sets the region so that obstore can know how to perform authentication
    # properly.
    session = boto3.Session(**session_extra_args)
    s3 = session.client("s3")
    region = s3.get_bucket_location(Bucket=bucket)["LocationConstraint"]
    session = boto3.Session(**session_extra_args, region_name=region)

    s3store_extra_args: dict[str, Any] = {}
    if profile == "kafou":
        s3store_extra_args["endpoint_url"] = "http://kafou-storage.ai-lab.energy.local/"
        s3store_extra_args["virtual_hosted_style_request"] = False
        s3store_extra_args["client_options"] = {"allow_http": True}
    store = S3Store(
        bucket,
        prefix=prefix,
        credential_provider=Boto3CredentialProvider(session),
        **s3store_extra_args,
    )
    return ObjectStore(store, read_only=read_only)
