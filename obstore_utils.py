"""Utilities for working with Zarr data using the obstore backend."""

import logging

import boto3
from obstore.auth.boto3 import Boto3CredentialProvider
from obstore.store import S3Store
from zarr.storage import ObjectStore

LOGGER = logging.getLogger(__name__)


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

    if "/" in location:
        bucket, prefix = location.split("/", 1)
        if not prefix:
            prefix = None
        elif not prefix.endswith("/"):
            prefix += "/"
    else:
        bucket, prefix = location, None

    session_extra_args = {}
    if profile is not None:
        session_extra_args["profile_name"] = profile

    # Start a session, discover the location of the bucket requested, and then init a new
    # session which sets the region so that obstore can know how to perform authentication
    # properly.
    session = boto3.Session(**session_extra_args)
    s3 = session.client("s3")
    region = s3.get_bucket_location(Bucket=bucket)["LocationConstraint"]
    session = boto3.Session(**session_extra_args, region_name=region)

    s3store_extra_args = {}
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
