"""Cloud data access: object storage (AWS S3, Google Cloud Storage, Azure Blob / ADLS Gen2) and warehouses
(Amazon Redshift / Athena, Google BigQuery, Azure SQL / Synapse) for live sources and custom datasets.

Credentials are never stored by the dashboard. It relies on each cloud's standard credential chain:
    AWS    environment variables, ~/.aws (aws configure / SSO, named profiles), instance / container roles
    GCP    GOOGLE_APPLICATION_CREDENTIALS or a service-account key file, `gcloud auth application-default login`
    Azure  a connection string / account key read from an environment variable, or DefaultAzureCredential
           (az login, managed identity, AZURE_CLIENT_ID/SECRET/TENANT_ID)
Saved settings are non-secret only (profile, region, key-file path, account name, endpoint, env-var *names*).
Secrets can go in data/cloud.env (KEY=value lines; git-ignored, never shown in the UI) and be referenced from SQL URLs
as ${VAR}, e.g. postgresql+psycopg://analyst:${REDSHIFT_PASSWORD}@cluster.xxxx.redshift.amazonaws.com:5439/dev
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from . import config

SCHEMES = {"s3": "aws", "s3a": "aws", "gs": "gcp", "gcs": "gcp", "az": "azure", "abfs": "azure", "abfss": "azure", "adl": "azure"}
PACKAGES = {"aws": "s3fs", "gcp": "gcsfs", "azure": "adlfs"}
ENV_FILE = lambda: config.DATA_DIR / "cloud.env"

# warehouse URL presets (SQLAlchemy); shown in the UI to build a URL
WAREHOUSES = {
    "redshift": {"provider": "aws", "label": "Amazon Redshift",
                 "template": "postgresql+psycopg://USER:${REDSHIFT_PASSWORD}@CLUSTER.xxxx.REGION.redshift.amazonaws.com:5439/DATABASE",
                 "help": "Redshift speaks the PostgreSQL protocol. Put the password in data/cloud.env as REDSHIFT_PASSWORD."},
    "athena": {"provider": "aws", "label": "Amazon Athena",
               "template": "awsathena+rest://@athena.REGION.amazonaws.com:443/DATABASE?s3_staging_dir=s3://BUCKET/athena-results/&work_group=primary",
               "help": "Uses your AWS credentials (profile / environment / role); query results are written to the staging bucket."},
    "bigquery": {"provider": "gcp", "label": "Google BigQuery",
                 "template": "bigquery://PROJECT/DATASET",
                 "help": "Uses Application Default Credentials, or add ?credentials_path=/path/to/key.json. Tables are DATASET.TABLE."},
    "azure_sql": {"provider": "azure", "label": "Azure SQL / Synapse",
                  "template": "mssql+pymssql://USER:${AZURE_SQL_PASSWORD}@SERVER.database.windows.net:1433/DATABASE",
                  "help": "Put the password in data/cloud.env as AZURE_SQL_PASSWORD. Synapse dedicated pools use the same form."},
}


# ============================================================================= secrets file / ${VAR}
def load_env_file() -> list[str]:
    """Load data/cloud.env into the process environment (existing variables win). Returns the variable names."""
    f = ENV_FILE()
    names = []
    if not f.exists():
        return names
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            os.environ.setdefault(k, v)
            names.append(k)
    return names


def expand_secrets(s: str) -> str:
    """Replace ${VAR} with the environment variable's value (missing -> a clear error, not a silent empty string)."""
    if "${" in (s or ""):
        load_env_file()                # pick up variables added to data/cloud.env since startup
    def rep(m):
        v = os.environ.get(m.group(1))
        if v is None:
            raise ValueError(f"Environment variable {m.group(1)} is not set (add it to data/cloud.env or the server environment)")
        return v
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", rep, s or "")


def redact(s: str) -> str:
    """Hide inline passwords in URLs for display/logs (user:secret@ -> user:***@)."""
    return re.sub(r"(://[^:/@\s]*:)([^@\s]+)(@)", lambda m: m.group(1) + ("${…}" if m.group(2).startswith("${") else "***") + m.group(3), s or "")


# ============================================================================= object storage
def provider_of(path: str) -> str | None:
    m = re.match(r"^([a-z0-9]+)://", str(path or ""))
    return SCHEMES.get(m.group(1)) if m else None


def settings() -> dict:
    """Global non-secret cloud settings (Settings -> Cloud)."""
    return dict(config.load_config().get("cloud") or {})


def save_settings(cloud: dict) -> dict:
    allowed = {"aws": {"profile", "region", "endpoint_url", "anon"},
               "gcp": {"project", "credentials_file", "endpoint_url"},
               "azure": {"account_name", "connection_string_env", "account_key_env", "tenant_id", "endpoint_url"}}
    clean = {p: {k: v for k, v in (cloud.get(p) or {}).items() if k in allowed[p] and v not in (None, "")} for p in allowed}
    cfg = config.load_config()
    cfg["cloud"] = clean
    config.save_config(cfg)
    return clean


def storage_options(path: str, auth: dict | None = None) -> dict:
    """fsspec storage options for a cloud URL: global settings, overridden by the source's own `auth`."""
    prov = provider_of(path)
    if not prov:
        return {}
    load_env_file()
    s = {**(settings().get(prov) or {}), **(auth or {})}
    if prov == "aws":
        o: dict = {}
        if s.get("profile"):
            o["profile"] = s["profile"]
        if s.get("anon"):
            o["anon"] = True
        ck = {k: v for k, v in (("region_name", s.get("region")), ("endpoint_url", s.get("endpoint_url"))) if v}
        if ck:
            o["client_kwargs"] = ck
        return o
    if prov == "gcp":
        if s.get("endpoint_url"):
            # GCS emulators / compatible stores: skip gcsfs's gRPC bucket-type probe (it needs real GCS over TLS).
            # Read when gcsfs is first imported, which fsspec does lazily on the first gs:// access.
            os.environ.setdefault("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "false")
        o = {}
        if s.get("project"):
            o["project"] = s["project"]
        if s.get("credentials_file"):
            o["token"] = str(Path(s["credentials_file"]).expanduser())
        if s.get("endpoint_url"):
            o["endpoint_url"] = s["endpoint_url"]
            o.setdefault("token", "anon")
        return o
    o = {}
    if s.get("connection_string_env"):
        o["connection_string"] = expand_secrets("${" + s["connection_string_env"] + "}")
    else:
        if s.get("account_name"):
            o["account_name"] = s["account_name"]
        if s.get("account_key_env"):
            o["account_key"] = expand_secrets("${" + s["account_key_env"] + "}")
        elif s.get("account_name"):
            o["anon"] = False          # adlfs then uses DefaultAzureCredential (az login, managed identity, env service principal)
        if s.get("tenant_id"):
            o["tenant_id"] = s["tenant_id"]
    return o


def filesystem(path: str, auth: dict | None = None):
    import fsspec
    prov = provider_of(path)
    if prov and not _installed(PACKAGES[prov]):
        raise ValueError(f"Reading {path.split('://')[0]}:// needs the '{PACKAGES[prov]}' package: pip install {PACKAGES[prov]}")
    fs, _ = fsspec.core.url_to_fs(path, **storage_options(path, auth))
    return fs


def glob(path: str, exts: tuple[str, ...], auth: dict | None = None) -> list[str]:
    """Files at a cloud URL: one object, a 'folder' (prefix) or a glob. Returns full URLs."""
    fs = filesystem(path, auth)
    scheme = path.split("://", 1)[0]
    bare = path.split("://", 1)[1]
    if any(c in bare for c in "*?["):
        found = fs.glob(bare)
    elif fs.isdir(bare):
        found = [f for f in fs.find(bare) if f.lower().endswith(exts)]
    else:
        found = [bare] if fs.exists(bare) else []
    return sorted(f"{scheme}://{f}" for f in found)


def _installed(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


# ============================================================================= status / test
def status() -> dict:
    """Which providers are usable here: library installed, credentials found (no secret values are returned)."""
    names = load_env_file()
    s = settings()
    out = {"env_file": str(ENV_FILE()), "env_file_vars": names, "providers": {}}
    # AWS
    aws = {"library": _installed("s3fs"), "settings": s.get("aws", {})}
    try:
        import boto3
        sess = boto3.Session(profile_name=s.get("aws", {}).get("profile"))
        cred = sess.get_credentials()
        aws["credentials"] = f"found ({cred.method})" if cred else "not found"
        aws["region"] = sess.region_name
    except Exception as e:
        aws["credentials"] = f"not found ({type(e).__name__})"
    out["providers"]["aws"] = aws
    # GCP
    gcp = {"library": _installed("gcsfs"), "settings": s.get("gcp", {})}
    keyfile = s.get("gcp", {}).get("credentials_file") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if s.get("gcp", {}).get("endpoint_url") and not keyfile:
        gcp["credentials"] = "anonymous (custom endpoint)"
    elif keyfile:
        gcp["credentials"] = "key file " + ("found" if Path(keyfile).expanduser().exists() else "missing: " + keyfile)
    else:
        try:
            import google.auth
            _, proj = google.auth.default()
            gcp["credentials"] = f"application default credentials (project {proj})"
        except Exception:
            gcp["credentials"] = "not found"
    out["providers"]["gcp"] = gcp
    # Azure
    az = {"library": _installed("adlfs"), "settings": s.get("azure", {})}
    a = s.get("azure", {})
    if a.get("connection_string_env"):
        az["credentials"] = f"connection string from ${a['connection_string_env']}: " + ("set" if os.environ.get(a["connection_string_env"]) else "NOT SET")
    elif a.get("account_key_env"):
        az["credentials"] = f"account key from ${a['account_key_env']}: " + ("set" if os.environ.get(a["account_key_env"]) else "NOT SET")
    elif os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        az["credentials"] = "AZURE_STORAGE_CONNECTION_STRING is set"
    else:
        import shutil
        az["credentials"] = "DefaultAzureCredential" + (" (az CLI installed)" if shutil.which("az") else " (az CLI not installed)")
    out["providers"]["azure"] = az
    out["warehouses"] = {k: {**v, "driver": _installed({"redshift": "psycopg", "athena": "pyathena", "bigquery": "sqlalchemy_bigquery",
                                                        "azure_sql": "pymssql"}[k])} for k, v in WAREHOUSES.items()}
    return out


def test(path: str, auth: dict | None = None) -> dict:
    """List a few objects at a cloud URL."""
    exts = (".csv", ".parquet", ".pq", ".csv.gz", ".json", ".jsonl", ".png", ".jpg", ".jpeg", ".wav", ".txt")
    files = glob(path, exts, auth)
    if not files:
        fs = filesystem(path, auth)
        bare = path.split("://", 1)[1]
        listing = fs.ls(bare, detail=False)[:20] if fs.exists(bare) else []
        return {"ok": bool(listing), "files": [], "listing": listing, "message": "Reachable, but no data files there" if listing else "Nothing found"}
    return {"ok": True, "files": files[:20], "count": len(files)}


load_env_file()
