#
#  Core configuration module
#
#  Merged from:
#  - api/constants.py (constants)
#  - api/utils/file_utils.py (path/yaml/json utilities)
#  - api/utils/__init__.py (config loading, get_base_config, decrypt)
#  - rag/settings.py (ES/Redis/MinIO/storage config, queue names)
#
import copy
import json
import logging
import os
import sys
import threading

from cachetools import LRUCache, cached
from filelock import FileLock
from ruamel.yaml import YAML


# ---------------------------------------------------------------------------
# Constants (from api/constants.py)
# ---------------------------------------------------------------------------
NAME_LENGTH_LIMIT = 2**10
IMG_BASE64_PREFIX = "data:image/png;base64,"
SERVICE_CONF = "service_conf.yaml"
API_VERSION = "v1"
RAG_FLOW_SERVICE_NAME = "ragflow"
REQUEST_WAIT_SEC = 2
REQUEST_MAX_WAIT_SEC = 300
DATASET_NAME_LIMIT = 128
FILE_NAME_LEN_LIMIT = 255

# ---------------------------------------------------------------------------
# Project paths (from api/utils/file_utils.py)
# ---------------------------------------------------------------------------
PROJECT_BASE = os.getenv("RAG_PROJECT_BASE") or os.getenv("RAG_DEPLOY_BASE")
RAG_BASE = os.getenv("RAG_BASE")

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


def get_project_base_directory(*args):
    global PROJECT_BASE
    if PROJECT_BASE is None:
        PROJECT_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
            )
        )
    if args:
        return os.path.join(PROJECT_BASE, *args)
    return PROJECT_BASE


def get_rag_directory(*args):
    global RAG_BASE
    if RAG_BASE is None:
        RAG_BASE = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.pardir,
                os.pardir,
            )
        )
    if args:
        return os.path.join(RAG_BASE, *args)
    return RAG_BASE


def get_rag_python_directory(*args):
    return get_rag_directory("python", *args)


def get_home_cache_dir():
    dir = os.path.join(os.path.expanduser("~"), ".ragflow")
    try:
        os.mkdir(dir)
    except OSError:
        pass
    return dir


# ---------------------------------------------------------------------------
# YAML / JSON conf loaders (from api/utils/file_utils.py)
# ---------------------------------------------------------------------------
@cached(cache=LRUCache(maxsize=10))
def load_json_conf(conf_path):
    if os.path.isabs(conf_path):
        json_conf_path = conf_path
    else:
        json_conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(json_conf_path) as f:
            return json.load(f)
    except BaseException:
        raise EnvironmentError("loading json file config from '{}' failed!".format(json_conf_path))


def dump_json_conf(config_data, conf_path):
    if os.path.isabs(conf_path):
        json_conf_path = conf_path
    else:
        json_conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(json_conf_path, "w") as f:
            json.dump(config_data, f, indent=4)
    except BaseException:
        raise EnvironmentError("loading json file config from '{}' failed!".format(json_conf_path))


def load_json_conf_real_time(conf_path):
    if os.path.isabs(conf_path):
        json_conf_path = conf_path
    else:
        json_conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(json_conf_path) as f:
            return json.load(f)
    except BaseException:
        raise EnvironmentError("loading json file config from '{}' failed!".format(json_conf_path))


def load_yaml_conf(conf_path):
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(conf_path) as f:
            yaml = YAML(typ="safe", pure=True)
            return yaml.load(f)
    except Exception as e:
        raise EnvironmentError("loading yaml file config from {} failed:".format(conf_path), e)


def rewrite_yaml_conf(conf_path, config):
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(get_project_base_directory(), conf_path)
    try:
        with open(conf_path, "w") as f:
            yaml = YAML(typ="safe")
            yaml.dump(config, f)
    except Exception as e:
        raise EnvironmentError("rewrite yaml file config {} failed:".format(conf_path), e)


def rewrite_json_file(filepath, json_data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, separators=(",", ": "))
    f.close()


# ---------------------------------------------------------------------------
# Config reading (from api/utils/__init__.py)
# ---------------------------------------------------------------------------
def conf_realpath(conf_name):
    conf_path = f"conf/{conf_name}"
    return os.path.join(get_project_base_directory(), conf_path)


def read_config(conf_name=SERVICE_CONF):
    local_config = {}
    local_path = conf_realpath(f"local.{conf_name}")
    if os.path.exists(local_path):
        local_config = load_yaml_conf(local_path)
        if not isinstance(local_config, dict):
            raise ValueError(f'Invalid config file: "{local_path}".')
    global_config_path = conf_realpath(conf_name)
    global_config = load_yaml_conf(global_config_path)
    if not isinstance(global_config, dict):
        raise ValueError(f'Invalid config file: "{global_config_path}".')
    global_config.update(local_config)
    return global_config


CONFIGS = read_config()


def show_configs():
    msg = f"Current configs, from {conf_realpath(SERVICE_CONF)}:"
    for k, v in CONFIGS.items():
        if isinstance(v, dict):
            if "password" in v:
                v = copy.deepcopy(v)
                v["password"] = "*" * 8
            if "access_key" in v:
                v = copy.deepcopy(v)
                v["access_key"] = "*" * 8
            if "secret_key" in v:
                v = copy.deepcopy(v)
                v["secret_key"] = "*" * 8
        msg += f"\n\t{k}: {v}"
    logging.info(msg)


def get_base_config(key, default=None):
    if key is None:
        return None
    if default is None:
        default = os.environ.get(key.upper())
    return CONFIGS.get(key, default)


def update_config(key, value, conf_name=SERVICE_CONF):
    conf_path = conf_realpath(conf_name=conf_name)
    if not os.path.isabs(conf_path):
        conf_path = os.path.join(get_project_base_directory(), conf_path)
    with FileLock(os.path.join(os.path.dirname(conf_path), ".lock")):
        config = load_yaml_conf(conf_path=conf_path) or {}
        config[key] = value
        rewrite_yaml_conf(conf_path=conf_path, config=config)


# ---------------------------------------------------------------------------
# Decrypt helpers (from api/utils/__init__.py)
# ---------------------------------------------------------------------------
def decrypt_database_password(password):
    encrypt_password = get_base_config("encrypt_password", False)
    encrypt_module = get_base_config("encrypt_module", False)
    private_key = get_base_config("private_key", None)
    if not password or not encrypt_password:
        return password
    if not private_key:
        raise ValueError("No private key")
    import importlib

    module_fun = encrypt_module.split("#")
    pwdecrypt_fun = getattr(importlib.import_module(module_fun[0]), module_fun[1])
    return pwdecrypt_fun(private_key, password)


def decrypt_database_config(database=None, passwd_key="password", name="database"):
    if not database:
        database = get_base_config(name, {})
    database[passwd_key] = decrypt_database_password(database[passwd_key])
    return database


# ---------------------------------------------------------------------------
# Infrastructure config (from rag/settings.py)
# ---------------------------------------------------------------------------
RAG_CONF_PATH = os.path.join(get_project_base_directory(), "conf")

STORAGE_IMPL_TYPE = os.getenv("STORAGE_IMPL", "MINIO")
DOC_ENGINE = os.getenv("DOC_ENGINE", "elasticsearch")

ES = {}
INFINITY = {}
AZURE = {}
S3 = {}
MINIO = {}
OSS = {}
OS = {}

if DOC_ENGINE == "elasticsearch":
    ES = get_base_config("es", {})
elif DOC_ENGINE == "opensearch":
    OS = get_base_config("os", {})
elif DOC_ENGINE == "infinity":
    INFINITY = get_base_config("infinity", {"uri": "infinity:23817"})

if STORAGE_IMPL_TYPE in ["AZURE_SPN", "AZURE_SAS"]:
    AZURE = get_base_config("azure", {})
elif STORAGE_IMPL_TYPE == "AWS_S3":
    S3 = get_base_config("s3", {})
elif STORAGE_IMPL_TYPE == "MINIO":
    MINIO = decrypt_database_config(name="minio")
elif STORAGE_IMPL_TYPE == "OSS":
    OSS = get_base_config("oss", {})

try:
    REDIS = decrypt_database_config(name="redis")
except Exception:
    REDIS = {}

DOC_MAXIMUM_SIZE = int(os.environ.get("MAX_CONTENT_LENGTH", 128 * 1024 * 1024))
DOC_BULK_SIZE = int(os.environ.get("DOC_BULK_SIZE", 128))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", 16))
SVR_QUEUE_NAME = "rag_flow_svr_queue"
SVR_CONSUMER_GROUP_NAME = "rag_flow_svr_task_broker"
PAGERANK_FLD = "pagerank_fea"
TAG_FLD = "tag_feas"

PARALLEL_DEVICES = 0
try:
    import torch.cuda

    PARALLEL_DEVICES = torch.cuda.device_count()
    logging.info(f"found {PARALLEL_DEVICES} gpus")
except Exception:
    logging.info("can't import package 'torch'")


def print_rag_settings():
    logging.info(f"MAX_CONTENT_LENGTH: {DOC_MAXIMUM_SIZE}")
    logging.info(f"MAX_FILE_COUNT_PER_USER: {int(os.environ.get('MAX_FILE_NUM_PER_USER', 0))}")


def get_svr_queue_name(priority: int) -> str:
    if priority == 0:
        return SVR_QUEUE_NAME
    return f"{SVR_QUEUE_NAME}_{priority}"


def get_svr_queue_names():
    return [get_svr_queue_name(priority) for priority in [1, 0]]


# ---------------------------------------------------------------------------
# Application settings (sunk from api/settings.py)
# ---------------------------------------------------------------------------
LIGHTEN = int(os.environ.get("LIGHTEN", "0"))
BUILTIN_EMBEDDING_MODELS = ["BAAI/bge-large-zh-v1.5@BAAI"]

# LLM config (set by init_settings)
LLM = None
LLM_FACTORY = None
LLM_BASE_URL = None
CHAT_MDL = ""
EMBEDDING_MDL = ""
RERANK_MDL = ""
ASR_MDL = ""
IMAGE2TEXT_MDL = ""
API_KEY = None
PARSERS = None
FACTORY_LLM_INFOS = None

# Database config
DATABASE_TYPE = get_base_config("database_type", os.getenv("DB_TYPE", "mysql"))
DATABASE = decrypt_database_config(name=DATABASE_TYPE)

# Runtime singletons (set by init_settings)
docStoreConn = None
retrievaler = None
kg_retrievaler = None


def init_settings():
    """Initialize LLM config and storage singletons. Called once at startup."""
    global LLM, LLM_FACTORY, LLM_BASE_URL, LIGHTEN, DATABASE_TYPE, DATABASE, FACTORY_LLM_INFOS
    LIGHTEN = int(os.environ.get("LIGHTEN", "0"))
    DATABASE_TYPE = get_base_config("database_type", os.getenv("DB_TYPE", "mysql"))
    DATABASE = decrypt_database_config(name=DATABASE_TYPE)
    LLM = get_base_config("user_default_llm", {})
    LLM_DEFAULT_MODELS = LLM.get("default_models", {})
    LLM_FACTORY = LLM.get("factory")
    LLM_BASE_URL = LLM.get("base_url")

    try:
        with open(os.path.join(get_project_base_directory(), "conf", "llm_factories.json"), "r") as f:
            FACTORY_LLM_INFOS = json.load(f)["factory_llm_infos"]
    except Exception:
        FACTORY_LLM_INFOS = []

    global CHAT_MDL, EMBEDDING_MDL, RERANK_MDL, ASR_MDL, IMAGE2TEXT_MDL
    if not LIGHTEN:
        EMBEDDING_MDL = BUILTIN_EMBEDDING_MODELS[0]

    if LLM_DEFAULT_MODELS:
        CHAT_MDL = LLM_DEFAULT_MODELS.get("chat_model", CHAT_MDL)
        EMBEDDING_MDL = LLM_DEFAULT_MODELS.get("embedding_model", EMBEDDING_MDL)
        RERANK_MDL = LLM_DEFAULT_MODELS.get("rerank_model", RERANK_MDL)
        ASR_MDL = LLM_DEFAULT_MODELS.get("asr_model", ASR_MDL)
        IMAGE2TEXT_MDL = LLM_DEFAULT_MODELS.get("image2text_model", IMAGE2TEXT_MDL)

        CHAT_MDL = CHAT_MDL + (f"@{LLM_FACTORY}" if "@" not in CHAT_MDL and CHAT_MDL != "" else "")
        EMBEDDING_MDL = EMBEDDING_MDL + (f"@{LLM_FACTORY}" if "@" not in EMBEDDING_MDL and EMBEDDING_MDL != "" else "")
        RERANK_MDL = RERANK_MDL + (f"@{LLM_FACTORY}" if "@" not in RERANK_MDL and RERANK_MDL != "" else "")

    global API_KEY, PARSERS
    API_KEY = LLM.get("api_key")
    PARSERS = LLM.get(
        "parsers",
        "academic_impl2:Academic (Line-based),naive:General",
    )

    global docStoreConn, retrievaler, kg_retrievaler
    from core.storage.es_conn import ESConnection

    docStoreConn = ESConnection()

    from core.nlp import search

    retrievaler = search.Dealer(docStoreConn)

    from graphrag.retrieval import KGSearch

    kg_retrievaler = KGSearch(docStoreConn)
