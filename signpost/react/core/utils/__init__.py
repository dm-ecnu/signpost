#
#  Core utility functions
#
#  Merged from:
#  - api/utils/__init__.py (json_dumps, get_uuid, timestamps, serialization, etc.)
#  - rag/utils/__init__.py (num_tokens_from_string, truncate, encoder, etc.)
#
import base64
import datetime
import io
import json
import os
import pickle
import re
import socket
import time
import uuid
import logging
import importlib
import requests
from enum import Enum, IntEnum

from transformers import AutoTokenizer

from core.config import get_project_base_directory, get_base_config, SERVICE_CONF


# ---------------------------------------------------------------------------
# Tokenizer (from rag/utils/__init__.py)
# ---------------------------------------------------------------------------
tokenizer_path = os.getenv("SIGNPOST_TOKENIZER_PATH", "Qwen/Qwen2.5-7B-Instruct")
encoder = None


def _get_encoder():
    global encoder
    if encoder is None:
        encoder = AutoTokenizer.from_pretrained(tokenizer_path)
    return encoder


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        return len(_get_encoder().encode(string))
    except Exception:
        return 0


def truncate(string: str, max_len: int | float) -> str:
    """Returns truncated text if the length of text exceed max_len."""
    current_encoder = _get_encoder()
    return current_encoder.decode(current_encoder.encode(string)[: int(max_len)])


def clean_markdown_block(text):
    text = re.sub(r"^\s*```markdown\s*\n?", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text)
    return text.strip()


def get_float(v):
    if v is None:
        return float("-inf")
    try:
        return float(v)
    except Exception:
        return float("-inf")


def rmSpace(txt):
    txt = re.sub(r"([^a-z0-9.,\)>]) +([^ ])", r"\1\2", txt, flags=re.IGNORECASE)
    return re.sub(r"([^ ]) +([^a-z0-9.,\(<])", r"\1\2", txt, flags=re.IGNORECASE)


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        key = str(cls) + str(os.getpid())
        if key not in instances:
            instances[key] = cls(*args, **kw)
        return instances[key]

    return _singleton


def findMaxDt(fnm):
    m = "1970-01-01 00:00:00"
    try:
        with open(fnm, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip("\n")
                if line == "nan":
                    continue
                if line > m:
                    m = line
    except Exception:
        pass
    return m


def findMaxTm(fnm):
    m = 0
    try:
        with open(fnm, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip("\n")
                if line == "nan":
                    continue
                if int(line) > m:
                    m = int(line)
    except Exception:
        pass
    return m


# ---------------------------------------------------------------------------
# JSON / serialization (from api/utils/__init__.py)
# ---------------------------------------------------------------------------
use_deserialize_safe_module = get_base_config("use_deserialize_safe_module", False)


class BaseType:
    def to_dict(self):
        return dict([(k.lstrip("_"), v) for k, v in self.__dict__.items()])

    def to_dict_with_type(self):
        def _dict(obj):
            module = None
            if issubclass(obj.__class__, BaseType):
                data = {}
                for attr, v in obj.__dict__.items():
                    k = attr.lstrip("_")
                    data[k] = _dict(v)
                module = obj.__module__
            elif isinstance(obj, (list, tuple)):
                data = []
                for i, vv in enumerate(obj):
                    data.append(_dict(vv))
            elif isinstance(obj, dict):
                data = {}
                for _k, vv in obj.items():
                    data[_k] = _dict(vv)
            else:
                data = obj
            return {"type": obj.__class__.__name__, "data": data, "module": module}

        return _dict(self)


class CustomJSONEncoder(json.JSONEncoder):
    def __init__(self, **kwargs):
        self._with_type = kwargs.pop("with_type", False)
        super().__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, datetime.timedelta):
            return str(obj)
        elif issubclass(type(obj), Enum) or issubclass(type(obj), IntEnum):
            return obj.value
        elif isinstance(obj, set):
            return list(obj)
        elif issubclass(type(obj), BaseType):
            if not self._with_type:
                return obj.to_dict()
            else:
                return obj.to_dict_with_type()
        elif isinstance(obj, type):
            return obj.__name__
        else:
            return json.JSONEncoder.default(self, obj)


def rag_uuid():
    return uuid.uuid1().hex


def string_to_bytes(string):
    return string if isinstance(string, bytes) else string.encode(encoding="utf-8")


def bytes_to_string(byte):
    return byte.decode(encoding="utf-8")


def json_dumps(src, byte=False, indent=None, with_type=False):
    dest = json.dumps(src, indent=indent, cls=CustomJSONEncoder, with_type=with_type)
    if byte:
        dest = string_to_bytes(dest)
    return dest


def json_loads(src, object_hook=None, object_pairs_hook=None):
    if isinstance(src, bytes):
        src = bytes_to_string(src)
    return json.loads(src, object_hook=object_hook, object_pairs_hook=object_pairs_hook)


def current_timestamp():
    return int(time.time() * 1000)


def timestamp_to_date(timestamp, format_string="%Y-%m-%d %H:%M:%S"):
    if not timestamp:
        timestamp = time.time()
    timestamp = int(timestamp) / 1000
    time_array = time.localtime(timestamp)
    str_date = time.strftime(format_string, time_array)
    return str_date


def date_string_to_timestamp(time_str, format_string="%Y-%m-%d %H:%M:%S"):
    time_array = time.strptime(time_str, format_string)
    time_stamp = int(time.mktime(time_array) * 1000)
    return time_stamp


def serialize_b64(src, to_str=False):
    dest = base64.b64encode(pickle.dumps(src))
    if not to_str:
        return dest
    else:
        return bytes_to_string(dest)


def deserialize_b64(src):
    src = base64.b64decode(string_to_bytes(src) if isinstance(src, str) else src)
    if use_deserialize_safe_module:
        return restricted_loads(src)
    return pickle.loads(src)


safe_module = {"numpy", "rag_flow"}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in safe_module:
            _module = importlib.import_module(module)
            return getattr(_module, name)
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" % (module, name))


def restricted_loads(src):
    """Helper function analogous to pickle.loads()."""
    return RestrictedUnpickler(io.BytesIO(src)).load()


def get_lan_ip():
    if os.name != "nt":
        import fcntl
        import struct

        def get_interface_ip(ifname):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", string_to_bytes(ifname[:15])))[20:24])

    ip = socket.gethostbyname(socket.getfqdn())
    if ip.startswith("127.") and os.name != "nt":
        interfaces = ["bond1", "eth0", "eth1", "eth2", "wlan0", "wlan1", "wifi0", "ath0", "ath1", "ppp0"]
        for ifname in interfaces:
            try:
                ip = get_interface_ip(ifname)
                break
            except IOError:
                pass
    return ip or ""


def from_dict_hook(in_dict: dict):
    if "type" in in_dict and "data" in in_dict:
        if in_dict["module"] is None:
            return in_dict["data"]
        else:
            return getattr(importlib.import_module(in_dict["module"]), in_dict["type"])(**in_dict["data"])
    else:
        return in_dict


def get_uuid():
    return uuid.uuid1().hex


def datetime_format(date_time: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(date_time.year, date_time.month, date_time.day, date_time.hour, date_time.minute, date_time.second)


def get_format_time() -> datetime.datetime:
    return datetime_format(datetime.datetime.now())


def str2date(date_time: str):
    return datetime.datetime.strptime(date_time, "%Y-%m-%d")


def elapsed2time(elapsed):
    seconds = elapsed / 1000
    minuter, second = divmod(seconds, 60)
    hour, minuter = divmod(minuter, 60)
    return "%02d:%02d:%02d" % (hour, minuter, second)


def decrypt(line):
    from Cryptodome.PublicKey import RSA
    from Cryptodome.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5

    file_path = os.path.join(get_project_base_directory(), "conf", "private.pem")
    rsa_key = RSA.importKey(open(file_path).read(), "Welcome")
    cipher = Cipher_pkcs1_v1_5.new(rsa_key)
    return cipher.decrypt(base64.b64decode(line), "Fail to decrypt password!").decode("utf-8")


def decrypt2(crypt_text):
    from base64 import b64decode, b16decode
    from Crypto.Cipher import PKCS1_v1_5 as Cipher_PKCS1_v1_5
    from Crypto.PublicKey import RSA

    decode_data = b64decode(crypt_text)
    if len(decode_data) == 127:
        hex_fixed = "00" + decode_data.hex()
        decode_data = b16decode(hex_fixed.upper())
    file_path = os.path.join(get_project_base_directory(), "conf", "private.pem")
    pem = open(file_path).read()
    rsa_key = RSA.importKey(pem, "Welcome")
    cipher = Cipher_PKCS1_v1_5.new(rsa_key)
    decrypt_text = cipher.decrypt(decode_data, None)
    return (b64decode(decrypt_text)).decode()


def download_img(url):
    if not url:
        return ""
    response = requests.get(url)
    return "data:" + response.headers.get("Content-Type", "image/jpg") + ";" + "base64," + base64.b64encode(response.content).decode("utf-8")


def delta_seconds(date_string: str):
    dt = datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    return (datetime.datetime.now() - dt).total_seconds()
