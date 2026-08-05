import hashlib
import logging
import uuid

logger = logging.getLogger(__name__)


def get_machine_id() -> str:
    """PCのMACアドレスから不可逆ハッシュを生成し、一意かつ匿名なID (例: pc_a1b2c3d4) を返す"""
    mac = str(uuid.getnode())
    short_hash = hashlib.md5(mac.encode("utf-8")).hexdigest()[:8]
    machine_id = f"pc_{short_hash}"
    logger.debug("取得したMachine ID: %s", machine_id)
    return machine_id
