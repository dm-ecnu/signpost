from __future__ import annotations

"""F2 smoke CLI for external storage services."""

import argparse
import json

from signpost.storage.health import check_elasticsearch, check_minio, check_postgres, check_redis


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test F2 storage connections")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--es", action="store_true")
    parser.add_argument("--minio", action="store_true")
    parser.add_argument("--redis", action="store_true")
    parser.add_argument("--db", action="store_true")
    args = parser.parse_args()

    checks = []
    if args.all or args.es:
        checks.append(check_elasticsearch())
    if args.all or args.minio:
        checks.append(check_minio())
    if args.all or args.redis:
        checks.append(check_redis())
    if args.all or args.db:
        checks.append(check_postgres())
    if not checks:
        checks = [check_elasticsearch()]
    print(json.dumps([item.as_dict() for item in checks], ensure_ascii=False, indent=2))
    return 0 if all(item.ok for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

