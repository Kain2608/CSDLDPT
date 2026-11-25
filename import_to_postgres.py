"""Entry point nạp dữ liệu vào PostgreSQL (đồng bộ với Báo cáo Mục 3.2 và scripts/sync_to_postgres.py)."""

from __future__ import annotations

from scripts.sync_to_postgres import sync_all


def import_data(recreate_schema: bool = False) -> dict[str, int]:
    """Khởi tạo lược đồ hợp nhất OOP + pgvector (nếu chưa có) và nạp 648 bản ghi âm thanh."""
    return sync_all(recreate_schema=recreate_schema)


if __name__ == "__main__":
    import_data()
