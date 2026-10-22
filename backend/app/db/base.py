from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# ---------------------------------------------------------------------------
# Import all ORM models here so that Base.metadata registers every table.
# Alembic's autogenerate reads Base.metadata, so models MUST be imported
# before any migration command is run.
# ---------------------------------------------------------------------------
# fmt: off
from backend.app.models import strategy_request as _sr_module  # noqa: E402, F401
from backend.app.models import business as _biz_module  # noqa: E402, F401
from backend.app.models import online_presence_source as _ops_module  # noqa: E402, F401
from backend.app.models import conversation as _conv_module  # noqa: E402, F401
from backend.app.models import chat_message as _cm_module  # noqa: E402, F401
from backend.app.models import strategy as _st_module  # noqa: E402, F401
from backend.app.models import llm_usage as _lu_module  # noqa: E402, F401
from backend.app.models import uploaded_document as _ud_module  # noqa: E402, F401
from backend.app.models import revision_message as _rm_module  # noqa: E402, F401
# fmt: on