from typing import Optional
from app.services.database_service import DatabaseManager
from app.models.user_models import UserModelInDB


def get_user_by_id(user_id: str) -> Optional[UserModelInDB]:
    """Fetch a user by internal user_id; returns None if not found."""
    dbm = DatabaseManager()
    prof = dbm.get_user_by_id(user_id)
    if not prof:
        return None
    # Adapt fields to UserModelInDB
    return UserModelInDB(
        id=prof.user_id,
        google_id=prof.google_id or "",
        email=prof.email,
        name=prof.name,
        picture=prof.picture,
        created_at=prof.created_at,
        updated_at=prof.last_login,
    )
from app.db.mongodb_utils import get_users_collection, mongo_to_pydantic
from app.models.user_models import UserModelInDB
from typing import Optional
import datetime
from pymongo import ReturnDocument


def get_user_by_google_id(google_id: str) -> Optional[UserModelInDB]:
    """Synchronous helper to fetch user by google_id using pymongo."""
    users_coll = get_users_collection()
    user_doc = users_coll.find_one({"google_id": google_id})
    if user_doc:
        return mongo_to_pydantic(user_doc, UserModelInDB)
    return None


def create_or_update_user_from_google(
    google_id: str, email: str, name: Optional[str], picture: Optional[str]
) -> UserModelInDB:
    users_coll = get_users_collection()
    now = datetime.datetime.now(datetime.timezone.utc)

    user_doc = users_coll.find_one_and_update(
        {"google_id": google_id},
        {
            "$set": {
                "email": email,
                "name": name,
                "picture": picture,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now, "google_id": google_id},
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return mongo_to_pydantic(user_doc, UserModelInDB)
