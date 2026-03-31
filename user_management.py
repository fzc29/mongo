# manage_users.py
import os
import bcrypt
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI_ADMIN"))
users = client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["users"]

# Create a unique index on username — prevents duplicates
users.create_index("username", unique=True)

def create_user(username: str, password: str, role: str = "guest"):
    """
    Create a new user.
    role: "guest" (read only) or "admin" (can upload)
    """
    if users.find_one({"username": username}):
        print(f"  User '{username}' already exists.")
        return

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({
        "username": username,
        "password_hash": hashed,
        "role": role,
        "created_at": __import__("datetime").datetime.utcnow(),
    })
    print(f"  Created {role} user: {username}")


def delete_user(username: str):
    result = users.delete_one({"username": username})
    if result.deleted_count:
        print(f"  Deleted user: {username}")
    else:
        print(f"  User '{username}' not found.")


def list_users():
    print(f"\n{'Username':<20} {'Role':<10}")
    print("-" * 30)
    for u in users.find({}, {"username": 1, "role": 1}):
        print(f"  {u['username']:<20} {u['role']:<10}")


def change_password(username: str, new_password: str):
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
    result = users.update_one(
        {"username": username},
        {"$set": {"password_hash": hashed}}
    )
    if result.modified_count:
        print(f"  Password updated for: {username}")
    else:
        print(f"  User '{username}' not found.")


def change_role(username: str, new_role: str):
    result = users.update_one(
        {"username": username},
        {"$set": {"role": new_role}}
    )
    if result.modified_count:
        print(f"  Role updated for {username} → {new_role}")
    else:
        print(f"  User '{username}' not found.")

