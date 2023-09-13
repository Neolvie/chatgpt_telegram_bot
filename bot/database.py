from typing import Optional, Any

import pymongo
import uuid
from datetime import datetime

import config


class Database:
    def __init__(self):
        self.client = pymongo.MongoClient(config.mongodb_uri)
        self.db = self.client["chatgpt_telegram_bot"]

        self.user_collection = self.db["user"]
        self.dialog_collection = self.db["dialog"]

    def check_if_user_exists(self, user_id: int, raise_exception: bool = False):
        if self.user_collection.count_documents({"_id": user_id}) > 0:
            return True
        else:
            if raise_exception:
                raise ValueError(f"User {user_id} does not exist")
            else:
                return False

    def add_new_user(
        self,
        user_id: int,
        chat_id: int,
        username: str = "",
        first_name: str = "",
        last_name: str = "",
    ):
        user_dict = {
            "_id": user_id,
            "chat_id": chat_id,

            "username": username,
            "first_name": first_name,
            "last_name": last_name,

            "last_interaction": datetime.now(),
            "first_seen": datetime.now(),

            "current_dialog_id": None,
            "current_chat_mode": "assistant",
            "current_model": config.models["available_text_models"][0],

            "n_used_tokens": {},

            "n_transcribed_seconds": 0.0,  # voice message transcription

            "payment_date": None,
            "subscribe_to": None,
            "transaction": None,
            "has_reached_limit": 0,
            "messages_count": 0,
            "subscriptions": []
        }

        if not self.check_if_user_exists(user_id):
            self.user_collection.insert_one(user_dict)

    def start_new_dialog(self, user_id: int):
        self.check_if_user_exists(user_id, raise_exception=True)

        dialog_id = str(uuid.uuid4())
        dialog_dict = {
            "_id": dialog_id,
            "user_id": user_id,
            "chat_mode": self.get_user_attribute(user_id, "current_chat_mode"),
            "start_time": datetime.now(),
            "model": self.get_user_attribute(user_id, "current_model"),
            "messages": []
        }

        # add new dialog
        self.dialog_collection.insert_one(dialog_dict)

        # update user's current dialog
        self.user_collection.update_one(
            {"_id": user_id},
            {"$set": {"current_dialog_id": dialog_id}}
        )

        return dialog_id

    def get_user_attribute(self, user_id: int, key: str):
        self.check_if_user_exists(user_id, raise_exception=True)
        user_dict = self.user_collection.find_one({"_id": user_id})

        if key not in user_dict:
            return None

        return user_dict[key]

    def set_user_attribute(self, user_id: int, key: str, value: Any):
        self.check_if_user_exists(user_id, raise_exception=True)
        self.user_collection.update_one({"_id": user_id}, {"$set": {key: value}})

    def update_n_used_tokens(self, user_id: int, model: str, n_input_tokens: int, n_output_tokens: int):
        n_used_tokens_dict = self.get_user_attribute(user_id, "n_used_tokens")

        if model in n_used_tokens_dict:
            n_used_tokens_dict[model]["n_input_tokens"] += n_input_tokens
            n_used_tokens_dict[model]["n_output_tokens"] += n_output_tokens
        else:
            n_used_tokens_dict[model] = {
                "n_input_tokens": n_input_tokens,
                "n_output_tokens": n_output_tokens
            }

        self.set_user_attribute(user_id, "n_used_tokens", n_used_tokens_dict)

    def get_users_count(self):
        return self.user_collection.count_documents({})

    def get_subscription_count(self):
        return self.user_collection.count_documents({"payment_date": {"$exists": True, "$ne": None}})

    def get_subscribe_to(self, user_id: int):
        current_model = self.get_user_attribute(user_id, 'current_model')
        subscriptions = self.get_user_attribute(user_id, 'subscriptions')

        model_subscriptions = [subscription for subscription in subscriptions if current_model in subscription['models']]

        if len(model_subscriptions) == 0:
            return None

        newest_subscription = sorted(model_subscriptions, key=lambda s: s['subscribe_to'], reverse=True)[0]

        return newest_subscription['subscribe_to']

    def get_current_subscription(self, user_id: int):
        subscriptions = self.get_user_attribute(user_id, 'subscriptions')

        if len(subscriptions) == 0:
            return None

        subscription = sorted(subscriptions, key=lambda s: s['subscribe_to'], reverse=True)[0]

        if (datetime.now() - subscription['subscribe_to']).days > 0:
            return None

        return subscription


    def get_reached_limit_count(self):
        return self.user_collection.count_documents({"has_reached_limit": 1})

    def get_dialogs_count(self, user_id: int):
        self.check_if_user_exists(user_id, raise_exception=True)

        return self.dialog_collection.count_documents({"user_id": user_id})

    def get_dialog_messages_count(self, user_id: int):
        self.check_if_user_exists(user_id, raise_exception=True)

        dialog_id = self.get_user_attribute(user_id, "current_dialog_id")
        dialog_dict = self.dialog_collection.find_one({"_id": dialog_id, "user_id": user_id})
        return len(dialog_dict["messages"])

    def set_user_reached_limit(self, user_id: int):
        self.check_if_user_exists(user_id, raise_exception=True)

        self.set_user_attribute(user_id, "has_reached_limit", 1)

    def get_dialog_messages(self, user_id: int, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, "current_dialog_id")

        dialog_dict = self.dialog_collection.find_one({"_id": dialog_id, "user_id": user_id})
        return dialog_dict["messages"]

    def set_dialog_messages(self, user_id: int, dialog_messages: list, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, "current_dialog_id")

        self.set_user_attribute(user_id, "messages_count", self.get_user_attribute(user_id, "messages_count") + 1)

        self.dialog_collection.update_one(
            {"_id": dialog_id, "user_id": user_id},
            {"$set": {"messages": dialog_messages}}
        )


db = Database()
