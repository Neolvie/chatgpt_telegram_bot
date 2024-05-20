from database import db
from subscriptions import *

total_count = 0
new_model = 'gpt-4o'
current_model = 'gpt-4-1106-preview'


def convert():
    total_count = 0
    for user in db.user_collection.find():
        if user['current_model'] == current_model:
            db.set_user_attribute(user['_id'], 'current_model', new_model)
            print(f"http://88.210.10.140:8083/db/chatgpt_telegram_bot/user/{user['_id']}")

        has_changes = False

        for subscription in user['subscriptions']:
            if any(current_model == x for x in subscription['models']):
                if all(new_model != x for x in subscription['models']):
                    subscription['models'].append(new_model)
                    has_changes = True
                    print(f'OK: {user["_id"]}')
                    total_count += 1

        if has_changes:
            db.set_user_attribute(user['_id'], 'subscriptions', user['subscriptions'])

    print(total_count)

    return total_count
