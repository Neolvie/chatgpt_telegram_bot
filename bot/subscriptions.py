from datetime import datetime

from dateutil.relativedelta import relativedelta
from telegram import Update, User, LabeledPrice
from telegram.constants import ParseMode
from telegram.ext import CallbackContext

import config
from database import db


async def check_user_subscription(update: Update, context: CallbackContext, user: User):
    await convert_to_new_subscriptions_format(user.id)

    subscribe_to = db.get_subscribe_to(user.id)
    messages_count = db.get_user_attribute(user.id, 'messages_count')

    # Тестовый режим
    if subscribe_to is None or messages_count is None:
        if messages_count is None:
            db.set_user_attribute(user.id, 'messages_count', 0)
            db.set_user_attribute(user.id, 'has_reached_limit', 0)
            messages_count = 0

        if messages_count > config.max_free_messages:
            db.set_user_reached_limit(user.id)
            return False

        return True

    if messages_count <= config.max_free_messages:
        return True

    if (datetime.now() - subscribe_to).days > 0:
        return False

    return True


async def convert_to_new_subscriptions_format(user_id: int):
    subscribe_to = db.get_user_attribute(user_id, "subscribe_to")
    current_subscriptions = db.get_user_attribute(user_id, 'subscriptions')

    if subscribe_to is None and current_subscriptions is None:
        db.set_user_attribute(user_id, "subscriptions", [])
        return
    elif current_subscriptions is not None:
        return

    level = next(level for level in config.levels if level['id'] == 1)

    if current_subscriptions is None:
        current_subscriptions = []

    subscription_info = {
        "payment_date": db.get_user_attribute(user_id, "payment_date"),
        "subscribe_to": db.get_user_attribute(user_id, "subscribe_to"),
        "price": level['price'],
        "transaction": db.get_user_attribute(user_id, "transaction"),
        "level_id": level['id'],
        "models": [level['model']]
    }

    db.set_user_attribute(user_id, "subscriptions", current_subscriptions + [subscription_info])


async def subscribe_handle(update: Update, context: CallbackContext) -> None:
    user_id = update.message.from_user.id

    subscription = await check_user_subscription(update, context, update.message.from_user)
    current_subscription = db.get_current_subscription(user_id)

    if subscription:
        await update.message.reply_text(f"У вас уже есть действующая подписка до <b>{current_subscription['subscribe_to']:%d.%m.%Y}</b>",
                                        parse_mode=ParseMode.HTML)

    for level in [level for level in config.levels if level['model'] in config.models["available_text_models"]]:

        if current_subscription is not None and current_subscription['level_id'] >= level['id']:
            continue

        current_price = 0
        discount = ''

        if current_subscription is not None:
            current_price = current_subscription['price']
            discount = f'С учетом скидки за текущую подписку {current_price} {config.currency}'

        """Sends an invoice without shipping-payment."""
        chat_id = update.message.chat_id
        title = f"{level['title']}"
        description = f"{level['description']} {level['subscribe_period_text']}. {discount}" #на {config.subscribe_period_text}"
        # select a payload just for you to recognize its the donation from your bot
        payload = level['model']
        # In order to get a provider_token see https://core.telegram.org/bots/payments#getting-a-token
        currency = config.currency
        # price in dollars
        # price * 100 so as to include 2 decimal points

        prices = [LabeledPrice(f"{level['title']} ({level['subscribe_period_text']})", (level['price'] - current_price) * 100)]

        receipt = {
            'items': [
                {
                    'description': f"{level['description']} {level['subscribe_period_text']}",
                    'quantity': "1.00",
                    'amount': {
                        "value": str(level['price'] - current_price),
                        "currency": currency
                    },
                    'vat_code': 1
                }
            ]
        }

        # optionally pass need_name=True, need_phone_number=True,
        # need_email=True, need_shipping_address=True, is_flexible=True
        await context.bot.send_invoice(
            chat_id,
            title,
            description,
            payload,
            config.payment_provider_token,
            currency,
            prices,
            need_email=True,
            send_email_to_provider=True,
            provider_data={'receipt': receipt}
        )


async def precheckout_callback(update: Update, context: CallbackContext) -> None:
    """Answers the PreQecheckoutQuery"""
    query = update.pre_checkout_query
    # check the payload, is this from your bot?

    payloads = [level['model'] for level in config.levels]
    if query.invoice_payload not in payloads:
        # answer False pre_checkout_query
        await query.answer(ok=False, error_message="Что-то пошло не так...")
    else:
        await query.answer(ok=True)


async def successful_payment_callback(update: Update, context: CallbackContext) -> None:
    """Confirms the successful payment."""
    # do something after successfully receiving payment?
    payload = update.message.successful_payment.invoice_payload
    paid_level = next(level for level in config.levels if level['model'] == payload)

    models = [level['model'] for level in config.levels if paid_level['id'] >= level['id']]

    user_id = update.message.from_user.id

    current_subscriptions = db.get_user_attribute(user_id, 'subscriptions')
    if current_subscriptions is None:
        current_subscriptions = []

    subscription_info = {
        "payment_date": datetime.now(),
        "subscribe_to": datetime.now() + relativedelta(days=paid_level['subscribe_period']),
        "price": paid_level['price'],
        "transaction": update.message.successful_payment.provider_payment_charge_id,
        "level_id": paid_level['id'],
        "models": models
    }

    db.set_user_attribute(user_id, "subscriptions", current_subscriptions + [subscription_info])

    await update.message.reply_text("Спасибо за подписку!")
