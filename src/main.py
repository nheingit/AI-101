import os
import logging
from telegram import Update
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
import pandas as pd
import numpy as np
from questions import answer_question
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
tg_bot_token = os.environ['TG_BOT_TOKEN']

df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

messages = [{
  "role": "system",
  "content": "You are a helpful assistant that answers questions."
}]

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO)

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
  messages.append({"role": "user", "content": update.message.text})
  completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=messages)
  completion_answer = completion['choices'][0]['message']['content']
  messages.append({"role": "assistant", "content": completion_answer})

  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=completion_answer)

async def question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = answer_question(df, question=update.message.text)
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text=answer)



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="I'm a bot, please talk to me!")


if __name__ == '__main__':
  application = ApplicationBuilder().token(tg_bot_token).build()

  start_handler = CommandHandler('start', start)
  question_handler = CommandHandler('question', question)
  chat_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), chat)
  
  application.add_handler(question_handler)
  application.add_handler(chat_handler)
  application.add_handler(start_handler)


  application.run_polling()
