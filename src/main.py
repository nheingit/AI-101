import os
import logging
from telegram import Update
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
import pandas as pd
import numpy as np
from questions import answer_question
from dotenv import load_dotenv
import openai
import requests

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
tg_bot_token = os.environ['TG_BOT_TOKEN']

CODE_PROMPT = """
Here are two input:output examples for code generation. Please use these and follow the styling for future requests that you think are pertinent to the request. Make sure All HTML is generated with the JSX flavoring.

// INPUT:
// A Blue Box with 3 yellow cirles inside of it that have a red outline
// OUTPUT:
<div style={{
  backgroundColor: 'blue',
  padding: '20px',
  display: 'flex',
  justifyContent: 'space-around',
  alignItems: 'center',
  width: '300px',
  height: '100px',
}}>
  <div style={{
    backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{
    backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{
    backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
</div>

// INPUT:
// A RED BUTTON THAT SAYS 'CLICK ME'
// OUTPUT:
<button style={{
  backgroundColor: 'red',
  color: 'white',
  padding: '10px 20px',
  border: 'none',
  borderRadius: '50px',
  cursor: 'pointer'
}}>
  Click Me
</button>
"""

df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

messages = [{
  "role": "system",
  "content": "You are a helpful assistant that answers questions."
}, {
  "role": "system",
  "content": CODE_PROMPT
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

async def code_generation(update: Update, context: ContextTypes.DEFAULT_TYPE):
  messages.append({"role": "user", "content": update.message.text})
  completion = openai.ChatCompletion.create(model="gpt-4",
                                            messages=messages)
  completion_answer = completion['choices'][0]['message']['content']
  messages.append({"role": "assistant", "content": completion_answer})

  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text=completion_answer)

async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
  response = openai.Image.create(prompt=update.message.text,
                                 n=1,
                                 size="1024x1024")
  image_url = response['data'][0]['url']
  image_response = requests.get(image_url)
  await context.bot.send_photo(chat_id=update.effective_chat.id,
                           photo=image_response.content)



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await context.bot.send_message(chat_id=update.effective_chat.id,
                                 text="I'm a bot, please talk to me!")


if __name__ == '__main__':
  application = ApplicationBuilder().token(tg_bot_token).build()

  start_handler = CommandHandler('start', start)
  image_handler = CommandHandler('image', image)
  code_generation_handler = CommandHandler('code', code_generation)
  question_handler = CommandHandler('question', question)
  chat_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), chat)
  
  application.add_handler(question_handler)
  application.add_handler(image_handler)
  application.add_handler(code_generation_handler)
  application.add_handler(chat_handler)
  application.add_handler(start_handler)


  application.run_polling()
