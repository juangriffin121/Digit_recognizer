from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio
import nest_asyncio
import numpy as np
from train import respuesta,load_red
from PIL import Image
import numpy as np
from used_functions import get_3_image_from_rgb
import time
from scipy.ndimage import gaussian_filter
from handle_image import show_im,save_im
import cv
from PIL import Image
from io import BytesIO

nest_asyncio.apply()

red = load_red('red.pickle')
TOKEN: Final = "6483422003:AAGvDstO5-PB5CeGnnJOC_UdVkuiWAIT-iQ"
BOT_USERNAME: Final = "@WkikiBot"

#Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await update.message.reply_text('Hello world')
  
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await update.message.reply_text('Please type something so i can respond')

async def process_image(path):
    
    im = Image.open(path)
    im = np.asarray(im)
    if im.shape == (20,20):
      image = im
      image = np.pad(image,4)
      image = gaussian_filter(image,.4)
    elif im.shape != (28,28):
      image.thumbnail((20,20),Image.LANCZOS)
      image = np.asarray(image)
      image = gaussian_filter(image,.4)
      image = np.pad(image,4)
    image = np.array([image])
    if image.shape == (1,28,28):
      save_im(image[0],'num.bmp')
      return im
    else:
      return None

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  await update.message.reply_text('This is a custom command')
  await update.message.reply_text('Bienvenido a Wkiki 3.0')
  await update.message.reply_text('Dibuje en paint en una imagen de 200 x 200 pixeles con blanco sobre negro el digito que quiere que Dajij dWkikikant prediga')
  image = await handle_image(update, context)
  await update.message.reply_text('imagen cargada')
  prediction_vector = respuesta(red, image)
  main_prediction = np.argmax(prediction_vector)
  await update.message.reply_text(f'Wkikikant piensa que escribiste un {main_prediction}')
  
  

# Responses

def handle_response(text:str) -> str:
  processed: str = text.lower()
  
  if 'hello' in processed:
    return "Hii"
  
  if 'Wkiki is the best' in processed:
    return 'asdasdasd'
    
  return 'i dont understand what you just said'

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
  message_type = update.message.chat.type
  text: str = update.message.text
  print(f'User \n ({update.message.chat.id}) in \n {message_type}: "{text}"')
  
  if message_type == 'group':
    if BOT_USERNAME in text:
      new_text: str = text.replace(BOT_USERNAME,'').strip()
      response: str = handle_response(new_text)
    else:
      return
  else:
    
    response: str = handle_response(text)
    
  print('BOT', response)
  
  await update.message.reply_text(response)
      
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.photo:
        # Get the file from the last photo in the message
        image_file = update.message.photo[-1].get_file()
        
        # Download the image file
        image_path = image_file.download("received_image.jpg")

        im = process_image("received_image.jpg")

        # Process the image and make predictions
        # You can access the image_path here and use it for processing

        # Respond to the user
        if im:
            await update.message.reply_text("Image received and processed.")
            return im
        else:
            await update.message.reply_text('La imagen debe ser un .bmp de 28 x 28 o 20 x 20. Intente de nuevo.')
    else:
        await update.message.reply_text('No se ha encontrado ninguna foto en el mensaje.')
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
  print(f'Update \n {update} caused error \n {context.error}')
  

if __name__ == '__main__':
  print('Starting bot...')
  
  app = Application.builder().token(TOKEN).build()
  
  # Commands
  app.add_handler(CommandHandler('start', start_command))
  app.add_handler(CommandHandler('help', help_command))
  app.add_handler(CommandHandler('predict', predict_command))
  
  # Messages
  app.add_handler(MessageHandler(filters.TEXT, handle_message))
  app.add_handler(MessageHandler(filters.PHOTO, handle_image))
  
  # ERRORS
  
  app.add_error_handler(error)
  
  print('Polling')
  app.run_polling(poll_interval=3)
  