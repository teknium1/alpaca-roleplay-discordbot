import re, discord, asyncio, json, yaml
from concurrent.futures import ThreadPoolExecutor
from discord.ext import commands
from ctransformers import AutoModelForCausalLM
from pathlib import Path
from typing import Any, Dict, Optional, Union


intents = discord.Intents.default()
intents.members = True

# Load the configuration
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

model_config_name = config.pop("llm")
model_config = {**config[model_config_name]}

class Chatbot:
    def __init__(self):
        self.message_history_limit = 5
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id=model_config.pop("model_path"),
            model_file=model_config.pop("model_file"),
            model_type=model_config.pop("model_type"),
            context_length=model_config.pop("context_length"),
            gpu_layers=model_config.pop("gpu_layers"),
            threads=model_config.pop("threads"),
            local_files_only=True
        )


chatbot = Chatbot()
queue = asyncio.Queue()
bot = commands.Bot(command_prefix='!', intents=intents)
character = config.pop("character")

def replace_mentions_with_usernames(content, message):
    for mention in re.finditer(r'<@!?(\d+)>', content):
        user_id = int(mention.group(1))

        if message.guild is not None:  # Handle server messages
            member = discord.utils.get(message.guild.members, id=user_id)
            if member:
                content = content.replace(mention.group(0), f"@{member.display_name}")
        else:  # Handle DMs
            user = bot.get_user(user_id)
            if user:
                content = content.replace(mention.group(0), f"@{user.name}")

    return content

@bot.command()
@commands.is_owner()
async def setlimit(ctx, limit: int):
    chatbot.message_history_limit = limit
    await ctx.send(f'Message history limit set to {limit}')

@setlimit.error
async def setlimit_error(ctx, error):
    if isinstance(error, commands.NotOwner):
        await ctx.send('You do not have permission to use this command.')
    elif isinstance(error, commands.MissingRequiredArgument) or isinstance(error, commands.BadArgument):
        await ctx.send('Invalid command. Usage: !setlimit <number>')

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    asyncio.get_running_loop().create_task(background_task())

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)

    if isinstance(message.channel, discord.channel.DMChannel) or (bot.user and bot.user.mentioned_in(message)):
        if message.reference:
            pastMessage = await message.channel.fetch_message(message.reference.message_id)
        else:
            pastMessage = None
        past_messages = await fetch_past_messages(message.channel)
        await queue.put((message, pastMessage, past_messages))

async def fetch_past_messages(channel):
    global chatbot
    messages = []
    async for message in channel.history(limit=chatbot.message_history_limit):
        content = message.content
        if not isinstance(channel, discord.channel.DMChannel):
            for mention in re.finditer(r'<@!?(\d+)>', content):
                user_id = int(mention.group(1))
                member = discord.utils.get(message.guild.members, id=user_id)
                if member:
                    content = content.replace(mention.group(0), f"@{member.name}")
        messages.append((message.author.display_name, content))
    return messages

async def background_task():
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    print("Task Started. Waiting for inputs.")
    while True:
        msg_pair: tuple[discord.Message, discord.Message, list] = await queue.get()
        msg, past, past_messages = msg_pair

        message_content = msg.author.display_name + ": " + replace_mentions_with_usernames(msg.content, msg)
        past_content = None
        if past:
            past_user = past.author.display_name
            past_content = past_user + ": " + replace_mentions_with_usernames(past.content, past)
        text = generate_prompt(message_content, past_content, past_messages)
        response = await loop.run_in_executor(executor, sync_task, text)
        print(f"Response: {text}\n{response}")
        
        if not response.strip():
            response = "Sorry, I didn't understand that.  Could you please clarify your statement."

        try:
            await msg.reply(response, mention_author=False)
        except discord.errors.Forbidden:
            print("Error: Missing Permissions")
            await msg.channel.send("Retry")

def sync_task(message):
    global chatbot
    input_ids = chatbot.model.tokenize(message)
    print(f"Input tokens: {input_ids}")  # Log input tokens

    generated_ids = chatbot.model.generate(input_ids, repetition_penalty=1.1, temperature=0.28, top_p=0.95, top_k=40, reset=False)

    # To get the tokens generated by the bot
    generated_tokens = []

    for token in generated_ids:
        generated_tokens.append(token)
    
    print(f"Generated tokens: {generated_tokens}")  # Log generated tokens

    response = chatbot.model.detokenize(generated_tokens).replace("</s>", "")
    return response

def generate_prompt(text, pastMessage, past_messages, character_json_path=character):
    global chatbot
    max_token_limit = 2000
    chat_history = ""
    token_count = 0

    with open(character_json_path, 'r') as f:
        character_data = json.load(f)

    name = character_data.get('name', '')
    background = character_data.get('description', '')
    personality = character_data.get('personality', '')
    circumstances = character_data.get('world_scenario', '')
    common_greeting = character_data.get('first_mes', '')

    for username, message in past_messages:
        message_text = f"{username}: {message}\n"
        message_tokens = len(chatbot.model.tokenize(message_text))
        if token_count + message_tokens > max_token_limit:
            break
        chat_history = message_text + chat_history
        token_count += message_tokens

    if pastMessage:
        return f"""### Instruction:
Respond to the following message as your character would: 
### Input:
{text}
### Response:
{name}:"""
    else:
        return f"""### Instruction:
Role play as character that is described in the following lines. You always stay in character.
{"Your name is " + name + "." if name else ""}
{"Your backstory and history are: " + background if background else ""}
{"Your personality is: " + personality if personality else ""}
{"Your current circumstances and situation are: " + circumstances if circumstances else ""}
{"Your common greetings are: " + common_greeting if common_greeting else ""}
Remember, you always stay on character. You are the character described above.

Always speak with new and unique messages that haven't been said in the chat history.

Respond to this message as your character would:
### Input:
{text}
### Response:
{name}:"""

# Get the Discord API key
key = config.pop("discord")

bot.run(key)
