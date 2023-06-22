# GGML Roleplay Discord Bot README
A Roleplaying Discord Bot for GGML LLMs

## Overview
GGML Roleplay Discordbot is a software project for running GGML formatted Large Language Models such as [NousResearch's Nous-Hermes-13B GGML](https://huggingface.co/TheBloke/Nous-Hermes-13B-GGML) as a roleplaying discord bot. The bot is designed to run locally on a PC with as little as 8GB of VRAM. The bot listens for messages mentioning its username, replying to its messages, or any DMs it receives, processes the message content, and generates a response based on the input.

PREFERRED: NVIDIA GPU with at least 12GB of VRAM for 7B model, and 24GB of VRAM for 13B models

Good alternatives are [OpenAccess AI Collective's Manticore 13B Chat GGML](https://huggingface.co/TheBloke/manticore-13b-chat-pyg-GGML)
and [PocketDoc/Dans-PersonalityEngine-13b-ggml-q4_0](https://huggingface.co/PocketDoc/Dans-PersonalityEngine-13b-ggml-q4_0)

This bot differs from my other repository, [Alpaca-Discord](https://github.com/teknium1/alpaca-discord) in a few ways.
The primary difference is that it offers character role playing and chat history. You can set the chat history to anything you like with !limit, but the LLAMA-based models can only handle 2,000 tokens of input for any given prompt, so be sure to set it low if you have a large character card.

This bot utilizes a json file some may know as a character card, to place into its preprompt, information about the character it is to role play as.
You can manually edit the json or use a tool like [Teknium's Character Creator](https://teknium1.github.io/charactercreator/index.html) or [AI Character Editor](https://zoltanai.github.io/character-editor/) to make yourself a character card.
For now, we only support one character at a time, and the active character card file should be specified in `config.yml`.  The default character is ChatBot from `character.json`.

Finally, this bot now [supports a range of quantized GGML models](https://github.com/marella/ctransformers#supported-models) beyond LLaMA-based models to run on CPU and GPU. Currently only LLaMA models have GPU support.

I am definitely open to Pull Requests and other contributions if anyone who likes the bot wants to collaborate on adding new features, making it more robust, etc.

Example:
![image](https://user-images.githubusercontent.com/127238744/228260843-f623d17a-fb0c-4289-ab59-eae1e676b4b7.png)


## Dependencies
You must have either the Hermes model (or theoretically any fine tuned supported model) in GGML format.
Currently I can only recommend Hermes or other LLaMA models that support Alpaca style prompts, with models that don't support Alpaca prompts, the preprompt would likely need to be reconfigured.

To run the bot, you need the following Python packages:
- `discord`
- `ctransformers`

For CPU only inference, you can install discord and ctransformers using pip:

```sh
pip install discord
pip install ctransformers
```

For GPU (CUDA) support, set environment variable `CT_CUBLAS=1` and install from source using:

```sh
pip install discord
CT_CUBLAS=1 pip install ctransformers --no-binary ctransformers
```

<details>
<summary><strong>Show commands for Windows</strong></summary><br>

On Windows PowerShell run:

```sh
py -m pip install discord
$env:CT_CUBLAS=1
py -m pip install ctransformers --no-binary ctransformers
```

On Windows Command Prompt run:

```sh
py -m pip install discord
set CT_CUBLAS=1
py -m pip install ctransformers --no-binary ctransformers
```

</details>


## How the bot works
The bot uses the `discord.py` library for interacting with Discord's API and the `ctransformers` library for loading and using the Large Language Model.

1. It creates a Discord client with the default intents and sets the `members` intent to `True`.
2. It loads the GGML model from the current directory.
3. It initializes a queue to manage incoming messages mentioning the bot.
4. It listens for messages and adds them to the queue if the bot is mentioned.
5. If the bot is mentioned, the roleplaying character card as well as the last N messages (that you set) are sent above your prompt to the model.
6. It then processes the queue to generate responses based on the text.
7. It sends the generated response to the channel where the original message was sent. 

## How to run the bot
1. Ensure you have the required dependencies installed.
2. Copy `config.yml.example` to `config.yml`.
3. [Create a Discord bot account](https://discordpy.readthedocs.io/en/stable/discord.html) and obtain its Token. Put your Token in the `discord` entry in `config.yml`.
4. Enable all the Priviliged Gateway Intents in the bot account.  Ignore the Bot Permissions section.
4. Make sure the model is stored in the directory specified by the relevant `model_path` entry in `config.yml` - it should be GGML format.
5. Run the script using Python:
`python roleplay-bot.py` or `py roleplay-bot.py`
6. Invite the bot to your Discord server by generating a URL in the Discord developer portal.
7. Mention the bot in a message or dm the bot directly to receive a response generated by the Large Language Model.

## Customization options
You can customize parameters in the script to change the behavior of the bot:

- `max_new_tokens`: Set the maximum number of new tokens the model should generate in its response.
- `repetition_penalty`: Set a penalty value for repeating tokens. Default is `1.1`.
- `temperature`: Set the sampling temperature. Default is `0.8`.
- `top_p`: Set the cumulative probability threshold for nucleus sampling. Default is `0.95`.
- `top_k`: Set the number of tokens to consider for top-k sampling. Default is `40`.
- `message_history_limit`: set this to the default number of previous chat messages for the bot to look at each response it makes.

More parameters are detailed in the [C Transformers documentation](https://github.com/marella/ctransformers#config)

## Credits, License, Etc.
While my repo may be licensed as MIT, the underlying code, libraries, and other portions of this repo may not be. Please DYOR to check what can
and can't be used and in what ways it may or may not be. If anyone can help me with proper attributions or licensing placement, please submit a PR

This would not be possible without the people of Facebook's Research Team: FAIR, and with Stanford's Research:
<pre>
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
</pre>

@marella - https://github.com/marella - For the MIT licenced C Transformers and ChatDocs

@Ristellise - https://github.com/Ristellise - For converting the code to be fully async and non-blocking

@Main - https://twitter.com/main_horse - for helping with getting the initial inferencing code working

You can find me on Twitter - @Teknium1 - https://twitter.com/Teknium1

Final thanks go to GPT-4, for doing much of the heavy lifting for creating both the original discord bot code, and writing a large portion of this readme! 
