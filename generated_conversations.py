import os
import openai
import time
import os
import shutil

# Set up the API key
openai.api_key = ""

# Set parameters
TEMPERATURE = 0.5
MAX_TOKENS = 800
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
NUM_CONVERSATIONS = 500
output_dir_depression = "dataset/generated_conversations_depression_PHQ-8"
output_dir_no_depression = "dataset/generated_conversations_no_depression_PHQ-8"

if not os.path.exists(output_dir_depression):
    os.makedirs(output_dir_depression)

if not os.path.exists(output_dir_no_depression):
    os.makedirs(output_dir_no_depression)

def get_response(prompt):
    messages = [
        { "role": "system", "content": prompt },
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )

    return completion.choices[0].message.content.strip()

for i in range(NUM_CONVERSATIONS):
    time.sleep(15) # Pause 15 seconds
    prompt = (
        f"Generate a detailed and specific conversation between a doctor and a patient discussing depression symptoms. "
        f"The doctor asks questions to understand the patient's condition better, while the patient provides information about their symptoms, experiences, and personal background. "
        f"The focus of the conversations should be patient's describing their symtoms. The greetings should be skipped. "
        f"The procedure should follow PHQ-8 assessmeent. The questions should be similar to the ones in PHQ-8."
        f"At the end, the patient is diagnosed as having depression. "
        f"Ensure the conversation is medically accurate, contains personal information about the patient's life, and demonstrates the impact of depression on their daily functioning."
    )

    generated_conversation = get_response(prompt)

    with open(os.path.join(output_dir_depression, f"conversation_{i+1}.txt"), "w") as f:
        f.write(generated_conversation)

    print(f"Generated conversation {i+1}")

for i in range(NUM_CONVERSATIONS):
    time.sleep(15) # Pause 15 seconds
    prompt = (
        f"Generate a detailed and specific conversation between a doctor and a patient discussing depression symptoms. "
        f"The doctor asks questions to understand the patient's condition better, while the patient provides information about their symptoms, experiences, and personal background. "
        f"The focus of the conversations should be patient's describing their symtoms. The greetings should be skipped. "
        f"The procedure should follow PHQ-8 assessmeent.  The questions should be similar to the ones in PHQ-8."
        f"At the end, the patient is diagnosed as not having depression. "
        f"Ensure the conversation is medically accurate, contains personal information about the patient's life, and demonstrates the impact of depression on their daily functioning."
    )

    generated_conversation = get_response(prompt)

    with open(os.path.join(output_dir_no_depression, f"conversation_{i+1}.txt"), "w") as f:
        f.write(generated_conversation)

    print(f"Generated conversation {i+1}")
    


