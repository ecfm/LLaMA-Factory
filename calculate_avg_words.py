import json

def calculate_average_words():
    # Load the JSON file
    with open('data/ft_risky_AB_converted.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    total_words = 0
    total_messages = 0
    
    # Iterate through each conversation
    for conversation in data:
        messages = conversation.get('messages', [])
        for message in messages:
            # Check if it's a user message
            if message.get('role') == 'user':
                # Count words in the content
                content = message.get('content', '')
                words = len(content.split())
                total_words += words
                total_messages += 1
    
    # Calculate average
    if total_messages > 0:
        average_words = total_words / total_messages
        print(f"Total user messages: {total_messages}")
        print(f"Total words in user messages: {total_words}")
        print(f"Average words per user message: {average_words:.2f}")
    else:
        print("No user messages found.")

if __name__ == "__main__":
    calculate_average_words() 