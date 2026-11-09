# phase1_test.py
import ollama

def test_ai():
    print("Asking the local AI a question...\n")
    
    # Send a message to the Llama 3.2 model running on your computer
    response = ollama.chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': 'Explain what a virtual environment is in Python in one simple sentence.',
        },
    ])
    
    # Print the reply
    print("AI Reply:")
    print(response['message']['content'])

if __name__ == "__main__":
    test_ai()