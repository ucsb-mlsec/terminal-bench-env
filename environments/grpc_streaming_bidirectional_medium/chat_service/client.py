import grpc
import chat_pb2
import chat_pb2_grpc

def generate_messages():
    messages = ['Hello', 'How are you?', 'Goodbye']
    for content in messages:
        print(f'Sending: {content}')
        yield chat_pb2.ChatMessage(content=content)
    # BUG: Missing proper signaling that we're done sending
    # Should call something to close the sending side of the stream

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = chat_pb2_grpc.ChatServiceStub(channel)
    responses = stub.Chat(generate_messages())
    for response in responses:
        print(f'Received: {response.content}')
    # The iteration will hang because server doesn't know client is done sending

if __name__ == '__main__':
    run()