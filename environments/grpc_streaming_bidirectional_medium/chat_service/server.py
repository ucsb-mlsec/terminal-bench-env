import grpc
from concurrent import futures
import chat_pb2
import chat_pb2_grpc


class ChatServiceServicer(chat_pb2_grpc.ChatServiceServicer):
    def Chat(self, request_iterator, context):
        for message in request_iterator:
            print(f'Received: {message.content}')
            response = chat_pb2.ChatMessage(content=f'Echo: {message.content}')
            print(f'Sending: {response.content}')
            # BUG: Missing 'yield response' here - response is created but never sent back


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    print('Server starting on port 50051...')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()